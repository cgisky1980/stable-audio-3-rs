# stable-audio-3-rs

Stable Audio 3 inference in Rust, powered by MNN CUDA + ONNX Runtime.

## Overview

This project provides a high-performance inference pipeline for [Stable Audio 3](https://huggingface.co/collections/stabilityai/stable-audio-3) with MNN CUDA backend. It achieves real-time audio generation on consumer GPUs through CUDA-accelerated MNN models with full INT8 quantization and chunked decoding.

Pre-converted MNN models (INT8 and FP16) are available at [cgisky/stable-audio-3-mnn](https://huggingface.co/cgisky/stable-audio-3-mnn).

Our MNN patches (Softmax fix, MatMul precision fix, Windows build fixes) are maintained at [cgisky1980/MNN](https://github.com/cgisky1980/MNN).

## Performance

Tested on RTX 2080 Ti (22 GB) + Ryzen 9 5900X, 8 diffusion steps:

### INT8 Full Pipeline (`--mnn-int8`)

| Variant | Duration | T5 | NC | DiT | Decoder | Total | RTF | VRAM |
|---------|----------|-----|-----|------|---------|-------|-----|------|
| Music | 10s | 0.59s | 0.01s | 0.22s | 1.25s | **2.07s** | **4.8x** | ~1.6 GB |
| Music | 30s | 0.55s | 0.01s | 0.32s | 0.68s | **1.56s** | **19.2x** | ~1.6 GB |
| Music | 60s | 0.58s | 0.01s | 0.50s | 1.03s | **2.13s** | **28.2x** | ~1.6 GB |
| Music | 120s | 0.58s | 0.01s | 1.01s | 2.05s | **3.68s** | **32.7x** | ~1.6 GB |
| SFX | 10s | 0.93s | 0.01s | 0.27s | 1.44s | **2.65s** | **3.8x** | ~1.6 GB |
| SFX | 30s | 0.97s | 0.01s | 0.32s | 0.69s | **2.00s** | **15.0x** | ~1.6 GB |
| SFX | 60s | 0.57s | 0.01s | 0.48s | 1.02s | **2.10s** | **28.6x** | ~1.6 GB |
| SFX | 120s | 0.59s | 0.01s | 0.99s | 1.99s | **3.62s** | **33.2x** | ~1.6 GB |

> RTF = Real-Time Factor (higher is faster). VRAM = incremental GPU memory (excluding desktop baseline). Constant regardless of audio length due to chunked decoding.

### Model Size

| Model | Quantization | Size | Backend |
|-------|-------------|------|---------|
| T5Gemma | INT4 + FP16 Embed | 538 MB | MNN CPU |
| NumberConditioner | INT8 | 0.2 MB | MNN CUDA |
| DiT | INT8 | 445 MB | MNN CUDA |
| Decoder | INT8 (FusedWN) | 53 MB | MNN CUDA |
| Encoder | INT8 | 52 MB | MNN CUDA |
| **Total** | | **~1.09 GB** | |

## Architecture

```
Text Prompt → T5Gemma (MNN CPU INT4) → Text Embedding
Duration   → NumberConditioner (MNN CUDA INT8) → Duration Embedding
                                    ↓
                        DiT (MNN CUDA INT8) ← Diffusion Denoising
                                    ↓
                        Decoder (MNN CUDA INT8 FusedWN) → Audio Waveform
```

| Model | Runtime | Precision | Notes |
|-------|---------|-----------|-------|
| T5Gemma | MNN CPU | INT4+FP16 (default) / FP32 | MNN CUDA output is incorrect; `--mnn-t5-fp32` to fallback |
| NumberConditioner | MNN CUDA | FP16 / INT8 | `--mnn-int8` for INT8 |
| DiT | MNN CUDA | FP16 / INT8 | `--mnn-int8` for INT8 |
| Decoder | MNN CUDA | INT8 (FusedWN) | WeightNorm pre-fused, Softmax kernel patched |
| Encoder | MNN CUDA | FP16 / INT8 | For music-to-music mode; `--mnn-int8` for INT8 |

## Features

- **INT8 Full Pipeline**: T5 INT4 + DiT INT8 + Decoder INT8 + Encoder INT8. Model size ~1.09 GB, RTF 10-33x
- **Chunked Decoding**: Decoder processes latents in chunks of 256 timesteps, enabling pseudo-streaming output (~23.8s of audio per chunk) and constant VRAM
- **Pre-allocated Memory**: Decoder initialized with chunk_size=256 at load time, no expensive resize during inference
- **WeightNorm Pre-fusion**: Conv1d WeightNorm pre-fused into weights before conversion, avoiding FP16 precision issues
- **Music-to-Music**: Init Audio variation and Inpainting modes via Encoder + SoftNormBottleneck
- **FP16 Fallback**: All models have FP16 versions for precision-sensitive use cases

## Prerequisites

- Windows (tested on 11)
- CUDA 12.x compatible GPU
- [MNN](https://github.com/cgisky1980/MNN) built with CUDA support
- [ONNX Runtime](https://onnxruntime.ai/) (for T5 text encoder when using `--mnn-t5-fp32` fallback)

## Setup

1. Download models and pre-built DLLs from [cgisky/stable-audio-3-mnn](https://huggingface.co/cgisky/stable-audio-3-mnn). The `dll/` directory contains pre-compiled Windows DLLs (`MNN.dll` and `mnn_dit_bridge.dll`) built with CUDA support — no need to compile MNN yourself.

2. Place all model files and DLLs in your models directory.

3. Build and run:
   ```bash
   cargo build --release
   ```

<details>
<summary>Building MNN from source (optional)</summary>

If you want to build MNN yourself, see [cgisky1980/MNN](https://github.com/cgisky1980/MNN) for Windows build patches:

```bash
cmake .. -G "Visual Studio 17 2022" -A x64 \
  -DMNN_BUILD_SHARED_LIBS=ON \
  -DMNN_CUDA=ON \
  -DMNN_CUDA_NATIVE_ARCH=ON \
  -DCMAKE_BUILD_TYPE=Release
cmake --build . --config Release
```

Then build the bridge DLL:
```bash
cd bridge && cmake .. && cmake --build . --config Release
```
</details>

## Usage

### CLI

```bash
# INT8 mode (recommended, full pipeline ~1.09 GB models)
sa3-cli --prompt "ambient electronic music" --duration 30 --steps 8 --mnn --mnn-gpu 1 --mnn-int8

# FP16 mode (higher precision fallback)
sa3-cli --prompt "ambient electronic music" --duration 30 --steps 8 --mnn --mnn-gpu 1

# SFX generation
sa3-cli --variant sfx --prompt "thunder and heavy rain" --duration 30 --steps 8 --mnn --mnn-gpu 1 --mnn-int8

# Music-to-music: variation from input audio
sa3-cli --prompt "jazz piano" --duration 30 --mnn --mnn-gpu 1 --init-audio input.wav --init-noise-level 0.9

# Music-to-music: inpainting (regenerate 5s-10s)
sa3-cli --prompt "electronic beat" --duration 30 --mnn --mnn-gpu 1 --inpaint-audio input.wav --inpaint-start 5.0 --inpaint-end 10.0
```

### Library

```rust
use sa3_rs::{StableAudio3, GenerateOptions, audio};

let mut sa3 = StableAudio3::new(
    std::path::Path::new("models"),
    "sm-music",
    false,  // use_gpu (ORT CUDA)
    true,   // use_mnn
    1,      // mnn_gpu: 0=CPU, 1=CUDA, 2=Vulkan
    true,   // mnn_int8
    false,  // mnn_fp32
    false,  // mnn_t5_fp32
    256,    // t_lat
)?;

let opts = GenerateOptions {
    prompt: "ambient electronic music".to_string(),
    duration: 30.0,
    steps: 8,
    seed: Some(42),
    ..Default::default()
};

let audio = sa3.generate(&opts)?;
audio::save_audio("output.wav", &audio, opts.duration)?;
```

### CLI Options

| Option | Default | Description |
|--------|---------|-------------|
| `--variant` | `sm-music` | Model variant: `sm-music` or `sm-sfx` |
| `--prompt` | required | Text prompt |
| `--duration` | `10.0` | Audio duration in seconds |
| `--steps` | `8` | Diffusion steps |
| `--seed` | `42` | Random seed |
| `--cfg` | `1.0` | CFG guidance scale |
| `--mnn` | `false` | Enable MNN backend (all models) |
| `--mnn-gpu` | `0` | MNN device: 0=CPU, 1=CUDA, 2=Vulkan |
| `--mnn-int8` | `false` | Use INT8 models (full pipeline: T5 INT4 + DiT/Decoder/Encoder INT8) |
| `--mnn-fp32` | `false` | Use FP32 precision (highest quality) |
| `--mnn-t5-fp32` | `false` | Use FP32 T5 (1075 MB) instead of INT4 (538 MB) |
| `--init-audio` | - | Input audio for variation mode |
| `--init-noise-level` | `0.9` | Noise level for variation (0.01-1.0) |
| `--inpaint-audio` | - | Input audio for inpainting |
| `--inpaint-start` | - | Inpainting start time (seconds) |
| `--inpaint-end` | - | Inpainting end time (seconds) |
| `--output` | `output.wav` | Output file path |

## Why T5 Uses MNN CPU

The T5Gemma text encoder runs on **MNN CPU** (not CUDA) due to a critical accuracy bug in MNN's CUDA backend. MNN CUDA output for T5 has a maximum difference of 50.68 compared to CPU reference, causing completely wrong text conditioning (e.g., SFX prompts generate music instead of sound effects).

MNN CPU produces identical results to ONNX Runtime (max_diff=0.14) and is actually faster than ORT CPU (~0.15s vs ~0.26s for 10s music). Combined with INT4 quantization, the T5 model is now 538 MB (vs 2.3 GB for the FP32 ONNX version).

All other models (NC, DiT, Decoder, Encoder) work correctly on MNN CUDA with negligible differences from CPU reference.

## Related

- [cgisky/stable-audio-3-mnn](https://huggingface.co/cgisky/stable-audio-3-mnn) — Pre-converted MNN models
- [cgisky1980/MNN](https://github.com/cgisky1980/MNN) — MNN fork with CUDA bug fixes
- [alibaba/MNN](https://github.com/alibaba/MNN) — Upstream MNN
- [stabilityai/stable-audio-3](https://huggingface.co/collections/stabilityai/stable-audio-3) — Original model

## License

Stable Audio Community License — see the [model card](https://huggingface.co/collections/stabilityai/stable-audio-3) for details.