# stable-audio-3-rs

Stable Audio 3 inference in Rust, powered by MNN CUDA + ONNX Runtime.

## Overview

This project provides a high-performance inference pipeline for [Stable Audio 3](https://huggingface.co/stabilityai/stable-audio-3) with MNN CUDA backend. It achieves real-time audio generation on consumer GPUs through CUDA-accelerated MNN models and chunked decoding.

Pre-converted MNN models are available at [🤗 cgisky/stable-audio-3-mnn](https://huggingface.co/cgisky/stable-audio-3-mnn).

Our MNN patches (Softmax fix, MatMul precision fix, Windows build fixes) are maintained at [cgisky1980/MNN](https://github.com/cgisky1980/MNN).

## Performance

Tested on RTX 2080 Ti (22 GB) + Ryzen 9 5900X, 8 diffusion steps:

| Mode | Duration | Total Inference | RTF | VRAM |
|------|----------|----------------|-----|------|
| FP16 | 10s | 1.06s | **9.4x** | ~1.9 GB |
| FP16 | 30s | 1.75s | **17.0x** | ~1.9 GB |
| FP16 | 120s | 4.16s | **28.4x** | ~1.9 GB |
| INT8 | 10s | 1.10s | **9.0x** | ~1.9 GB |
| INT8 | 30s | 1.53s | **19.4x** | ~1.9 GB |
| INT8 | 120s | 4.23s | **28.0x** | ~1.9 GB |

> RTF = Real-Time Factor (higher is faster). VRAM = incremental GPU memory (excluding display baseline).

## Architecture

```
Text Prompt → T5Gemma (ORT CPU, Q8) → Text Embedding
Duration   → NumberConditioner (MNN CUDA) → Duration Embedding
                                    ↓
                        DiT (MNN CUDA FP16/INT8) ← Diffusion Denoising
                                    ↓
                        Decoder (MNN CUDA FusedWN) → Audio Waveform
```

| Model | Runtime | Precision | Notes |
|-------|---------|-----------|-------|
| T5Gemma | ORT CPU | INT8 (Q8) | MNN CUDA output is incorrect (max_diff=50.68) |
| NumberConditioner | MNN CUDA | FP16 / INT8 | |
| DiT | MNN CUDA | FP16 / INT8 | |
| Decoder | MNN CUDA | FP16 (FusedWN) | WeightNorm pre-fused, Softmax kernel patched |
| Encoder | MNN CUDA | FP16 / INT8 | For music-to-music mode only |

## Features

- **Chunked Decoding**: Decoder processes latents in chunks of 256 timesteps, enabling pseudo-streaming output (~23.8s of audio per chunk)
- **Pre-allocated Memory**: Decoder initialized with chunk_size=256 at load time, no expensive resize during inference
- **WeightNorm Pre-fusion**: Conv1d WeightNorm pre-fused into weights before conversion, avoiding FP16 precision issues
- **Music-to-Music**: Init Audio variation and Inpainting modes via Encoder + SoftNormBottleneck
- **INT8 Support**: Lower model size with minimal quality loss

## Prerequisites

- Windows (tested on 11)
- CUDA 12.x compatible GPU
- [MNN](https://github.com/cgisky1980/MNN) built with CUDA support
- [ONNX Runtime](https://onnxruntime.ai/) (for T5 text encoder)

## Setup

1. Download models from [🤗 cgisky/stable-audio-3-mnn](https://huggingface.co/cgisky/stable-audio-3-mnn)

2. Build MNN with CUDA (see [cgisky1980/MNN](https://github.com/cgisky1980/MNN) for Windows build patches):
   ```bash
   cmake .. -G "Visual Studio 17 2022" -A x64 \
     -DMNN_BUILD_SHARED_LIBS=ON \
     -DMNN_CUDA=ON \
     -DMNN_CUDA_NATIVE_ARCH=ON \
     -DCMAKE_BUILD_TYPE=Release
   cmake --build . --config Release
   ```

3. Build the bridge DLL:
   ```bash
   cd bridge && cmake .. && cmake --build . --config Release
   ```

4. Place `MNN.dll`, `mnn_dit_bridge.dll`, and model files in your models directory.

5. Build and run:
   ```bash
   cargo build --release
   ```

## Usage

### CLI

```bash
# FP16 mode (recommended)
sa3-cli --prompt "ambient electronic music" --duration 30 --steps 8 --mnn --mnn-gpu 1

# INT8 mode (lower VRAM)
sa3-cli --prompt "ambient electronic music" --duration 30 --steps 8 --mnn --mnn-gpu 1 --mnn-int8

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
    1,      // mnn_gpu: 0=CPU, 1=CUDA
    false,  // mnn_int8
    30.0,   // duration
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
| `--mnn` | `false` | Enable MNN backend |
| `--mnn-gpu` | `0` | MNN device: 0=CPU, 1=CUDA, 2=Vulkan |
| `--mnn-int8` | `false` | Use INT8 models |
| `--init-audio` | - | Input audio for variation mode |
| `--init-noise-level` | `0.9` | Noise level for variation (0.01-1.0) |
| `--inpaint-audio` | - | Input audio for inpainting |
| `--inpaint-start` | - | Inpainting start time (seconds) |
| `--inpaint-end` | - | Inpainting end time (seconds) |

## Why T5 Uses ONNX Runtime

The T5Gemma text encoder currently **must** use ONNX Runtime instead of MNN CUDA due to a critical accuracy bug in MNN's CUDA backend. MNN CUDA output for T5 has a maximum difference of 50.68 compared to CPU reference, causing completely wrong text conditioning (e.g., SFX prompts generate music instead of sound effects). All other models (NC, DiT, Decoder) work correctly on MNN CUDA.

## Related

- [🤗 cgisky/stable-audio-3-mnn](https://huggingface.co/cgisky/stable-audio-3-mnn) — Pre-converted MNN models
- [cgisky1980/MNN](https://github.com/cgisky1980/MNN) — MNN fork with CUDA bug fixes
- [alibaba/MNN](https://github.com/alibaba/MNN) — Upstream MNN
- [stabilityai/stable-audio-3](https://huggingface.co/stabilityai/stable-audio-3) — Original model

## License

Stable Audio Community License — see the [model card](https://huggingface.co/stabilityai/stable-audio-3) for details.
