use std::io::Write;
use std::path::{Path, PathBuf};

use anyhow::{anyhow, Result};
use ndarray::{Array1, Array2, Array3};
use ort::session::builder::GraphOptimizationLevel;
use ort::session::Session;
use ort::value::Tensor;
use tokenizers::Tokenizer;

use crate::config::{AUDIO_CHANNELS, IO_CHANNELS, PATCHED_CHANNELS, PATCH_SIZE, TEXT_MAX_LENGTH};

#[cfg(target_os = "windows")]
const BRIDGE_LIB_NAME: &str = "mnn_dit_bridge.dll";
#[cfg(target_os = "linux")]
const BRIDGE_LIB_NAME: &str = "libmnn_dit_bridge.so";
#[cfg(target_os = "macos")]
const BRIDGE_LIB_NAME: &str = "libmnn_dit_bridge.dylib";

fn find_bridge_lib(models_dir: &Path) -> Result<PathBuf> {
    let local = models_dir.join(BRIDGE_LIB_NAME);
    if local.exists() {
        return Ok(local);
    }

    if let Some(dir) = option_env!("MNN_LIBS_DIR") {
        let build_lib = Path::new(dir).join(BRIDGE_LIB_NAME);
        if build_lib.exists() {
            return Ok(build_lib);
        }
    }

    if let Ok(exe) = std::env::current_exe() {
        if let Some(exe_dir) = exe.parent() {
            let exe_lib = exe_dir.join(BRIDGE_LIB_NAME);
            if exe_lib.exists() {
                return Ok(exe_lib);
            }
        }
    }

    Err(anyhow!(
        "Bridge library '{BRIDGE_LIB_NAME}' not found in {} or MNN_LIBS_DIR.\n\
         Run 'cargo build' with internet access for automatic download, \
         or download MNN libs manually.",
        models_dir.display()
    ))
}

fn log(msg: &str) {
    println!("{msg}");
    let _ = std::io::stdout().flush();
}

#[repr(C)]
struct MNNModelHandle {
    _opaque: [u8; 0],
}

type FnCreate = unsafe extern "system" fn(
    *const std::os::raw::c_char,
    std::os::raw::c_int,
    std::os::raw::c_int,
    std::os::raw::c_int,
) -> *mut MNNModelHandle;
type FnResize = unsafe extern "system" fn(
    *mut MNNModelHandle,
    *const std::os::raw::c_char,
    std::os::raw::c_int,
    *const std::os::raw::c_int,
) -> std::os::raw::c_int;
type FnSetInput = unsafe extern "system" fn(
    *mut MNNModelHandle,
    *const std::os::raw::c_char,
    *const f32,
    std::os::raw::c_int,
) -> std::os::raw::c_int;
type FnSetInputI64 = unsafe extern "system" fn(
    *mut MNNModelHandle,
    *const std::os::raw::c_char,
    *const i64,
    std::os::raw::c_int,
) -> std::os::raw::c_int;
type FnResizeCommit = unsafe extern "system" fn(*mut MNNModelHandle) -> std::os::raw::c_int;
type FnRun = unsafe extern "system" fn(*mut MNNModelHandle) -> std::os::raw::c_int;
type FnGetOutput = unsafe extern "system" fn(
    *mut MNNModelHandle,
    *const std::os::raw::c_char,
    *mut f32,
    std::os::raw::c_int,
) -> std::os::raw::c_int;
type FnGetOutputDims = unsafe extern "system" fn(
    *mut MNNModelHandle,
    *const std::os::raw::c_char,
    *mut std::os::raw::c_int,
    std::os::raw::c_int,
) -> std::os::raw::c_int;
type FnDestroy = unsafe extern "system" fn(*mut MNNModelHandle);

pub(crate) struct MNNModel {
    handle: *mut MNNModelHandle,
    _lib: libloading::Library,
    fn_resize: FnResize,
    fn_resize_commit: FnResizeCommit,
    fn_set_input: FnSetInput,
    fn_set_input_i64: FnSetInputI64,
    fn_run: FnRun,
    fn_get_output: FnGetOutput,
    fn_get_output_dims: FnGetOutputDims,
    fn_destroy: FnDestroy,
}

impl MNNModel {
    pub(crate) fn load(
        models_dir: &Path,
        model_file: &str,
        use_gpu: i32,
        threads: i32,
        precision: i32,
    ) -> Result<Self> {
        let bridge_dll = find_bridge_lib(models_dir)?;
        let model_path = models_dir.join(model_file);

        let lib = unsafe {
            libloading::Library::new(&bridge_dll)
                .map_err(|e| anyhow!("Failed to load bridge DLL: {e}"))?
        };

        macro_rules! get_fn {
            ($name:expr, $ty:ty) => {
                unsafe {
                    *lib.get::<$ty>($name).map_err(|e| {
                        anyhow!(
                            "Failed to find {}: {e}",
                            std::str::from_utf8($name).unwrap_or("?")
                        )
                    })?
                }
            };
        }

        let fn_create: FnCreate = get_fn!(b"mnn_model_create\0", FnCreate);
        let fn_resize: FnResize = get_fn!(b"mnn_model_resize\0", FnResize);
        let fn_resize_commit: FnResizeCommit =
            get_fn!(b"mnn_model_resize_commit\0", FnResizeCommit);
        let fn_set_input: FnSetInput = get_fn!(b"mnn_model_set_input\0", FnSetInput);
        let fn_set_input_i64: FnSetInputI64 = get_fn!(b"mnn_model_set_input_i64\0", FnSetInputI64);
        let fn_run: FnRun = get_fn!(b"mnn_model_run\0", FnRun);
        let fn_get_output: FnGetOutput = get_fn!(b"mnn_model_get_output\0", FnGetOutput);
        let fn_get_output_dims: FnGetOutputDims =
            get_fn!(b"mnn_model_get_output_dims\0", FnGetOutputDims);
        let fn_destroy: FnDestroy = get_fn!(b"mnn_model_destroy\0", FnDestroy);

        let path_cstr = std::ffi::CString::new(
            model_path
                .to_str()
                .ok_or_else(|| anyhow!("Invalid model path"))?,
        )
        .map_err(|e| anyhow!("CString error: {e}"))?;

        let handle = unsafe { fn_create(path_cstr.as_ptr(), use_gpu, threads, precision) };
        if handle.is_null() {
            return Err(anyhow!("MNN model create failed: {}", model_file));
        }

        let backend = match use_gpu {
            1 => "CUDA",
            2 => "Vulkan",
            _ => "CPU",
        };
        let prec_label = if precision == 1 { "High" } else { "Normal" };
        log(&format!(
            "  MNN 已加载: {} (backend={}, precision={})",
            model_file, backend, prec_label
        ));

        Ok(Self {
            handle,
            _lib: lib,
            fn_resize,
            fn_resize_commit,
            fn_set_input,
            fn_set_input_i64,
            fn_run,
            fn_get_output,
            fn_get_output_dims,
            fn_destroy,
        })
    }

    pub(crate) fn resize(&self, input_name: &str, dims: &[i32]) -> Result<()> {
        let name_cstr =
            std::ffi::CString::new(input_name).map_err(|e| anyhow!("CString error: {e}"))?;
        let ret = unsafe {
            (self.fn_resize)(
                self.handle,
                name_cstr.as_ptr(),
                dims.len() as i32,
                dims.as_ptr(),
            )
        };
        if ret != 0 {
            return Err(anyhow!("MNN resize '{}' failed: {}", input_name, ret));
        }
        Ok(())
    }

    pub(crate) fn resize_commit(&self) -> Result<()> {
        let ret = unsafe { (self.fn_resize_commit)(self.handle) };
        if ret != 0 {
            return Err(anyhow!("MNN resize_commit failed: {}", ret));
        }
        Ok(())
    }

    pub(crate) fn set_input(&self, input_name: &str, data: &[f32]) -> Result<()> {
        let name_cstr =
            std::ffi::CString::new(input_name).map_err(|e| anyhow!("CString error: {e}"))?;
        let ret = unsafe {
            (self.fn_set_input)(
                self.handle,
                name_cstr.as_ptr(),
                data.as_ptr(),
                data.len() as i32,
            )
        };
        if ret != 0 {
            return Err(anyhow!("MNN set_input '{}' failed: {}", input_name, ret));
        }
        Ok(())
    }

    fn set_input_i64(&self, input_name: &str, data: &[i64]) -> Result<()> {
        let name_cstr =
            std::ffi::CString::new(input_name).map_err(|e| anyhow!("CString error: {e}"))?;
        let ret = unsafe {
            (self.fn_set_input_i64)(
                self.handle,
                name_cstr.as_ptr(),
                data.as_ptr(),
                data.len() as i32,
            )
        };
        if ret != 0 {
            return Err(anyhow!(
                "MNN set_input_i64 '{}' failed: {}",
                input_name,
                ret
            ));
        }
        Ok(())
    }

    pub(crate) fn run(&self) -> Result<()> {
        let ret = unsafe { (self.fn_run)(self.handle) };
        if ret != 0 {
            return Err(anyhow!("MNN run failed: {}", ret));
        }
        Ok(())
    }

    pub(crate) fn get_output_array3(&self, output_name: &str) -> Result<Array3<f32>> {
        let name_cstr =
            std::ffi::CString::new(output_name).map_err(|e| anyhow!("CString error: {e}"))?;

        let mut dims = [0i32; 8];
        let ndim = unsafe {
            (self.fn_get_output_dims)(self.handle, name_cstr.as_ptr(), dims.as_mut_ptr(), 8)
        };
        if ndim < 0 {
            return Err(anyhow!("MNN get_output_dims failed: {}", ndim));
        }

        let mut total = 1i32;
        for &d in dims.iter().take(ndim as usize) {
            total *= d;
        }

        let mut out = vec![0.0f32; total as usize];
        let actual = unsafe {
            (self.fn_get_output)(self.handle, name_cstr.as_ptr(), out.as_mut_ptr(), total)
        };
        if actual < 0 {
            return Err(anyhow!("MNN get_output failed: {}", actual));
        }

        if ndim != 3 {
            return Err(anyhow!("Expected 3D output, got {}D", ndim));
        }

        Array3::from_shape_vec((dims[0] as usize, dims[1] as usize, dims[2] as usize), out)
            .map_err(|e| anyhow!("Failed to reshape output: {e}"))
    }
}

impl Drop for MNNModel {
    fn drop(&mut self) {
        if !self.handle.is_null() {
            unsafe { (self.fn_destroy)(self.handle) };
        }
    }
}

unsafe impl Send for MNNModel {}

struct SoftNormBottleneck {
    scaling_factor: Array3<f32>,
    bias: Array3<f32>,
    running_std: f32,
}

impl SoftNormBottleneck {
    fn load(models_dir: &Path) -> Result<Self> {
        let params_path = models_dir.join("bottleneck_params.json");
        if !params_path.exists() {
            return Err(anyhow!("bottleneck_params.json not found"));
        }
        let json_str = std::fs::read_to_string(&params_path)
            .map_err(|e| anyhow!("Failed to read bottleneck params: {e}"))?;
        let params: serde_json::Value = serde_json::from_str(&json_str)
            .map_err(|e| anyhow!("Failed to parse bottleneck params: {e}"))?;

        let sf = params["scaling_factor"]
            .as_array()
            .ok_or_else(|| anyhow!("Missing scaling_factor"))?;
        let sf_inner = sf[0]
            .as_array()
            .ok_or_else(|| anyhow!("Invalid scaling_factor shape"))?;
        let sf_data: Vec<f32> = sf_inner
            .iter()
            .map(|v| v.as_array().unwrap()[0].as_f64().unwrap() as f32)
            .collect();
        let scaling_factor = Array3::from_shape_vec((1, IO_CHANNELS, 1), sf_data)
            .map_err(|e| anyhow!("Failed to reshape scaling_factor: {e}"))?;

        let b = params["bias"]
            .as_array()
            .ok_or_else(|| anyhow!("Missing bias"))?;
        let b_inner = b[0]
            .as_array()
            .ok_or_else(|| anyhow!("Invalid bias shape"))?;
        let b_data: Vec<f32> = b_inner
            .iter()
            .map(|v| v.as_array().unwrap()[0].as_f64().unwrap() as f32)
            .collect();
        let bias = Array3::from_shape_vec((1, IO_CHANNELS, 1), b_data)
            .map_err(|e| anyhow!("Failed to reshape bias: {e}"))?;

        let running_std = params["running_std"]
            .as_array()
            .and_then(|arr| arr[0].as_f64())
            .unwrap_or(1.0) as f32;

        log(&format!(
            "  SoftNormBottleneck 已加载 (running_std={running_std:.6})"
        ));

        Ok(Self {
            scaling_factor,
            bias,
            running_std,
        })
    }

    fn encode(&self, x: &Array3<f32>) -> Array3<f32> {
        let mut result = x * &self.scaling_factor + &self.bias;
        if self.running_std.abs() > 1e-8 {
            result.mapv_inplace(|v| v / self.running_std);
        }
        result
    }
}

fn patched_pretransform_encode(audio: &Array3<f32>) -> Array3<f32> {
    let (_batch, _channels, t_audio) = audio.dim();
    let t_padded = t_audio.div_ceil(PATCH_SIZE) * PATCH_SIZE;

    let mut padded = Array3::zeros((1, AUDIO_CHANNELS, t_padded));
    for c in 0..AUDIO_CHANNELS {
        for t in 0..t_audio.min(t_padded) {
            padded[[0, c, t]] = audio[[0, c, t]];
        }
    }

    let n_patches = t_padded / PATCH_SIZE;
    let mut result = Array3::zeros((1, PATCHED_CHANNELS, n_patches));
    for p in 0..n_patches {
        for c in 0..AUDIO_CHANNELS {
            for h in 0..PATCH_SIZE {
                result[[0, c * PATCH_SIZE + h, p]] = padded[[0, c, p * PATCH_SIZE + h]];
            }
        }
    }
    result
}

fn create_session(model_path: &Path, use_gpu: bool) -> Result<Session> {
    if use_gpu {
        match create_session_gpu(model_path) {
            Ok(s) => Ok(s),
            Err(e) => {
                log(&format!("  GPU 加载失败 ({e}), 回退 CPU"));
                create_session_cpu(model_path)
            }
        }
    } else {
        create_session_cpu(model_path)
    }
}

fn create_session_gpu(model_path: &Path) -> Result<Session> {
    let mut builder = Session::builder()
        .map_err(|e| anyhow!("Failed to create session builder: {e}"))?
        .with_optimization_level(GraphOptimizationLevel::Level1)
        .map_err(|e| anyhow!("Failed to set optimization level: {e}"))?
        .with_execution_providers([ort::ep::CUDA::default().build()])
        .map_err(|e| anyhow!("Failed to set CUDA EP: {e}"))?;
    log("  CUDA EP 已启用");
    builder
        .commit_from_file(model_path)
        .map_err(|e| anyhow!("Failed to load ONNX model {}: {e}", model_path.display()))
}

fn create_session_cpu(model_path: &Path) -> Result<Session> {
    let intra = 12;
    let inter = 12;
    log(&format!("  CPU 模式 (intra={intra}, inter={inter})"));
    Session::builder()
        .map_err(|e| anyhow!("Failed to create session builder: {e}"))?
        .with_optimization_level(GraphOptimizationLevel::Level3)
        .map_err(|e| anyhow!("Failed to set optimization level: {e}"))?
        .with_intra_threads(intra)
        .map_err(|e| anyhow!("Failed to set intra threads: {e}"))?
        .with_inter_threads(inter)
        .map_err(|e| anyhow!("Failed to set inter threads: {e}"))?
        .commit_from_file(model_path)
        .map_err(|e| anyhow!("Failed to load ONNX model {}: {e}", model_path.display()))
}

fn extract_array3_f32(
    outputs: &ort::session::SessionOutputs<'_>,
    name: &str,
) -> Result<Array3<f32>> {
    let view = outputs[name]
        .try_extract_array::<f32>()
        .map_err(|e| anyhow!("Failed to extract output '{name}': {e}"))?;
    let shape = view.shape();
    if shape.len() != 3 {
        return Err(anyhow!(
            "Expected 3D output for '{name}', got {}D",
            shape.len()
        ));
    }
    let owned = view.to_owned();
    let arr3 = owned
        .into_dimensionality::<ndarray::Ix3>()
        .map_err(|e| anyhow!("Failed to reshape output '{name}': {e}"))?;
    Ok(arr3)
}

pub struct StableAudio3Models {
    t5_mnn: Option<MNNModel>,
    nc_ort: Option<Session>,
    nc_mnn: Option<MNNModel>,
    dit_ort: Option<Session>,
    dit_mnn: Option<MNNModel>,
    dec_ort: Option<Session>,
    dec_mnn: Option<MNNModel>,
    dec_needs_resize: bool,
    encoder_mnn: Option<MNNModel>,
    bottleneck: Option<SoftNormBottleneck>,
    tokenizer: Tokenizer,
}

impl StableAudio3Models {
    #[allow(clippy::too_many_arguments)]
    pub fn load(
        models_dir: &Path,
        variant: &str,
        use_gpu: bool,
        use_mnn: bool,
        mnn_gpu: i32,
        mnn_int8: bool,
        mnn_fp32: bool,
        mnn_t5_fp32: bool,
        t_lat: usize,
    ) -> Result<Self> {
        let variant_key = variant.replace("sm-", "");
        let mnn_precision: i32 = if mnn_fp32 { 1 } else { 0 };

        if use_mnn {
            let t5_mnn = {
                let t0 = std::time::Instant::now();
                let t5_file = if mnn_t5_fp32 {
                    "text_encoder.mnn"
                } else {
                    "text_encoder_int4.mnn"
                };
                log(&format!("  加载 T5 (MNN CPU): {t5_file}..."));
                let m = MNNModel::load(models_dir, t5_file, 0, 12, mnn_precision)?;
                log(&format!(
                    "    T5 加载耗时: {:.2}s",
                    t0.elapsed().as_secs_f32()
                ));
                Some(m)
            };
            let nc = {
                let t0 = std::time::Instant::now();
                log("  加载 NC (MNN)...");
                let name = if mnn_int8 {
                    format!("number_conditioner_{variant_key}_int8.mnn")
                } else {
                    format!("number_conditioner_{variant_key}_fp16.mnn")
                };
                let m = MNNModel::load(models_dir, &name, mnn_gpu, 12, mnn_precision)?;
                log(&format!(
                    "    NC 加载耗时: {:.2}s",
                    t0.elapsed().as_secs_f32()
                ));
                m
            };
            let (dit_ort, dit_mnn) = {
                let t0 = std::time::Instant::now();
                log("  加载 DiT (MNN)...");
                let name = if mnn_int8 {
                    format!("dit_{variant_key}_int8.mnn")
                } else {
                    let p1 = models_dir.join(format!("dit_{variant_key}_fp16_f32io.mnn"));
                    if p1.exists() {
                        format!("dit_{variant_key}_fp16_f32io.mnn")
                    } else {
                        format!("dit_{variant_key}_fp16_v2_f32io.mnn")
                    }
                };
                let m = MNNModel::load(models_dir, &name, mnn_gpu, 12, mnn_precision)?;
                m.resize("x", &[1, 256, t_lat as i32])?;
                m.resize("cross_attn_cond", &[1, 257, 768])?;
                m.resize("global_embed", &[1, 768])?;
                m.resize("local_add_cond", &[1, 257, t_lat as i32])?;
                m.resize("padding_mask", &[1, t_lat as i32])?;
                m.resize_commit()?;
                log(&format!(
                    "    DiT 加载耗时: {:.2}s",
                    t0.elapsed().as_secs_f32()
                ));
                (None, Some(m))
            };
            let (dec_ort, dec_mnn) = {
                let t0 = std::time::Instant::now();
                log("  加载 Decoder (MNN FusedWN)...");
                let m = MNNModel::load(
                    models_dir,
                    "decoder_fused_wn.mnn",
                    mnn_gpu,
                    12,
                    mnn_precision,
                )?;
                m.resize("latents", &[1, 256, 256])?;
                m.resize_commit()?;
                log(&format!(
                    "    Decoder 加载耗时: {:.2}s",
                    t0.elapsed().as_secs_f32()
                ));
                (None, Some(m))
            };
            let dec_needs_resize = true;
            let (encoder_mnn, bottleneck) = {
                let enc_name = if mnn_int8 {
                    "encoder_int8.mnn"
                } else {
                    "encoder.mnn"
                };
                let enc_path = models_dir.join(enc_name);
                let bn_path = models_dir.join("bottleneck_params.json");
                if enc_path.exists() && bn_path.exists() {
                    let t0 = std::time::Instant::now();
                    log(&format!("  加载 Encoder (MNN): {enc_name}..."));
                    let m = MNNModel::load(models_dir, enc_name, mnn_gpu, 12, mnn_precision)?;
                    let bn = SoftNormBottleneck::load(models_dir)?;
                    log(&format!(
                        "    Encoder 加载耗时: {:.2}s",
                        t0.elapsed().as_secs_f32()
                    ));
                    (Some(m), Some(bn))
                } else {
                    log("  Encoder 模型未找到，跳过 (music-to-music 模式不可用)");
                    (None, None)
                }
            };
            let tokenizer = {
                let tok_path = models_dir.join("tokenizer.json");
                log(&format!("  加载 Tokenizer: {}", tok_path.display()));
                Tokenizer::from_file(&tok_path)
                    .map_err(|e| anyhow!("Failed to load tokenizer: {e}"))?
            };
            let t5_label = if mnn_t5_fp32 {
                "MNN-CPU"
            } else {
                "MNN-CPU-INT4"
            };
            let mnn_label = if mnn_int8 { "MNN-INT8" } else { "MNN-CUDA" };
            log(&format!(
                "  所有模型加载完成 (T5={t5_label}, NC/DiT/Decoder={mnn_label})"
            ));
            Ok(Self {
                t5_mnn,
                nc_ort: None,
                nc_mnn: Some(nc),
                dit_ort,
                dit_mnn,
                dec_ort,
                dec_mnn,
                dec_needs_resize,
                encoder_mnn,
                bottleneck,
                tokenizer,
            })
        } else {
            let dit_path = {
                let p1 = models_dir.join(format!("dit_{variant_key}_fp16_v2_f32io.onnx"));
                let p2 = models_dir.join(format!("dit_{variant_key}_fp16_f32io.onnx"));
                if p1.exists() {
                    p1
                } else {
                    p2
                }
            };
            let dec_path = models_dir.join("decoder_q8.onnx");
            let tok_path = models_dir.join("tokenizer.json");

            let t5 = MNNModel::load(models_dir, "text_encoder.mnn", 0, 12, 0)?;
            let nc = {
                log("  加载 NC (MNN FP16)...");
                let name = format!("number_conditioner_{variant_key}_fp16.mnn");
                MNNModel::load(models_dir, &name, 0, 12, 0)?
            };
            let dit = {
                log(&format!(
                    "  加载 DiT (ORT): {} [GPU={}]",
                    dit_path.display(),
                    use_gpu
                ));
                create_session(&dit_path, use_gpu)?
            };
            let dec = {
                log(&format!(
                    "  加载 Decoder: {} [GPU={}]",
                    dec_path.display(),
                    use_gpu
                ));
                create_session(&dec_path, use_gpu)?
            };
            let tokenizer = {
                log(&format!("  加载 Tokenizer: {}", tok_path.display()));
                Tokenizer::from_file(&tok_path)
                    .map_err(|e| anyhow!("Failed to load tokenizer: {e}"))?
            };
            log("  所有 ORT 模型加载完成 (NC=MNN FP16)");
            Ok(Self {
                t5_mnn: Some(t5),
                nc_ort: None,
                nc_mnn: Some(nc),
                dit_ort: Some(dit),
                dit_mnn: None,
                dec_ort: Some(dec),
                dec_mnn: None,
                dec_needs_resize: false,
                encoder_mnn: None,
                bottleneck: None,
                tokenizer,
            })
        }
    }

    pub fn encode_text(&mut self, text: &str) -> Result<(Array3<f32>, Array2<i64>)> {
        let (ids, mask) = if text.trim().is_empty() {
            let ids = Array2::zeros((1, TEXT_MAX_LENGTH));
            let mask = Array2::zeros((1, TEXT_MAX_LENGTH));
            (ids, mask)
        } else {
            let enc = self
                .tokenizer
                .encode(text, true)
                .map_err(|e| anyhow!("Tokenization failed: {e}"))?;
            let token_ids = enc.get_ids();
            let len = token_ids.len().min(TEXT_MAX_LENGTH);
            let mut ids = Array2::zeros((1, TEXT_MAX_LENGTH));
            let mut mask = Array2::zeros((1, TEXT_MAX_LENGTH));
            for i in 0..len {
                ids[[0, i]] = token_ids[i] as i64;
                mask[[0, i]] = 1;
            }
            (ids, mask)
        };

        if let Some(ref mnn) = self.t5_mnn {
            let ids_flat: Vec<i64> = ids.iter().copied().collect();
            let mask_flat: Vec<i64> = mask.iter().copied().collect();
            mnn.set_input_i64("input_ids", &ids_flat)?;
            mnn.set_input_i64("attention_mask", &mask_flat)?;
            mnn.run()?;
            let hidden = mnn.get_output_array3("last_hidden_state")?;
            Ok((hidden, mask))
        } else {
            Err(anyhow!("T5 model not loaded"))
        }
    }

    pub fn encode_seconds(&mut self, seconds: f32) -> Result<Array3<f32>> {
        if let Some(ref mnn) = self.nc_mnn {
            mnn.set_input("seconds", &[seconds])?;
            mnn.run()?;
            mnn.get_output_array3("embedding")
        } else if let Some(ref mut sess) = self.nc_ort {
            let arr = Array1::from_vec(vec![seconds]);
            let input = Tensor::<f32>::from_array(arr)
                .map_err(|e| anyhow!("Failed to create seconds tensor: {e}"))?;
            let outputs = sess
                .run(ort::inputs!["seconds" => input])
                .map_err(|e| anyhow!("NumberConditioner inference failed: {e}"))?;
            extract_array3_f32(&outputs, "embedding")
        } else {
            Err(anyhow!("No NC backend available"))
        }
    }

    pub fn run_dit(
        &mut self,
        x: &Array3<f32>,
        t: f32,
        cross_attn_cond: &Array3<f32>,
        global_embed: &Array2<f32>,
        local_add_cond: &Array3<f32>,
        padding_mask: &Array2<bool>,
    ) -> Result<Array3<f32>> {
        if let Some(ref mnn) = self.dit_mnn {
            let mask_f32: Vec<f32> = padding_mask
                .iter()
                .map(|&b| if b { 1.0f32 } else { 0.0f32 })
                .collect();

            mnn.set_input("x", x.as_slice().unwrap())?;
            mnn.set_input("t", &[t])?;
            mnn.set_input("cross_attn_cond", cross_attn_cond.as_slice().unwrap())?;
            mnn.set_input("global_embed", global_embed.as_slice().unwrap())?;
            mnn.set_input("local_add_cond", local_add_cond.as_slice().unwrap())?;
            mnn.set_input("padding_mask", &mask_f32)?;
            mnn.run()?;
            mnn.get_output_array3("out")
        } else if let Some(ref mut dit) = self.dit_ort {
            let t_arr = Array1::from_vec(vec![t]);
            let x_val = Tensor::<f32>::from_array(x.clone())
                .map_err(|e| anyhow!("Failed to create x tensor: {e}"))?;
            let t_val = Tensor::<f32>::from_array(t_arr)
                .map_err(|e| anyhow!("Failed to create t tensor: {e}"))?;
            let cross_val = Tensor::<f32>::from_array(cross_attn_cond.clone())
                .map_err(|e| anyhow!("Failed to create cross_attn tensor: {e}"))?;
            let global_val = Tensor::<f32>::from_array(global_embed.clone())
                .map_err(|e| anyhow!("Failed to create global_embed tensor: {e}"))?;
            let local_val = Tensor::<f32>::from_array(local_add_cond.clone())
                .map_err(|e| anyhow!("Failed to create local_add_cond tensor: {e}"))?;
            let mask_val = Tensor::<bool>::from_array(padding_mask.clone())
                .map_err(|e| anyhow!("Failed to create mask tensor: {e}"))?;

            let outputs = dit
                .run(ort::inputs![
                    "x" => x_val,
                    "t" => t_val,
                    "cross_attn_cond" => cross_val,
                    "global_embed" => global_val,
                    "local_add_cond" => local_val,
                    "padding_mask" => mask_val
                ])
                .map_err(|e| anyhow!("DiT inference failed: {e}"))?;

            extract_array3_f32(&outputs, "out")
        } else {
            Err(anyhow!("No DiT backend available"))
        }
    }

    pub fn decode(&mut self, latents: &Array3<f32>) -> Result<Array3<f32>> {
        if let Some(ref mnn) = self.dec_mnn {
            let t_lat = latents.shape()[2];
            let chunk_size = 256;
            if t_lat <= chunk_size {
                if self.dec_needs_resize {
                    let t0 = std::time::Instant::now();
                    mnn.resize("latents", &[1, 256, t_lat as i32])?;
                    mnn.resize_commit()?;
                    self.dec_needs_resize = false;
                    log(&format!(
                        "    Decoder resize 耗时: {:.2}s",
                        t0.elapsed().as_secs_f32()
                    ));
                }
                mnn.set_input("latents", latents.as_slice().unwrap())?;
                mnn.run()?;
                mnn.get_output_array3("audio")
            } else {
                let n_chunks = t_lat.div_ceil(chunk_size);
                let audio_len = t_lat * 4096;
                let mut audio_out = Array3::zeros((1, 2, audio_len));

                for ci in 0..n_chunks {
                    let start = ci * chunk_size;
                    let end = (start + chunk_size).min(t_lat);
                    let chunk_t = end - start;

                    let mut chunk_latent = Array3::zeros((1, 256, chunk_size));
                    for c in 0..256 {
                        for t in 0..chunk_t {
                            chunk_latent[[0, c, t]] = latents[[0, c, start + t]];
                        }
                    }

                    mnn.set_input("latents", chunk_latent.as_slice().unwrap())?;
                    mnn.run()?;
                    let chunk_audio = mnn.get_output_array3("audio")?;

                    let audio_start = start * 4096;
                    let audio_chunk_len = chunk_audio.shape()[2].min(chunk_t * 4096);
                    let copy_len = audio_chunk_len.min(audio_len - audio_start);
                    for ch in 0..2 {
                        for t in 0..copy_len {
                            audio_out[[0, ch, audio_start + t]] = chunk_audio[[0, ch, t]];
                        }
                    }
                }

                Ok(audio_out)
            }
        } else if let Some(ref mut sess) = self.dec_ort {
            let input = Tensor::<f32>::from_array(latents.clone())
                .map_err(|e| anyhow!("Failed to create latents tensor: {e}"))?;
            let outputs = sess
                .run(ort::inputs!["latents" => input])
                .map_err(|e| anyhow!("Decoder inference failed: {e}"))?;
            extract_array3_f32(&outputs, "audio")
        } else {
            Err(anyhow!("No Decoder backend available"))
        }
    }

    pub fn decode_chunks<F>(
        &mut self,
        latents: &Array3<f32>,
        mut on_chunk: F,
    ) -> Result<Array3<f32>>
    where
        F: FnMut(usize, usize, &Array3<f32>),
    {
        if let Some(ref mnn) = self.dec_mnn {
            let t_lat = latents.shape()[2];
            let chunk_size = 256;
            if t_lat <= chunk_size {
                if self.dec_needs_resize {
                    mnn.resize("latents", &[1, 256, t_lat as i32])?;
                    mnn.resize_commit()?;
                    self.dec_needs_resize = false;
                }
                mnn.set_input("latents", latents.as_slice().unwrap())?;
                mnn.run()?;
                let audio = mnn.get_output_array3("audio")?;
                on_chunk(1, 1, &audio);
                Ok(audio)
            } else {
                let n_chunks = t_lat.div_ceil(chunk_size);
                let audio_len = t_lat * 4096;
                let mut audio_out = Array3::zeros((1, 2, audio_len));

                for ci in 0..n_chunks {
                    let start = ci * chunk_size;
                    let end = (start + chunk_size).min(t_lat);
                    let chunk_t = end - start;

                    let mut chunk_latent = Array3::zeros((1, 256, chunk_size));
                    for c in 0..256 {
                        for t in 0..chunk_t {
                            chunk_latent[[0, c, t]] = latents[[0, c, start + t]];
                        }
                    }

                    mnn.set_input("latents", chunk_latent.as_slice().unwrap())?;
                    mnn.run()?;
                    let chunk_audio = mnn.get_output_array3("audio")?;

                    let audio_start = start * 4096;
                    let audio_chunk_len = chunk_audio.shape()[2].min(chunk_t * 4096);
                    let copy_len = audio_chunk_len.min(audio_len - audio_start);
                    let mut chunk_trimmed = Array3::zeros((1, 2, copy_len));
                    for ch in 0..2 {
                        for t in 0..copy_len {
                            audio_out[[0, ch, audio_start + t]] = chunk_audio[[0, ch, t]];
                            chunk_trimmed[[0, ch, t]] = chunk_audio[[0, ch, t]];
                        }
                    }

                    on_chunk(ci + 1, n_chunks, &chunk_trimmed);
                }

                Ok(audio_out)
            }
        } else if let Some(ref mut sess) = self.dec_ort {
            let input = Tensor::<f32>::from_array(latents.clone())
                .map_err(|e| anyhow!("Failed to create latents tensor: {e}"))?;
            let outputs = sess
                .run(ort::inputs!["latents" => input])
                .map_err(|e| anyhow!("Decoder inference failed: {e}"))?;
            let audio = extract_array3_f32(&outputs, "audio")?;
            on_chunk(1, 1, &audio);
            Ok(audio)
        } else {
            Err(anyhow!("No Decoder backend available"))
        }
    }

    pub fn encode_audio(&mut self, audio: &Array3<f32>) -> Result<Array3<f32>> {
        let encoder = self
            .encoder_mnn
            .as_mut()
            .ok_or_else(|| anyhow!("Encoder not loaded (music-to-music mode unavailable)"))?;
        let bottleneck = self
            .bottleneck
            .as_ref()
            .ok_or_else(|| anyhow!("Bottleneck not loaded"))?;

        let patched = patched_pretransform_encode(audio);
        let t_patched = patched.shape()[2] as i32;

        encoder.resize("patched_audio", &[1, PATCHED_CHANNELS as i32, t_patched])?;
        encoder.resize_commit()?;

        encoder.set_input("patched_audio", patched.as_slice().unwrap())?;
        encoder.run()?;
        let encoder_out = encoder.get_output_array3("encoder_latent")?;

        let latent = bottleneck.encode(&encoder_out);

        Ok(latent)
    }

    pub fn has_encoder(&self) -> bool {
        self.encoder_mnn.is_some() && self.bottleneck.is_some()
    }
}
