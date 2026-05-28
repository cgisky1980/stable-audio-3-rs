use std::io::Write;
use std::path::Path;

use anyhow::Result;
use ndarray::Array3;

fn log(msg: &str) {
    println!("{msg}");
    let _ = std::io::stdout().flush();
}

struct BridgeLib {
    _lib: libloading::Library,
    fn_create: unsafe extern "system" fn(
        *const std::os::raw::c_char,
        std::os::raw::c_int,
        std::os::raw::c_int,
        std::os::raw::c_int,
    ) -> *mut std::ffi::c_void,
    fn_resize: unsafe extern "system" fn(
        *mut std::ffi::c_void,
        *const std::os::raw::c_char,
        std::os::raw::c_int,
        *const std::os::raw::c_int,
    ) -> std::os::raw::c_int,
    fn_resize_commit: unsafe extern "system" fn(*mut std::ffi::c_void) -> std::os::raw::c_int,
    fn_set_input: unsafe extern "system" fn(
        *mut std::ffi::c_void,
        *const std::os::raw::c_char,
        *const f32,
        std::os::raw::c_int,
    ) -> std::os::raw::c_int,
    fn_run: unsafe extern "system" fn(*mut std::ffi::c_void) -> std::os::raw::c_int,
    fn_get_output: unsafe extern "system" fn(
        *mut std::ffi::c_void,
        *const std::os::raw::c_char,
        *mut f32,
        std::os::raw::c_int,
    ) -> std::os::raw::c_int,
    fn_get_output_dims: unsafe extern "system" fn(
        *mut std::ffi::c_void,
        *const std::os::raw::c_char,
        *mut std::os::raw::c_int,
        std::os::raw::c_int,
    ) -> std::os::raw::c_int,
    fn_destroy: unsafe extern "system" fn(*mut std::ffi::c_void),
}

impl BridgeLib {
    fn load(models_dir: &Path) -> Result<Self> {
        let bridge_dll = models_dir.join("mnn_dit_bridge.dll");
        let lib = unsafe {
            libloading::Library::new(&bridge_dll)
                .map_err(|e| anyhow::anyhow!("Failed to load bridge DLL: {e}"))?
        };
        macro_rules! get_fn {
            ($name:expr, $ty:ty) => {
                unsafe { *lib.get::<$ty>($name).map_err(|e| anyhow::anyhow!("{e}"))? }
            };
        }
        Ok(Self {
            fn_create: get_fn!(b"mnn_model_create\0", _),
            fn_resize: get_fn!(b"mnn_model_resize\0", _),
            fn_resize_commit: get_fn!(b"mnn_model_resize_commit\0", _),
            fn_set_input: get_fn!(b"mnn_model_set_input\0", _),
            fn_run: get_fn!(b"mnn_model_run\0", _),
            fn_get_output: get_fn!(b"mnn_model_get_output\0", _),
            fn_get_output_dims: get_fn!(b"mnn_model_get_output_dims\0", _),
            fn_destroy: get_fn!(b"mnn_model_destroy\0", _),
            _lib: lib,
        })
    }

    fn run_decoder(
        &self,
        models_dir: &Path,
        latent: &[f32],
        t_lat: usize,
        use_gpu: i32,
    ) -> Result<Vec<f32>> {
        let model_path = models_dir.join("decoder_fused_wn.mnn");
        let path_cstr = std::ffi::CString::new(model_path.to_str().unwrap()).unwrap();
        let handle = unsafe { (self.fn_create)(path_cstr.as_ptr(), use_gpu, 12, 0) };
        if handle.is_null() {
            anyhow::bail!("Failed to create model");
        }

        let name_cstr = std::ffi::CString::new("latents").unwrap();
        let dims = [1i32, 256, t_lat as i32];
        unsafe { (self.fn_resize)(handle, name_cstr.as_ptr(), 3, dims.as_ptr()) };
        unsafe { (self.fn_resize_commit)(handle) };
        unsafe {
            (self.fn_set_input)(
                handle,
                name_cstr.as_ptr(),
                latent.as_ptr(),
                latent.len() as i32,
            )
        };

        let t0 = std::time::Instant::now();
        let ret = unsafe { (self.fn_run)(handle) };
        let elapsed = t0.elapsed().as_secs_f64();

        let out_name = std::ffi::CString::new("audio").unwrap();
        let mut out_dims = [0i32; 8];
        let ndim = unsafe {
            (self.fn_get_output_dims)(handle, out_name.as_ptr(), out_dims.as_mut_ptr(), 8)
        };
        let mut total = 1i32;
        for dim in out_dims.iter().take(ndim as usize) {
            total *= dim;
        }

        let mut out_data = vec![0.0f32; total as usize];
        unsafe { (self.fn_get_output)(handle, out_name.as_ptr(), out_data.as_mut_ptr(), total) };
        unsafe { (self.fn_destroy)(handle) };

        if ret != 0 {
            anyhow::bail!("Run failed: ret={}", ret);
        }

        log(&format!("  run: {:.2}s, {} samples", elapsed, total));
        Ok(out_data)
    }
}

fn save_wav(path: &str, data: &[f32]) {
    let spec = hound::WavSpec {
        channels: 1,
        sample_rate: 44100,
        bits_per_sample: 32,
        sample_format: hound::SampleFormat::Float,
    };
    let mut writer = hound::WavWriter::create(path, spec).unwrap();
    let peak = data.iter().map(|v| v.abs()).fold(0.0f32, f32::max);
    let scale = if peak > 0.001 { 0.95 / peak } else { 1.0 };
    for &s in data {
        writer.write_sample((s * scale).clamp(-1.0, 1.0)).unwrap();
    }
    writer.finalize().unwrap();
}

fn compute_diff(a: &[f32], b: &[f32]) -> (f32, f32, f32) {
    let n = a.len().min(b.len());
    let mut max_diff = 0.0f32;
    let mut sum_diff = 0.0f32;
    let mut big_diff_count = 0usize;
    for i in 0..n {
        let d = (a[i] - b[i]).abs();
        if d > max_diff {
            max_diff = d;
        }
        sum_diff += d;
        if d > 0.01 {
            big_diff_count += 1;
        }
    }
    let avg_diff = sum_diff / n as f32;
    (max_diff, avg_diff, big_diff_count as f32)
}

fn main() -> Result<()> {
    let models_dir = Path::new(r"c:\work\make-muice\models");
    {
        let current_path = std::env::var("PATH").unwrap_or_default();
        let new_path = format!("{};{}", models_dir.display(), current_path);
        std::env::set_var("PATH", &new_path);
    }

    let ort_lib_dir =
        r"c:\work\make-muice\onnxruntime-gpu-1.23.2\onnxruntime-win-x64-gpu-1.23.2\lib";
    {
        let current_path = std::env::var("PATH").unwrap_or_default();
        let new_path = format!("{};{}", ort_lib_dir, current_path);
        std::env::set_var("PATH", &new_path);
    }

    let test_dir = r"c:\work\make-muice\test\fallback_test";
    std::fs::create_dir_all(test_dir)?;

    let latent_path = Path::new(test_dir).join("real_latent.bin");

    let duration = 30.0f32;

    if latent_path.exists() {
        log("=== 发现已保存的 latent，直接使用 ===");
    } else {
        log("=== 步骤 0: 用完整管线生成真实 denoised latent ===");
        let ort_lib = std::path::PathBuf::from(ort_lib_dir).join("onnxruntime.dll");
        ort::init_from(&ort_lib)?.commit();

        let mut sa3 = sa3_rs::StableAudio3::new(
            models_dir, "sm-music", false, true, 1, false, false, duration,
        )?;

        use rand::{Rng, SeedableRng};
        use rand_distr::StandardNormal;
        use sa3_rs::config::{compute_latent_len, CROSS_ATTN_SEQ_LEN, IO_CHANNELS};
        use sa3_rs::schedule::build_schedule;

        let t_lat = compute_latent_len(duration);
        let mut rng = rand::rngs::StdRng::seed_from_u64(42);

        let (pos_hidden, _) = sa3
            .models
            .encode_text("gentle piano melody with soft strings")?;
        let seconds_embed = sa3.models.encode_seconds(duration)?;
        let cross_attn_cond =
            ndarray::concatenate!(ndarray::Axis(1), pos_hidden.view(), seconds_embed.view());
        let global_embed = seconds_embed.clone().remove_axis(ndarray::Axis(1));
        let local_add_cond = Array3::zeros((1, CROSS_ATTN_SEQ_LEN, t_lat));
        let padding_mask = ndarray::Array2::from_elem((1, t_lat), true);

        let schedule = build_schedule(8, t_lat);
        let mut x = Array3::from_shape_fn((1, IO_CHANNELS, t_lat), |(_, _, _)| {
            rng.sample::<f32, _>(StandardNormal)
        });

        for i in 0..8 {
            log(&format!("  去噪步骤 {}/8...", i + 1));
            let t_curr = schedule[i];
            let t_next = schedule[i + 1];
            let v = sa3.models.run_dit(
                &x,
                t_curr,
                &cross_attn_cond,
                &global_embed,
                &local_add_cond,
                &padding_mask,
            )?;
            let denoised = &x - t_curr * &v;
            if t_next > 1e-7 {
                let noise = Array3::from_shape_fn(x.dim(), |(_, _, _)| {
                    rng.sample::<f32, _>(StandardNormal)
                });
                x = (1.0 - t_next) * &denoised + t_next * noise;
            } else {
                x = denoised;
            }
        }

        let latent_flat: Vec<f32> = x.iter().copied().collect();
        log(&format!(
            "  Latent shape: {:?}, 保存到 {}",
            x.shape(),
            latent_path.display()
        ));
        let mut f = std::fs::File::create(&latent_path)?;
        let t_lat_bytes = t_lat as u64;
        f.write_all(&t_lat_bytes.to_le_bytes())?;
        f.write_all(unsafe {
            std::slice::from_raw_parts(latent_flat.as_ptr() as *const u8, latent_flat.len() * 4)
        })?;
    }

    log("=== 加载 latent ===");
    let data = std::fs::read(&latent_path)?;
    let t_lat_saved = u64::from_le_bytes(data[0..8].try_into()?) as usize;
    let latent_flat: Vec<f32> = data[8..]
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
        .collect();
    let t_lat = t_lat_saved;
    log(&format!(
        "  Latent: t_lat={}, {} values",
        t_lat,
        latent_flat.len()
    ));

    let bridge = BridgeLib::load(models_dir)?;

    log("\n=== 1. 生成 CPU 参考音频 ===");
    let ref_data = bridge.run_decoder(models_dir, &latent_flat, t_lat, 0)?;
    save_wav(&format!("{}/ref_cpu.wav", test_dir), &ref_data);

    log("=== 2. 生成全 CUDA 音频 ===");
    let _ = std::fs::remove_file("mnn_cuda_fallback_ops.txt");
    let cuda_data = bridge.run_decoder(models_dir, &latent_flat, t_lat, 1)?;
    save_wav(&format!("{}/cuda_all.wav", test_dir), &cuda_data);

    let (max_d, avg_d, big_d) = compute_diff(&ref_data, &cuda_data);
    log(&format!(
        "  CPU vs CUDA: max_diff={:.6} avg_diff={:.8} big_diff_count={:.0}",
        max_d, avg_d, big_d
    ));

    let op_types = [
        "Convolution",
        "MatMul",
        "BinaryOp",
        "UnaryOp",
        "Softmax",
        "Raster",
        "While",
        "Select",
        "Reshape",
        "Slice",
        "Concat",
        "Transpose",
    ];

    log("\n=== 3. 逐个算子禁用 CUDA 测试 ===");
    log(&format!(
        "{:20} {:>10} {:>12} {:>12} {:>10}",
        "op_type", "time(s)", "max_diff", "avg_diff", "big_diffs"
    ));
    log(&format!(
        "{:20} {:>10} {:>12} {:>12} {:>10}",
        "-------", "------", "--------", "--------", "---------"
    ));
    log(&format!(
        "{:20} {:10.2} {:12.6} {:12.8} {:10.0}",
        "CUDA_ALL", 0.0, max_d, avg_d, big_d
    ));

    for op_type in &op_types {
        std::fs::write("mnn_cuda_fallback_ops.txt", op_type)?;

        let result = bridge.run_decoder(models_dir, &latent_flat, t_lat, 1);
        let _ = std::fs::remove_file("mnn_cuda_fallback_ops.txt");

        match result {
            Ok(data) => {
                let (md, ad, bd) = compute_diff(&ref_data, &data);
                save_wav(&format!("{}/fallback_{}.wav", test_dir, op_type), &data);
                log(&format!(
                    "{:20} {:10.2} {:12.6} {:12.8} {:10.0}",
                    op_type, 0.0, md, ad, bd
                ));
            }
            Err(e) => {
                log(&format!("{:20} FAILED: {}", op_type, e));
            }
        }
    }

    log(&format!("\n测试音频保存在: {}", test_dir));
    log("请对比 ref_cpu.wav 和各 fallback_*.wav，找出哪个算子禁用后噪点消失");

    Ok(())
}
