use std::io::Write;
use std::path::Path;

use anyhow::Result;
use rand::{Rng, SeedableRng};

fn log(msg: &str) {
    println!("{msg}");
    let _ = std::io::stdout().flush();
}

fn main() -> Result<()> {
    let models_dir = Path::new(r"c:\work\make-muice\models");
    {
        let current_path = std::env::var("PATH").unwrap_or_default();
        let new_path = format!("{};{}", models_dir.display(), current_path);
        std::env::set_var("PATH", &new_path);
    }

    let t_lat_30s: usize = ((30.0_f32 + 6.0) * 44100.0 / 8192.0).ceil() as usize * 2;

    let mut rng = rand::rngs::StdRng::seed_from_u64(42);
    let latent: Vec<f32> = (0..256 * t_lat_30s)
        .map(|_| rng.random::<f32>() * 2.0 - 1.0)
        .collect();

    let bridge_dll = models_dir.join("mnn_dit_bridge.dll");
    let lib = unsafe {
        libloading::Library::new(&bridge_dll)
            .map_err(|e| anyhow::anyhow!("Failed to load bridge DLL: {e}"))?
    };

    type FnCreate = unsafe extern "system" fn(
        *const std::os::raw::c_char,
        std::os::raw::c_int,
        std::os::raw::c_int,
        std::os::raw::c_int,
    ) -> *mut std::ffi::c_void;
    type FnResize = unsafe extern "system" fn(
        *mut std::ffi::c_void,
        *const std::os::raw::c_char,
        std::os::raw::c_int,
        *const std::os::raw::c_int,
    ) -> std::os::raw::c_int;
    type FnResizeCommit = unsafe extern "system" fn(*mut std::ffi::c_void) -> std::os::raw::c_int;
    type FnSetInput = unsafe extern "system" fn(
        *mut std::ffi::c_void,
        *const std::os::raw::c_char,
        *const f32,
        std::os::raw::c_int,
    ) -> std::os::raw::c_int;
    type FnRun = unsafe extern "system" fn(*mut std::ffi::c_void) -> std::os::raw::c_int;
    type FnGetOutput = unsafe extern "system" fn(
        *mut std::ffi::c_void,
        *const std::os::raw::c_char,
        *mut f32,
        std::os::raw::c_int,
    ) -> std::os::raw::c_int;
    type FnGetOutputDims = unsafe extern "system" fn(
        *mut std::ffi::c_void,
        *const std::os::raw::c_char,
        *mut std::os::raw::c_int,
        std::os::raw::c_int,
    ) -> std::os::raw::c_int;
    type FnDestroy = unsafe extern "system" fn(*mut std::ffi::c_void);

    macro_rules! get_fn {
        ($name:expr, $ty:ty) => {
            unsafe { *lib.get::<$ty>($name).map_err(|e| anyhow::anyhow!("{e}"))? }
        };
    }

    let fn_create: FnCreate = get_fn!(b"mnn_model_create\0", FnCreate);
    let fn_resize: FnResize = get_fn!(b"mnn_model_resize\0", FnResize);
    let fn_resize_commit: FnResizeCommit = get_fn!(b"mnn_model_resize_commit\0", FnResizeCommit);
    let fn_set_input: FnSetInput = get_fn!(b"mnn_model_set_input\0", FnSetInput);
    let fn_run: FnRun = get_fn!(b"mnn_model_run\0", FnRun);
    let fn_get_output: FnGetOutput = get_fn!(b"mnn_model_get_output\0", FnGetOutput);
    let fn_get_output_dims: FnGetOutputDims =
        get_fn!(b"mnn_model_get_output_dims\0", FnGetOutputDims);
    let fn_destroy: FnDestroy = get_fn!(b"mnn_model_destroy\0", FnDestroy);

    let mut outputs: Vec<(String, Vec<f32>, Vec<usize>)> = Vec::new();

    for &(label, gpu_flag, prec) in &[
        ("cpu", 0i32, 0i32),
        ("cuda", 1i32, 0i32),
        ("cuda_high", 1i32, 1i32),
    ] {
        log(&format!("\n=== {} ===", label.to_uppercase()));

        let model_path = models_dir.join("decoder_fused_wn.mnn");
        let path_cstr = std::ffi::CString::new(model_path.to_str().unwrap()).unwrap();
        let handle = unsafe { fn_create(path_cstr.as_ptr(), gpu_flag, 12, prec) };
        if handle.is_null() {
            anyhow::bail!("Failed to create model for {}", label);
        }

        let name_cstr = std::ffi::CString::new("latents").unwrap();
        let dims = [1i32, 256, t_lat_30s as i32];
        unsafe { fn_resize(handle, name_cstr.as_ptr(), 3, dims.as_ptr()) };
        unsafe { fn_resize_commit(handle) };

        unsafe {
            fn_set_input(
                handle,
                name_cstr.as_ptr(),
                latent.as_ptr(),
                latent.len() as i32,
            )
        };

        let t0 = std::time::Instant::now();
        let ret = unsafe { fn_run(handle) };
        log(&format!(
            "  run ret={}, elapsed={:.2}s",
            ret,
            t0.elapsed().as_secs_f64()
        ));

        let out_name = std::ffi::CString::new("audio").unwrap();
        let mut out_dims = [0i32; 8];
        let ndim =
            unsafe { fn_get_output_dims(handle, out_name.as_ptr(), out_dims.as_mut_ptr(), 8) };
        let mut total = 1i32;
        let mut shape = Vec::new();
        for dim in out_dims.iter().take(ndim as usize) {
            total *= dim;
            shape.push(*dim as usize);
        }
        log(&format!("  output shape: {:?}, total={}", shape, total));

        let mut out_data = vec![0.0f32; total as usize];
        let actual =
            unsafe { fn_get_output(handle, out_name.as_ptr(), out_data.as_mut_ptr(), total) };
        log(&format!("  got {} samples", actual));

        unsafe { fn_destroy(handle) };

        outputs.push((label.to_string(), out_data, shape));
    }

    let cpu = &outputs[0];
    let cuda = &outputs[1];
    let cuda_high = &outputs[2];

    log("\n=== 输出差异分析 ===");

    let cpu_data = &cpu.1;
    let cuda_data = &cuda.1;
    let cuda_high_data = &cuda_high.1;

    let n = cpu_data
        .len()
        .min(cuda_data.len())
        .min(cuda_high_data.len());

    let mut max_diff_cuda = 0.0f32;
    let mut max_diff_high = 0.0f32;
    let mut sum_diff_cuda = 0.0f32;
    let mut sum_diff_high = 0.0f32;
    let mut big_diff_positions = Vec::new();

    for i in 0..n {
        let d_cuda = (cpu_data[i] - cuda_data[i]).abs();
        let d_high = (cpu_data[i] - cuda_high_data[i]).abs();
        if d_cuda > max_diff_cuda {
            max_diff_cuda = d_cuda;
        }
        if d_high > max_diff_high {
            max_diff_high = d_high;
        }
        sum_diff_cuda += d_cuda;
        sum_diff_high += d_high;
        if d_cuda > 0.01 {
            big_diff_positions.push((
                i,
                cpu_data[i],
                cuda_data[i],
                cuda_high_data[i],
                d_cuda,
                d_high,
            ));
        }
    }

    log(&format!("样本数: {}", n));
    log(&format!(
        "CUDA Normal: max_diff={:.6}, avg_diff={:.8}",
        max_diff_cuda,
        sum_diff_cuda / n as f32
    ));
    log(&format!(
        "CUDA High:   max_diff={:.6}, avg_diff={:.8}",
        max_diff_high,
        sum_diff_high / n as f32
    ));
    log(&format!(
        "大差异位置数 (|diff|>0.01): {}",
        big_diff_positions.len()
    ));

    if !big_diff_positions.is_empty() {
        log("\n前50个大差异位置:");
        for (i, (pos, cpu_v, cuda_v, high_v, d_cuda, d_high)) in
            big_diff_positions.iter().take(50).enumerate()
        {
            let time_sec = *pos as f32 / 44100.0;
            log(&format!("  [{:5}] pos={:>8} t={:.4}s CPU={:+.6} CUDA={:+.6} High={:+.6} diff_N={:.6} diff_H={:.6}",
                i, pos, time_sec, cpu_v, cuda_v, high_v, d_cuda, d_high));
        }

        let sample_rate = 44100usize;
        log("\n差异的周期性分析 (间隔统计):");
        let mut intervals = Vec::new();
        for i in 1..big_diff_positions.len().min(500) {
            let gap = big_diff_positions[i].0 - big_diff_positions[i - 1].0;
            intervals.push(gap);
        }
        if !intervals.is_empty() {
            intervals.sort();
            let min_gap = intervals[0];
            let max_gap = intervals[intervals.len() - 1];
            let median_gap = intervals[intervals.len() / 2];
            log(&format!(
                "  间隔: min={}, max={}, median={}",
                min_gap, max_gap, median_gap
            ));
            log(&format!(
                "  min_gap 对应频率: {:.1} Hz",
                sample_rate as f32 / min_gap as f32
            ));
            log(&format!(
                "  median_gap 对应频率: {:.1} Hz",
                sample_rate as f32 / median_gap as f32
            ));

            let mut gap_counts: std::collections::HashMap<usize, usize> =
                std::collections::HashMap::new();
            for &g in &intervals {
                *gap_counts.entry(g).or_insert(0) += 1;
            }
            let mut sorted_gaps: Vec<_> = gap_counts.iter().collect();
            sorted_gaps.sort_by(|a, b| b.1.cmp(a.1));
            log("  最常见间隔 (top 10):");
            for (gap, count) in sorted_gaps.iter().take(10) {
                let freq = sample_rate as f32 / **gap as f32;
                log(&format!(
                    "    gap={} count={} freq={:.1}Hz",
                    gap, count, freq
                ));
            }
        }
    }

    let spec = hound::WavSpec {
        channels: 1,
        sample_rate: 44100,
        bits_per_sample: 32,
        sample_format: hound::SampleFormat::Float,
    };

    for (label, data, _shape) in &outputs {
        let path = format!(r"c:\work\make-muice\test\decoder_compare_{}.wav", label);
        let mut writer = hound::WavWriter::create(&path, spec).unwrap();
        let peak = data.iter().map(|v| v.abs()).fold(0.0f32, f32::max);
        let scale = if peak > 0.001 { 0.95 / peak } else { 1.0 };
        for &s in data.iter() {
            writer.write_sample((s * scale).clamp(-1.0, 1.0)).unwrap();
        }
        writer.finalize().unwrap();
        log(&format!("Saved: {}", path));
    }

    let diff_cuda: Vec<f32> = (0..n)
        .map(|i| (cpu_data[i] - cuda_data[i]).abs() * 100.0)
        .collect();
    let diff_path = r"c:\work\make-muice\test\decoder_compare_diff_cuda.wav";
    let mut writer = hound::WavWriter::create(diff_path, spec).unwrap();
    let peak = diff_cuda.iter().map(|v| v.abs()).fold(0.0f32, f32::max);
    let scale = if peak > 0.001 { 0.95 / peak } else { 1.0 };
    for &s in diff_cuda.iter() {
        writer.write_sample((s * scale).clamp(-1.0, 1.0)).unwrap();
    }
    writer.finalize().unwrap();
    log(&format!("Saved diff: {}", diff_path));

    Ok(())
}
