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
    log(&format!("t_lat_30s = {}", t_lat_30s));

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
    type FnSetDebug = unsafe extern "system" fn(*const std::os::raw::c_char) -> std::os::raw::c_int;
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
    let fn_set_debug: FnSetDebug = get_fn!(b"mnn_model_set_debug\0", FnSetDebug);
    let fn_destroy: FnDestroy = get_fn!(b"mnn_model_destroy\0", FnDestroy);

    for &(backend, gpu_flag, prec) in &[
        ("CPU", 0i32, 0i32),
        ("CUDA", 1i32, 0i32),
        ("CUDA-High", 1i32, 1i32),
    ] {
        log(&format!("\n=== {} ===", backend));

        let debug_path = format!(
            r"c:\work\make-muice\test\decoder_debug_{}.txt",
            backend.to_lowercase().replace('-', "_")
        );
        let debug_cstr = std::ffi::CString::new(debug_path.as_str()).unwrap();
        unsafe { fn_set_debug(debug_cstr.as_ptr()) };

        let model_path = models_dir.join("decoder_fused_wn.mnn");
        let path_cstr = std::ffi::CString::new(model_path.to_str().unwrap()).unwrap();
        let handle = unsafe { fn_create(path_cstr.as_ptr(), gpu_flag, 12, prec) };
        if handle.is_null() {
            anyhow::bail!("Failed to create model for {}", backend);
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

        unsafe { fn_destroy(handle) };

        let no_debug_cstr = std::ffi::CString::new("").unwrap();
        unsafe { fn_set_debug(no_debug_cstr.as_ptr()) };
    }

    log("\n=== 逐层精确对比 ===");

    let parse_file = |path: &str| -> Vec<(String, String, usize, f32, f32, f32, f32)> {
        let content = std::fs::read_to_string(path).unwrap_or_default();
        content
            .lines()
            .filter(|l| l.starts_with("OP|"))
            .filter_map(|l| {
                let parts: Vec<&str> = l.split('|').collect();
                if parts.len() < 8 {
                    return None;
                }
                let op_type = parts.get(1).unwrap_or(&"?").to_string();
                let op_name = parts.get(2).unwrap_or(&"?").to_string();
                let idx: usize = parts.get(3).unwrap_or(&"0").parse().ok()?;
                let mut kv = std::collections::HashMap::new();
                for p in &parts[4..] {
                    if let Some((k, v)) = p.split_once('=') {
                        kv.insert(k.to_string(), v.parse::<f32>().ok()?);
                    }
                }
                Some((
                    op_type,
                    op_name,
                    idx,
                    kv.get("abs_sum").copied().unwrap_or(0.0),
                    kv.get("min").copied().unwrap_or(0.0),
                    kv.get("max").copied().unwrap_or(0.0),
                    kv.get("count").copied().unwrap_or(0.0),
                ))
            })
            .collect()
    };

    let cpu_ops = parse_file(r"c:\work\make-muice\test\decoder_debug_cpu.txt");
    let cuda_ops = parse_file(r"c:\work\make-muice\test\decoder_debug_cuda.txt");
    let high_ops = parse_file(r"c:\work\make-muice\test\decoder_debug_cuda_high.txt");

    log(&format!(
        "CPU: {} ops, CUDA: {} ops, High: {} ops",
        cpu_ops.len(),
        cuda_ops.len(),
        high_ops.len()
    ));

    let n = cpu_ops.len().min(cuda_ops.len()).min(high_ops.len());

    log(&format!(
        "{:>4} {:>20} {:>35} {:>10} {:>10} {:>10} {:>10} {:>10} {:>10} {:>10} {:>10}",
        "#",
        "type",
        "name",
        "CPU_abs",
        "rel_N%",
        "rel_H%",
        "min_cpu",
        "min_cuda",
        "min_high",
        "maxdiff_N",
        "maxdiff_H"
    ));

    let mut first_significant = None;
    for i in 0..n {
        let (ref t, ref name, _, cpu_abs, cpu_min, cpu_max, _) = cpu_ops[i];
        let (_, _, _, cuda_abs, cuda_min, cuda_max, _) = cuda_ops[i];
        let (_, _, _, high_abs, high_min, high_max, _) = high_ops[i];

        let rel_n = if cpu_abs > 1.0 {
            (cpu_abs - cuda_abs).abs() / cpu_abs * 100.0
        } else {
            0.0
        };
        let rel_h = if cpu_abs > 1.0 {
            (cpu_abs - high_abs).abs() / cpu_abs * 100.0
        } else {
            0.0
        };
        let max_diff_n = (cpu_max - cuda_max).abs().max((cpu_min - cuda_min).abs());
        let max_diff_h = (cpu_max - high_max).abs().max((cpu_min - high_min).abs());

        let show = rel_n > 0.05 || rel_h > 0.05 || max_diff_n > 1.0 || max_diff_h > 1.0;

        if show {
            if first_significant.is_none() && (rel_n > 0.1 || max_diff_n > 5.0) {
                first_significant = Some(i);
            }
            log(&format!(
                "{:4} {:>20} {:>35} {:10.1} {:9.3}% {:9.3}% {:10.2} {:10.2} {:10.2} {:10.4} {:10.4}",
                i, t, name, cpu_abs, rel_n, rel_h, cpu_min, cuda_min, high_min, max_diff_n, max_diff_h
            ));
        }
    }

    if let Some(idx) = first_significant {
        let (ref t, ref name, _, _, _, _, _) = cpu_ops[idx];
        log(&format!(
            "\n>>> 首个显著差异: #{} type={} name={}",
            idx, t, name
        ));
    }

    Ok(())
}
