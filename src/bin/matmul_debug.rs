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
    let fn_destroy: FnDestroy = get_fn!(b"mnn_model_destroy\0", FnDestroy);

    log("=== CUDA Normal (看 MatMul debug 输出) ===");
    let model_path = models_dir.join("decoder_fused_wn.mnn");
    let path_cstr = std::ffi::CString::new(model_path.to_str().unwrap()).unwrap();
    let handle = unsafe { fn_create(path_cstr.as_ptr(), 1, 12, 0) };
    if handle.is_null() {
        anyhow::bail!("Failed to create model");
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
        "run ret={}, elapsed={:.2}s",
        ret,
        t0.elapsed().as_secs_f64()
    ));

    unsafe { fn_destroy(handle) };
    Ok(())
}
