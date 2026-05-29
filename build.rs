use std::env;
use std::fs;
use std::io::Read;
use std::path::PathBuf;
use std::process::Command;

fn main() {
    let target = env::var("TARGET").unwrap();

    let (platform_os, platform_backend) = match target.as_str() {
        "x86_64-pc-windows-msvc" => ("windows-latest", "cuda"),
        "x86_64-unknown-linux-gnu" => ("ubuntu-latest", "cuda"),
        "aarch64-apple-darwin" => ("macos-latest", "metal"),
        other => {
            println!(
                "cargo:warning=sa3-rs: unsupported target {other}, MNN libs not auto-downloaded"
            );
            return;
        }
    };

    let is_windows = platform_os == "windows-latest";
    let ext = if is_windows { "zip" } else { "tar.gz" };
    let asset_name = format!("mnn-libs-{platform_os}-{platform_backend}.{ext}");
    let download_url = format!(
        "https://github.com/cgisky1980/stable-audio-3-rs/releases/latest/download/{asset_name}"
    );

    let out_dir = PathBuf::from(env::var("OUT_DIR").unwrap());
    let archive_path = out_dir.join(&asset_name);

    println!("cargo:warning=Downloading MNN libs from {download_url}");

    let response = match ureq::get(&download_url)
        .set("User-Agent", "sa3-rs-build")
        .call()
    {
        Ok(r) => r,
        Err(e) => {
            println!("cargo:warning=Failed to download MNN libs: {e}");
            println!(
                "cargo:warning=Please download MNN libs manually and place in models/ directory"
            );
            return;
        }
    };

    let mut buf = Vec::new();
    if response.into_reader().read_to_end(&mut buf).is_err() {
        println!("cargo:warning=Failed to read MNN libs download");
        return;
    }
    if fs::write(&archive_path, &buf).is_err() {
        println!("cargo:warning=Failed to write MNN libs archive");
        return;
    }

    let mnn_libs_dir = out_dir.join("mnn-libs");
    let _ = fs::create_dir_all(&mnn_libs_dir);

    let extract_ok = if is_windows {
        Command::new("tar")
            .args([
                "-xf",
                archive_path.to_str().unwrap(),
                "-C",
                mnn_libs_dir.to_str().unwrap(),
            ])
            .status()
            .map(|s| s.success())
            .unwrap_or(false)
    } else {
        Command::new("tar")
            .args([
                "-xzf",
                archive_path.to_str().unwrap(),
                "-C",
                mnn_libs_dir.to_str().unwrap(),
            ])
            .status()
            .map(|s| s.success())
            .unwrap_or(false)
    };

    let _ = fs::remove_file(&archive_path);

    if extract_ok {
        println!("cargo:rustc-env=MNN_LIBS_DIR={}", mnn_libs_dir.display());
        println!(
            "cargo:warning=MNN libs downloaded to {}",
            mnn_libs_dir.display()
        );
    } else {
        println!("cargo:warning=Failed to extract MNN libs");
    }
}
