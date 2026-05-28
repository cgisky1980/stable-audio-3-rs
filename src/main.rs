use std::io::Write;
use std::path::PathBuf;

use anyhow::Result;
use clap::Parser;

use sa3_rs::{audio, GenerateOptions, StableAudio3};

fn log(msg: &str) {
    println!("{msg}");
    let _ = std::io::stdout().flush();
}

#[derive(Parser)]
#[command(name = "sa3-cli", about = "Stable Audio 3 推理 CLI")]
struct Cli {
    #[arg(short, long, default_value = "sm-music")]
    variant: String,

    #[arg(short, long)]
    prompt: String,

    #[arg(short, long, default_value = "")]
    negative_prompt: String,

    #[arg(short, long, default_value_t = 10.0)]
    duration: f32,

    #[arg(short = 'S', long, default_value_t = 8)]
    steps: usize,

    #[arg(long, default_value_t = 1.0)]
    cfg_scale: f32,

    #[arg(short, long, default_value_t = 42)]
    seed: u64,

    #[arg(short, long, default_value = "output.wav")]
    output: String,

    #[arg(short, long, default_value = r"c:\work\make-muice\models")]
    models_dir: PathBuf,

    #[arg(
        long,
        default_value = r"c:\work\make-muice\onnxruntime-gpu-1.23.2\onnxruntime-win-x64-gpu-1.23.2\lib\onnxruntime.dll"
    )]
    ort_lib: PathBuf,

    #[arg(long, default_value_t = false)]
    gpu: bool,

    #[arg(long, default_value_t = false)]
    mnn: bool,

    #[arg(long, default_value_t = 0, value_parser = clap::value_parser!(i32).range(0..=2))]
    mnn_gpu: i32,

    #[arg(long, default_value_t = false)]
    mnn_int8: bool,

    #[arg(long, default_value_t = false)]
    mnn_fp32: bool,

    #[arg(long, default_value_t = false)]
    mnn_t5: bool,

    #[arg(long)]
    init_audio: Option<String>,

    #[arg(long, default_value_t = 0.9)]
    init_noise_level: f32,

    #[arg(long)]
    inpaint_audio: Option<String>,

    #[arg(long)]
    inpaint_start: Option<f32>,

    #[arg(long)]
    inpaint_end: Option<f32>,
}

fn main() -> Result<()> {
    let cli = Cli::parse();

    if let Some(parent) = cli.ort_lib.parent() {
        let current_path = std::env::var("PATH").unwrap_or_default();
        let new_path = format!("{};{}", parent.display(), current_path);
        std::env::set_var("PATH", &new_path);
        log(&format!("添加 DLL 目录到 PATH: {}", parent.display()));
    }

    log(&format!("加载 ONNX Runtime: {}", cli.ort_lib.display()));
    ort::init_from(&cli.ort_lib)
        .map_err(|e| {
            log(&format!("ORT init_from 失败: {e}"));
            e
        })?
        .commit();
    log("ORT 初始化完成");

    if cli.mnn {
        let mnn_dir = cli.models_dir.join("MNN.dll");
        if let Some(parent) = mnn_dir.parent() {
            let current_path = std::env::var("PATH").unwrap_or_default();
            let new_path = format!("{};{}", parent.display(), current_path);
            std::env::set_var("PATH", &new_path);
        }
        log(&format!("MNN 模式 (gpu={})", cli.mnn_gpu));
    }

    let mnn_prec = if cli.mnn_int8 { "int8" } else { "fp16" };
    log(&format!(
        "[{}] 加载模型 (目录: {}, MNN={}, precision={})...",
        cli.variant,
        cli.models_dir.display(),
        cli.mnn,
        mnn_prec
    ));
    let t0 = std::time::Instant::now();
    let mut sa3 = StableAudio3::new(
        &cli.models_dir,
        &cli.variant,
        cli.gpu,
        cli.mnn,
        cli.mnn_gpu,
        cli.mnn_int8,
        cli.mnn_fp32,
        cli.mnn_t5,
        cli.duration,
    )?;
    let t_load = t0.elapsed();

    log(&format!(
        "[{}] 生成音频: \"{}\" ({}s, {}步, CFG={})",
        cli.variant, cli.prompt, cli.duration, cli.steps, cli.cfg_scale
    ));

    let init_audio_data = if let Some(ref path) = cli.init_audio {
        Some(audio::load_audio(path, cli.duration)?)
    } else {
        None
    };

    let inpaint_audio_data = if let Some(ref path) = cli.inpaint_audio {
        Some(audio::load_audio(path, cli.duration)?)
    } else {
        None
    };

    let opts = GenerateOptions {
        prompt: cli.prompt.clone(),
        negative_prompt: cli.negative_prompt.clone(),
        duration: cli.duration,
        steps: cli.steps,
        cfg_scale: cli.cfg_scale,
        seed: Some(cli.seed),
        init_audio: init_audio_data,
        init_noise_level: cli.init_noise_level,
        inpaint_audio: inpaint_audio_data,
        inpaint_start_seconds: cli.inpaint_start,
        inpaint_end_seconds: cli.inpaint_end,
    };

    let t1 = std::time::Instant::now();
    let audio_arr = sa3.generate(&opts)?;
    let t_gen = t1.elapsed();

    let t2 = std::time::Instant::now();
    audio::save_audio(&cli.output, &audio_arr, cli.duration)?;
    let t_save = t2.elapsed();

    let total = t0.elapsed();
    log(&format!(
        "耗时统计: 模型加载={:.2}s, 推理={:.2}s, 保存={:.2}s, 总计={:.2}s",
        t_load.as_secs_f64(),
        t_gen.as_secs_f64(),
        t_save.as_secs_f64(),
        total.as_secs_f64()
    ));
    log(&format!(
        "实时率 (RTF): {:.3}x ({}s 音频 / {:.2}s 推理)",
        cli.duration / t_gen.as_secs_f32(),
        cli.duration,
        t_gen.as_secs_f64()
    ));
    log(&format!(
        "BENCH_RTF {:.4}",
        cli.duration / t_gen.as_secs_f32()
    ));
    log(&format!("已保存: {}", cli.output));

    Ok(())
}
