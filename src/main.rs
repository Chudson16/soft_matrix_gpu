/// soft_matrix_gpu — GPU stereo → 5.1 upmixer
///
/// Matrices:
///   default    — phase/amplitude steering (soft_matrix compatible)
///   coherence  — coherence-weighted steering
///   mvdr       — RTF/MVDR beamforming (best quality on complex material)
///   ambisonic  — 1st-order B-format intermediate then 5.1 decode
///
/// Sub-options (where applicable):
///   --wiener              upgrade coherence scaling to Wiener optimal gain
///   --gaussian            use Gaussian temporal weighting in neighbourhood
///   --gaussian-sigma N    Gaussian width (default 1.5)
///   --coherence-radius N  neighbourhood half-width in windows (default 4)
///   --widen-factor N      stereo width multiplier (default 1.0)
use clap::Parser;
use std::error::Error;
use std::path::Path;
use std::process::Command;

use symphonia::core::audio::SampleBuffer;
use symphonia::core::codecs::DecoderOptions;
use symphonia::core::formats::FormatOptions;
use symphonia::core::io::MediaSourceStream;
use symphonia::core::meta::MetadataOptions;
use symphonia::core::probe::Hint;

mod gpu;
mod matrix;

use matrix::{AmbisonicMatrix, CoherenceMatrix, DefaultMatrix, MvdrMatrix};

#[derive(Parser)]
#[command(
    name = "soft_matrix_gpu",
    about = "GPU stereo → 5.1 upmixer\nMatrices: default | coherence | mvdr | ambisonic"
)]
struct Args {
    /// Input audio file (WAV, FLAC, MP3, AAC, OGG, AIFF, Opus, …)
    source: String,

    /// Output file (.wav or .flac)
    dest: String,

    /// Output format override: wav | flac
    #[arg(long)]
    output_format: Option<String>,

    /// Matrix algorithm: default | coherence | mvdr | ambisonic
    #[arg(long, default_value = "coherence")]
    matrix: String,

    /// [coherence] Use Wiener filter optimal gain instead of raw coherence scaling
    #[arg(long, default_value = "false")]
    wiener: bool,

    /// [coherence/mvdr/ambisonic] Use Gaussian temporal weighting in neighbourhood
    #[arg(long, default_value = "false")]
    gaussian: bool,

    /// [coherence/mvdr/ambisonic] Gaussian sigma in windows (default 1.5)
    #[arg(long, default_value = "1.5")]
    gaussian_sigma: f32,

    /// [coherence/mvdr/ambisonic] Neighbourhood half-width in windows (default 4 ≈ ±185ms at 44.1kHz)
    #[arg(long, default_value = "4")]
    coherence_radius: i32,

    /// Stereo width multiplier: 1.0=normal, 2.0=horseshoe-wide
    #[arg(long, default_value = "1.0")]
    widen_factor: f32,

    /// Matrix preset for rear channel phase: default | qs | dolby
    /// (overrides --widen-factor and rear adjustment)
    #[arg(long, default_value = "default")]
    preset: String,

    /// Minimum amplitude threshold for phase steering
    #[arg(long, default_value = "0.01")]
    min_amplitude: f32,

    /// Reduce output amplitude to prevent clipping in center/LFE
    #[arg(long, default_value = "true")]
    quiet: bool,

    /// FLAC compression level 0–8
    #[arg(long, default_value = "8")]
    flac_compression: u8,
}

fn main() -> Result<(), Box<dyn Error>> {
    let args = Args::parse();

    // Output format
    let out_fmt = match &args.output_format {
        Some(f) => f.to_lowercase(),
        None => Path::new(&args.dest)
            .extension()
            .and_then(|e| e.to_str())
            .unwrap_or("wav")
            .to_lowercase(),
    };
    if out_fmt != "wav" && out_fmt != "flac" {
        eprintln!("Output format must be wav or flac (got '{}')", out_fmt);
        std::process::exit(1);
    }

    // Amplitude adjustment
    let amplitude_adjustment: f32 = if args.quiet { 1.0 } else { 0.707106781186548 };

    // Preset: widen_factor and rear_adjustment
    let (widen_factor, rear_adjustment) = match args.preset.as_str() {
        "default"  => (args.widen_factor, 1.0f32),
        "qs" | "rm" => {
            let s = 0.924f32 + 0.383f32;
            let p = (0.924f32 / s) * 2.0 - 1.0;
            (1.0f32 / p, 1.0f32)
        }
        "dolby"    => (args.widen_factor, 2.0f32.sqrt()),
        "horseshoe"=> (2.0f32, 1.0f32),
        other => {
            eprintln!("Unknown preset '{}'. Options: default | qs | dolby | horseshoe", other);
            std::process::exit(1);
        }
    };

    // Build matrix
    let matrix: Box<dyn matrix::StereoMatrix> = match args.matrix.as_str() {
        "default" => Box::new(DefaultMatrix::new(
            args.min_amplitude, widen_factor, rear_adjustment, amplitude_adjustment,
        )),
        "coherence" => Box::new(CoherenceMatrix::new(
            args.min_amplitude, widen_factor, rear_adjustment, amplitude_adjustment,
            args.coherence_radius,
            args.gaussian, args.gaussian_sigma,
            args.wiener,
        )),
        "mvdr" => Box::new(MvdrMatrix::new(
            args.min_amplitude, widen_factor, rear_adjustment, amplitude_adjustment,
            args.coherence_radius,
            args.gaussian, args.gaussian_sigma,
        )),
        "ambisonic" => Box::new(AmbisonicMatrix::new(
            args.min_amplitude, widen_factor, rear_adjustment, amplitude_adjustment,
            args.coherence_radius,
            args.gaussian, args.gaussian_sigma,
        )),
        other => {
            eprintln!(
                "Unknown matrix '{}'. Options: default | coherence | mvdr | ambisonic", other
            );
            std::process::exit(1);
        }
    };

    // Decode input
    println!("Reading {}...", args.source);
    let (left, right, sample_rate) = decode_audio(&args.source)?;
    println!(
        "  {} samples, {:.1}s at {}Hz",
        left.len(),
        left.len() as f32 / sample_rate as f32,
        sample_rate
    );

    // GPU upmix
    println!("Upmixing on GPU...");
    let channels = gpu::upmix(&left, &right, sample_rate, matrix.as_ref())?;

    // Write output
    println!("Writing {} ({})...", args.dest, out_fmt);
    match out_fmt.as_str() {
        "wav"  => write_wav(&args.dest, &channels, sample_rate)?,
        "flac" => write_flac(&args.dest, &channels, sample_rate, args.flac_compression)?,
        _      => unreachable!(),
    }
    println!("Done.");
    Ok(())
}

// ---------------------------------------------------------------------------
// Input decoding via Symphonia
// ---------------------------------------------------------------------------

fn decode_audio(path: &str) -> Result<(Vec<f32>, Vec<f32>, u32), Box<dyn Error>> {
    let file = std::fs::File::open(path)?;
    let mss  = MediaSourceStream::new(Box::new(file), Default::default());

    let mut hint = Hint::new();
    if let Some(ext) = Path::new(path).extension().and_then(|e| e.to_str()) {
        hint.with_extension(ext);
    }

    let probed = symphonia::default::get_probe().format(
        &hint, mss,
        &FormatOptions::default(),
        &MetadataOptions::default(),
    )?;

    let mut format = probed.format;
    let track = format.default_track().ok_or("No audio track found")?;

    let sample_rate  = track.codec_params.sample_rate.ok_or("No sample rate")?;
    let num_channels = track.codec_params.channels.map(|c| c.count()).unwrap_or(2);
    let track_id     = track.id;

    let mut decoder = symphonia::default::get_codecs()
        .make(&track.codec_params, &DecoderOptions::default())?;

    let mut left  = Vec::new();
    let mut right = Vec::new();

    loop {
        let packet = match format.next_packet() {
            Ok(p)  => p,
            Err(symphonia::core::errors::Error::ResetRequired) => continue,
            Err(_) => break,
        };
        if packet.track_id() != track_id { continue; }

        let decoded = match decoder.decode(&packet) {
            Ok(d)  => d,
            Err(_) => continue,
        };

        let spec = *decoded.spec();
        let mut sample_buf = SampleBuffer::<f32>::new(decoded.capacity() as u64, spec);
        sample_buf.copy_interleaved_ref(decoded);

        let ch = num_channels.max(1);
        for frame in sample_buf.samples().chunks(ch) {
            left.push(frame[0]);
            right.push(if ch > 1 { frame[1] } else { frame[0] });
        }
    }

    if left.is_empty() {
        return Err("No samples decoded — check the file is valid stereo/mono".into());
    }
    Ok((left, right, sample_rate))
}

// ---------------------------------------------------------------------------
// WAV output
// ---------------------------------------------------------------------------

fn write_wav(path: &str, channels: &[Vec<f32>; 6], sample_rate: u32) -> Result<(), Box<dyn Error>> {
    let spec = hound::WavSpec {
        channels: 6, sample_rate,
        bits_per_sample: 32,
        sample_format: hound::SampleFormat::Float,
    };
    let mut writer = hound::WavWriter::create(path, spec)?;
    for i in 0..channels[0].len() {
        for ch in channels.iter() { writer.write_sample(ch[i])?; }
    }
    writer.finalize()?;
    Ok(())
}

// ---------------------------------------------------------------------------
// FLAC output via flac CLI
// ---------------------------------------------------------------------------

fn write_flac(
    path: &str, channels: &[Vec<f32>; 6],
    sample_rate: u32, compression: u8,
) -> Result<(), Box<dyn Error>> {
    let temp_wav = format!("{}.tmp_encode.wav", path);
    let spec = hound::WavSpec {
        channels: 6, sample_rate,
        bits_per_sample: 24,
        sample_format: hound::SampleFormat::Int,
    };
    let mut writer = hound::WavWriter::create(&temp_wav, spec)?;
    let scale = (1i32 << 23) as f32;
    for i in 0..channels[0].len() {
        for ch in channels.iter() {
            let s = ((ch[i].clamp(-1.0, 1.0) * scale) as i32)
                .clamp(-(1i32 << 23), (1i32 << 23) - 1);
            writer.write_sample(s)?;
        }
    }
    writer.finalize()?;

    let level  = format!("-{}", compression.min(8));
    let status = Command::new("flac")
        .args([&level, "--silent", "--force", &temp_wav, "-o", path])
        .status();
    let _ = std::fs::remove_file(&temp_wav);

    match status {
        Ok(s) if s.success() => Ok(()),
        Ok(s) => Err(format!("flac exited with status {}", s).into()),
        Err(e) => Err(format!("Could not run `flac`: {}", e).into()),
    }
}
