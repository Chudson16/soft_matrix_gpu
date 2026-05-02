/// GPU upmix pipeline with automatic chunked processing.
///
/// If the full file fits within the VRAM budget it is processed in a single
/// GPU pass (fastest).  If not, the windows are split into chunks that each
/// fit comfortably, and the overlap-add accumulation is done on the CPU
/// across chunks.  Guard windows at each chunk boundary give the coherence
/// kernel the context it needs without any audible seam.
use std::error::Error;
use std::sync::Arc;
use std::time::Instant;

use cudarc::cufft::safe::{CudaFft, FftDirection};
use cudarc::cufft::sys;
use cudarc::driver::safe::{CudaContext, CudaFunction, CudaSlice, CudaStream, LaunchConfig};
use cudarc::driver::PushKernelArg;
use cudarc::nvrtc::compile_ptx;

use crate::matrix::StereoMatrix;

// ---------------------------------------------------------------------------
// Public entry point
// ---------------------------------------------------------------------------

pub fn upmix(
    left:        &[f32],
    right:       &[f32],
    sample_rate: u32,
    matrix:      &dyn StereoMatrix,
) -> Result<[Vec<f32>; 6], Box<dyn Error>> {
    let window_size = ideal_window_size(sample_rate, 20.0);
    let hop_size    = window_size / 2;
    let pad = hop_size;
    let scale       = 1.0f32 / window_size as f32;

    println!("  Matrix  : {}", matrix.name());
    println!("  Window  : {} samples   Hop: {} samples", window_size, hop_size);

    // ---- Extract all windows on CPU ----
    let t = Instant::now();
    let hann = hann_window(window_size);
    let (left_wins, num_windows) = extract_windows(left,  &hann, hop_size);
    let (right_wins, _)          = extract_windows(right, &hann, hop_size);
    println!("  Windowing : {:.0}ms  ({} windows)", ms(t), num_windows);

    if num_windows == 0 {
        return Err("Input is too short to process".into());
    }

    // ---- GPU setup (done once, shared across all chunks) ----
    let ctx    = CudaContext::new(0)?;
    let stream = ctx.default_stream();

    let t = Instant::now();
    let ptx    = compile_ptx(matrix.kernel_src())?;
    let module = ctx.load_module(ptx)?;
    let kernel = module.load_function("steer_and_assign")?;
    println!("  Kernel compile: {:.0}ms", ms(t));

    // ---- Determine chunk size from available VRAM ----
    //
    // We need 14 float2 buffers of (num_windows × window_size) each:
    //   2 input (L freq, R freq) + 6 output freq + 6 output time domain
    //
    // Use 2/3 of total VRAM as a budget, leaving room for cuFFT work areas.
    let bytes_per_window_set = window_size * std::mem::size_of::<sys::float2>() * 14;
    let vram_budget = match ctx.total_mem() {
        Ok(total) => (total as f64 * 0.67) as usize,
        Err(_)    => 8 * 1024 * 1024 * 1024,   // fallback: 8 GB
    };
    let chunk_windows = (vram_budget / bytes_per_window_set).max(1).min(num_windows);

    // Guard: extra windows on each side of a chunk for context.
    // Must be >= context_windows() (coherence radius) and >= 1 (overlap-add smoothness).
    let guard      = matrix.context_windows().max(1);
    let num_chunks = (num_windows + chunk_windows - 1) / chunk_windows;

    println!(
        "  VRAM budget : {:.1} GB  →  {} windows/chunk  →  {} chunk(s)",
        vram_budget as f64 / 1e9,
        chunk_windows,
        num_chunks
    );

    // ---- Global output accumulation ----
    let out_samples = (num_windows.saturating_sub(1)) * hop_size + window_size;
    let mut output: [Vec<f32>; 6] = std::array::from_fn(|_| vec![0.0f32; out_samples]);
    let mut norm                  = vec![0.0f32; out_samples];

    let t_total = Instant::now();

    for chunk_idx in 0..num_chunks {
        let core_start = chunk_idx * chunk_windows;
        let core_end   = (core_start + chunk_windows).min(num_windows);
        let proc_start = core_start.saturating_sub(guard);
        let proc_end   = (core_end + guard).min(num_windows);
        let proc_n     = proc_end - proc_start;

        if num_chunks > 1 {
            print!(
                "  Chunk {}/{}: windows {}-{} ({}+{} guard)... ",
                chunk_idx + 1, num_chunks,
                core_start, core_end,
                guard, guard
            );
            let _ = std::io::Write::flush(&mut std::io::stdout());
        }

        // Slice the flat window arrays for this chunk
        let lw = &left_wins [proc_start * window_size .. proc_end * window_size];
        let rw = &right_wins[proc_start * window_size .. proc_end * window_size];

        // Run GPU pipeline on this chunk
        let chunk_time = process_chunk(
            &stream, &kernel,
            lw, rw,
            window_size, proc_n,
            matrix, sample_rate,
        )?;

        // Overlap-add: accumulate only the CORE windows (skip guard regions).
        // Guard windows were processed for context but their samples are discarded.
        let guard_before = core_start - proc_start;
        let core_n       = core_end - core_start;

        for w in 0..core_n {
            let w_local  = guard_before + w;
            let w_global = proc_start + w_local;   // = core_start + w
            let g_base   = w_global * hop_size;

            for s in 0..window_size {
                let g = g_base + s;
                if g < out_samples {
                    for ch in 0..6usize {
		    	output[ch][g] += (chunk_time[ch][w_local * window_size + s].x * scale).clamp(-4.0, 4.0);                       
                    }
                    norm[g] += hann[s];
                }
            }
        }

        if num_chunks > 1 { println!("done"); }
    }

    println!("  Total GPU+OLA : {:.0}ms", ms(t_total));

    // ---- Normalise (divide by accumulated Hann window weights) ----
    for g in 0..out_samples {
        if norm[g] > 1e-6 {
            for ch in output.iter_mut() { ch[g] /= norm[g]; }
        }
	for ch in output.iter_mut() {
	    ch[g] = ch[g].clamp(-1.0, 1.0);
	}
    }

    // out_samples already covers the padded range; trim will remove the pad
    let trim = |mut v: Vec<f32>| {
        // Remove the leading pad samples, then truncate to input length
        if v.len() > pad { v.drain(0..pad); }
        v.truncate(left.len());
        v
    };
    Ok(output.map(trim))
}

// ---------------------------------------------------------------------------
// Per-chunk GPU pipeline
// ---------------------------------------------------------------------------

fn process_chunk(
    stream:      &Arc<CudaStream>,
    kernel:      &CudaFunction,
    left_wins:   &[sys::float2],
    right_wins:  &[sys::float2],
    window_size: usize,
    num_windows: usize,
    matrix:      &dyn StereoMatrix,
    sample_rate: u32,
) -> Result<[Vec<sys::float2>; 6], Box<dyn Error>> {
    let half_window = window_size / 2;

    // ---- H → D ----
    let mut d_left:  CudaSlice<sys::float2> = stream.clone_htod(left_wins)?;
    let mut d_right: CudaSlice<sys::float2> = stream.clone_htod(right_wins)?;

    // ---- Forward FFT ----
    let fft_fwd = CudaFft::plan_1d(
        window_size as i32, sys::cufftType::CUFFT_C2C,
        num_windows as i32, stream.clone(),
    )?;
    let mut d_lf: CudaSlice<sys::float2> = stream.alloc_zeros(num_windows * window_size)?;
    let mut d_rf: CudaSlice<sys::float2> = stream.alloc_zeros(num_windows * window_size)?;
    fft_fwd.exec_c2c(&mut d_left,  &mut d_lf, FftDirection::Forward)?;
    fft_fwd.exec_c2c(&mut d_right, &mut d_rf, FftDirection::Forward)?;

    // ---- Allocate 6 output freq-domain buffers ----
    let alloc = || -> Result<CudaSlice<sys::float2>, Box<dyn Error>> {
        Ok(stream.alloc_zeros(num_windows * window_size)?)
    };
    let mut d_fl  = alloc()?;
    let mut d_fr  = alloc()?;
    let mut d_fc  = alloc()?;
    let mut d_lfe = alloc()?;
    let mut d_bl  = alloc()?;
    let mut d_br  = alloc()?;

    // ---- Steering kernel ----
    let bp           = matrix.base_params();
    let ws           = window_size as i32;
    let nw           = num_windows as i32;
    let sr           = sample_rate as f32;
    let extra_ints   = matrix.extra_int_params();
    let extra_floats = matrix.extra_float_params();
    let cfg          = LaunchConfig::for_num_elems((num_windows * half_window) as u32);

    let mut builder = stream.launch_builder(kernel);
    builder.arg(&d_lf);
    builder.arg(&d_rf);
    builder.arg(&mut d_fl);
    builder.arg(&mut d_fr);
    builder.arg(&mut d_fc);
    builder.arg(&mut d_lfe);
    builder.arg(&mut d_bl);
    builder.arg(&mut d_br);
    builder.arg(&ws);
    builder.arg(&nw);
    builder.arg(&bp.min_amplitude);
    builder.arg(&bp.widen_factor);
    builder.arg(&bp.rear_adjustment);
    builder.arg(&bp.amplitude_adjustment);
    builder.arg(&sr);
    for p in &extra_ints   { builder.arg(p); }
    for p in &extra_floats { builder.arg(p); }
    unsafe { builder.launch(cfg) }?;

    // ---- Inverse FFT (6 channels, freq → time domain) ----
    let fft_inv = CudaFft::plan_1d(
        window_size as i32, sys::cufftType::CUFFT_C2C,
        num_windows as i32, stream.clone(),
    )?;
    let mut ifft = |d: &mut CudaSlice<sys::float2>| -> Result<CudaSlice<sys::float2>, Box<dyn Error>> {
        let mut t: CudaSlice<sys::float2> = stream.alloc_zeros(num_windows * window_size)?;
        fft_inv.exec_c2c(d, &mut t, FftDirection::Inverse)?;
        Ok(t)
    };
    let d_tfl  = ifft(&mut d_fl)?;
    let d_tfr  = ifft(&mut d_fr)?;
    let d_tfc  = ifft(&mut d_fc)?;
    let d_tlfe = ifft(&mut d_lfe)?;
    let d_tbl  = ifft(&mut d_bl)?;
    let d_tbr  = ifft(&mut d_br)?;

    stream.synchronize()?;

    // ---- D → H ----
    let h_fl  = stream.clone_dtoh(&d_tfl)?;
    let h_fr  = stream.clone_dtoh(&d_tfr)?;
    let h_fc  = stream.clone_dtoh(&d_tfc)?;
    let h_lfe = stream.clone_dtoh(&d_tlfe)?;
    let h_bl  = stream.clone_dtoh(&d_tbl)?;
    let h_br  = stream.clone_dtoh(&d_tbr)?;

    Ok([h_fl, h_fr, h_fc, h_lfe, h_bl, h_br])
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

fn ms(t: Instant) -> f64 { t.elapsed().as_secs_f64() * 1e3 }

fn ideal_window_size(sample_rate: u32, lowest_hz: f32) -> usize {
    let min = (sample_rate as f32 / lowest_hz).ceil() as usize;
    let mut w = 1usize;
    while w < min { w <<= 1; }
    w
}

fn hann_window(n: usize) -> Vec<f32> {
    let tau = 2.0 * std::f32::consts::PI;
    (0..n).map(|i| 0.5 * (1.0 - (tau * i as f32 / (n - 1) as f32).cos())).collect()
}

fn extract_windows(channel: &[f32], hann: &[f32], hop_size: usize)
    -> (Vec<sys::float2>, usize)
{
    let window_size = hann.len();
    let pad = hop_size;  // half a window of silence at each end

    // Build padded channel: [pad zeros] + channel + [pad zeros]
    let mut padded = vec![0.0f32; pad + channel.len() + pad];
    padded[pad..pad + channel.len()].copy_from_slice(channel);

    let n = padded.len();
    let num_windows = if n >= window_size {
        (n - window_size) / hop_size + 1
    } else { 0 };

    let mut out = Vec::with_capacity(num_windows * window_size);
    for w in 0..num_windows {
        let start = w * hop_size;
        for s in 0..window_size {
            out.push(sys::float2 { x: padded[start + s] * hann[s], y: 0.0 });
        }
    }
    (out, num_windows)
}