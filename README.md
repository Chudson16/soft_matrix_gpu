# soft_matrix_gpu

A GPU-accelerated stereo to 5.1 surround upmixer for Linux, written in Rust.
Processes audio files in a fraction of real time using CUDA, with support for
high sample rates (up to 192kHz), multiple steering algorithms, and a wide
range of input formats.

Inspired by and based on the steering matrix math from
[soft_matrix](https://github.com/GWBasic/soft_matrix) by GWBasic, which is
an excellent CPU-based upmixer. This project reimplements the core algorithms
on the GPU for dramatically faster processing and adds several higher-quality
steering methods.

---

## Why GPU?

soft_matrix processes one FFT window per output sample sequentially on the
CPU. On a typical modern machine it runs at approximately real time — a
4-minute song takes ~4 minutes to upmix.

soft_matrix_gpu loads the entire file into GPU memory and processes all
windows simultaneously using batched cuFFT. On an NVIDIA GTX Titan XP
(and similar Pascal/Turing/Ampere cards), a 4-minute song at 44.1kHz
takes under a second of GPU compute time. 192kHz files that would be
impractical on CPU are handled automatically via chunked processing.

---

## Features

- **Multiple steering algorithms**: default, coherence, MVDR beamforming, ambisonic
- **Gaussian temporal smoothing** for all neighbourhood-based matrices
- **Wiener filter gain** as an upgrade for coherence steering
- **Automatic chunked processing** — adapts to available VRAM, handles files
  of any length including 192kHz albums
- **Multiformat input** via Symphonia: WAV, FLAC, MP3, AAC, OGG Vorbis,
  AIFF, Opus, and more
- **WAV and FLAC output**
- **Batch processing** shell script included

---

## Prerequisites

### Required

- **Linux** (tested on NixOS; should work on any Linux distribution)
- **NVIDIA GPU** with CUDA support (Pascal architecture or newer recommended)
- **NVIDIA driver** compatible with your CUDA toolkit version
- **CUDA toolkit** — version must match the `cuda-XXXXX` feature flag in
  `Cargo.toml` (see [Build](#build) below)
- **Rust** toolchain — install via [rustup](https://rustup.rs)
- **flac** CLI tool — required for FLAC output only
  (`apt install flac` / `pacman -S flac` / `nix-shell`)

### Check your CUDA version

```bash
nvcc --version
```

The output will show something like `release 12.9` — you need this number
for the build step below.

---

## Build

### NixOS (recommended)

A `shell.nix` is provided that sets up the full environment automatically:

```bash
nix-shell
cargo build --release
```

The `shell.nix` pulls in the CUDA toolkit, libcufft, and the flac encoder.
Make sure `config.allowUnfree = true` is set in your Nix configuration
(required for CUDA).

### Other Linux distributions

1. Install the CUDA toolkit for your distribution from
   [developer.nvidia.com/cuda-downloads](https://developer.nvidia.com/cuda-downloads)

2. Set the required environment variables:

```bash
export CUDA_ROOT=/usr/local/cuda
export CUDA_PATH=/usr/local/cuda
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:/run/opengl-driver:$LD_LIBRARY_PATH
```

3. Update `Cargo.toml` to match your CUDA version. Find this line:

```toml
cudarc = { version = "0.19", features = ["cufft", "nvrtc", "cuda-12090"] }
```

Replace `cuda-12090` with your version. The format is `cuda-MAJOR-MINOR-PATCH`
zero-padded to three digits per component. Examples:

| nvcc output  | Feature flag    |
|-------------|-----------------|
| release 11.8 | `cuda-11080`   |
| release 12.0 | `cuda-12000`   |
| release 12.6 | `cuda-12060`   |
| release 12.9 | `cuda-12090`   |

4. Build:

```bash
cargo build --release
```

---

## Usage

```
soft_matrix_gpu <source> <dest> [options]
```

### Basic examples

```bash
# Default coherence matrix
./target/release/soft_matrix_gpu song.flac surround.wav

# Best quality settings (recommended)
./target/release/soft_matrix_gpu song.flac surround.wav \
  --matrix mvdr \
  --gaussian \
  --gaussian-sigma 1.5 \
  --coherence-radius 6 \
  --widen-factor 1.2 \
  --min-amplitude 0.005

# Classical / acoustic music
./target/release/soft_matrix_gpu song.flac surround.wav \
  --matrix ambisonic \
  --gaussian \
  --gaussian-sigma 2.0 \
  --coherence-radius 8

# FLAC output
./target/release/soft_matrix_gpu song.flac surround.flac --matrix mvdr --gaussian
```

---

## Matrices

### `default`
Direct port of soft_matrix's DefaultMatrix. Steers each frequency bin based
on the phase difference (→ front/back) and amplitude ratio (→ left/right)
between L and R. Fast and transparent. Good baseline.

### `coherence`
Extends default steering with a per-bin coherence estimate computed over a
temporal neighbourhood of windows. Coherent content (consistent phase
relationship over time = a real spatial cue) is steered confidently. Incoherent
content (reverb, noise, dense mix) is kept in the front stereo field rather
than being aggressively thrown to the rears.

Sub-options:
- `--wiener` — upgrades the coherence scaling to a Wiener filter optimal gain.
  More mathematically principled, noticeably better on material with prominent
  noise or reverb.
- `--gaussian` — use Gaussian temporal weighting in the neighbourhood instead
  of flat averaging. Closer windows count more. Improves transient response.

### `mvdr`
RTF/MVDR (Relative Transfer Function / Minimum Variance Distortionless
Response) beamforming. For each frequency bin, accumulates the 2×2
cross-spectral matrix over the neighbourhood, analytically inverts it, and
derives an optimal separation of the direct sound from the diffuse field.
This is the approach used in high-end professional decoders.

The direct component is steered to the appropriate speaker based on the RTF's
implied direction. The diffuse component is spread evenly across all channels.

Sub-options:
- `--gaussian` — Gaussian weighting of the cross-spectral matrix accumulation.
  This is how professional MVDR implementations work and is recommended.

Best for: pop, rock, jazz, complex mixes, anything with discrete panned
instruments.

### `ambisonic`
First-order B-format Ambisonic intermediate representation. For each bin,
derives a virtual source azimuth from the stereo panning information, encodes
to B-format (W, X, Y components), then decodes to the 5.1 speaker positions
using standard Ambisonic decoding coefficients.

Speaker positions used:
- FL: +30°, FR: −30°, FC: 0°, BL: +110°, BR: −110°

Sub-options:
- `--gaussian` — smooths the direction estimate over time before encoding.
  Recommended for material with long reverb tails.

Best for: classical, ambient, live recordings, anything with a prominent
natural reverb field.

---

## All options

| Option | Default | Description |
|--------|---------|-------------|
| `--matrix` | `coherence` | Algorithm: `default` \| `coherence` \| `mvdr` \| `ambisonic` |
| `--wiener` | false | [coherence] Wiener filter optimal gain |
| `--gaussian` | false | Gaussian temporal weighting |
| `--gaussian-sigma` | 1.5 | Gaussian width in windows |
| `--coherence-radius` | 4 | Neighbourhood half-width in windows |
| `--widen-factor` | 1.0 | Stereo width multiplier (1.0=normal, 2.0=wide) |
| `--preset` | `default` | `default` \| `qs` \| `dolby` \| `horseshoe` |
| `--min-amplitude` | 0.01 | Minimum amplitude threshold for steering |
| `--quiet` | true | Reduce output amplitude to prevent clipping |
| `--output-format` | (from extension) | `wav` \| `flac` |
| `--flac-compression` | 8 | FLAC compression level 0–8 |

### Coherence radius guide

| Radius | Context window (44.1kHz) | Best for |
|--------|--------------------------|----------|
| 2–3 | ±92–138ms | Rock, electronic, fast transients |
| 4–6 | ±185–276ms | Pop, jazz, most music |
| 6–8 | ±276–370ms | Classical, ambient, long reverb |

---

## Batch processing

A batch script is included that upmixes all FLAC files in a directory:

```bash
./upmix_batch.sh <source_dir> <output_dir>
```

Example:
```bash
./upmix_batch.sh /mnt/music/Album /mnt/upmixed/Album
```

Output files are named `<original_name>_MVDR_gaussian_upmix.wav`.

To customise the settings, edit the options in `upmix_batch.sh`.

---

## Output format

Output is a 6-channel WAV or FLAC file with channels in standard 5.1 order:

| Channel | Position |
|---------|----------|
| 1 | Front Left |
| 2 | Front Right |
| 3 | Front Centre |
| 4 | LFE (subwoofer) |
| 5 | Back Left |
| 6 | Back Right |

This channel order is compatible with most media players and AV receivers.
In VLC, set Audio → Channels → 5.1. In mpv, surround output is automatic
if your audio device is configured for 5.1.

---

## Limitations

- Input must be stereo (2-channel). Mono and multichannel sources are not
  supported.
- Very long files at 192kHz may require multiple processing passes (handled
  automatically via chunked processing — no action needed from the user).
- FLAC output requires the `flac` command-line tool to be installed.
- MP3 and other lossy input formats work but codec artefacts may cause
  occasional spurious steering decisions, particularly at low bitrates.
  Lossless input (FLAC, WAV) always gives the best results.
- The CUDA toolkit version in `Cargo.toml` must match your installed version
  (see [Build](#build)).

---

## Attribution

Steering matrix mathematics based on
[soft_matrix](https://github.com/GWBasic/soft_matrix) by GWBasic (MIT License).

---

## License

MIT License — see [LICENSE](LICENSE) for details.
