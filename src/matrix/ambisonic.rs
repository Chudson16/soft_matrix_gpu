/// First-order Ambisonic (B-format) intermediate matrix.
///
/// For each frequency bin:
///
///   1. Derives a virtual source direction θ from the stereo panning
///      (amplitude ratio) and a front/back estimate from phase difference.
///
///   2. Encodes to 1st-order B-format:
///        W = (L + R) / sqrt(2)           omnidirectional
///        X = amplitude * cos(θ)          front-back figure-8
///        Y = amplitude * sin(θ)          left-right figure-8
///        Z = 0                           (no height information in stereo)
///
///   3. Decodes B-format to 5.1 speaker positions using standard
///      Ambisonic decoding coefficients:
///        FL  at  +30°
///        FR  at  −30°
///        FC  at    0°
///        BL  at +110°
///        BR  at −110°
///        LFE: low-pass filtered W
///
/// Optional Gaussian temporal smoothing of the direction estimate
/// is available via --gaussian / --gaussian-sigma.
///
/// Extra kernel params (in order after fixed args):
///   int   coherence_radius   (for Gaussian smoothing; 0 = no smoothing)
///   int   use_gaussian
///   float gaussian_sigma
use super::{BaseParams, StereoMatrix};

const KERNEL: &str = r#"
#define PI        3.14159265358979323846f
#define TWO_PI    6.28318530717958647692f
#define HALF_PI   1.57079632679489661923f
#define SQRT2     1.41421356237309504880f
#define LFE_FULL   20.0f
#define LFE_START  40.0f

// Standard 5.1 speaker azimuths in radians
// FL=+30°, FR=−30°, FC=0°, BL=+110°, BR=−110°
#define AZ_FL   0.52359877559f   //  30 deg
#define AZ_FR  -0.52359877559f   // -30 deg
#define AZ_FC   0.0f
#define AZ_BL   1.91986217719f   // 110 deg
#define AZ_BR  -1.91986217719f   // -110 deg

__device__ float wrap(float p) {
    if (p >  PI) p -= TWO_PI;
    if (p < -PI) p += TWO_PI;
    return p;
}

// 1st-order Ambisonic decode gain for a speaker at azimuth spk_az
// given a source at azimuth src_az:
//   g = (1/sqrt(2)) * W_gain + cos(src_az - spk_az) * X_gain
//       + sin(src_az) * cos(spk_az) * Y_gain  (simplified for 2D)
// Using standard max-rE decode: W=1/sqrt(2), XY=sqrt(3)/2 * cos/sin(az)
#define W_GAIN   0.70710678118f  // 1/sqrt(2)
#define XY_GAIN  0.86602540378f  // sqrt(3)/2

__device__ float ambisonic_decode_gain(float src_az, float spk_az) {
    return W_GAIN + XY_GAIN * cosf(src_az - spk_az);
}

extern "C" __global__ void steer_and_assign(
    const float2* __restrict__ left_freq,
    const float2* __restrict__ right_freq,
    float2* __restrict__ out_fl,  float2* __restrict__ out_fr,
    float2* __restrict__ out_fc,  float2* __restrict__ out_lfe,
    float2* __restrict__ out_bl,  float2* __restrict__ out_br,
    int window_size, int num_windows,
    float minimum_amplitude,
    float widen_factor, float rear_adjustment, float amplitude_adjustment,
    float sample_rate,
    int coherence_radius,
    int use_gaussian,
    float gaussian_sigma
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int half_window = window_size / 2;
    if (idx >= num_windows * half_window) return;

    int win_idx  = idx / half_window;
    int freq_off = idx % half_window;
    int bin      = freq_off + 1;
    int flat     = win_idx * window_size + bin;
    int mirror   = win_idx * window_size + (window_size - bin);

    // ---- Optional: smooth the direction estimate over a neighbourhood ----
    // We accumulate a weighted average of the cross-spectrum to get a
    // stable direction, then apply it to the current window's signal.
    float avg_ltr = 0.0f, avg_btf = 0.0f, weight_sum = 0.0f;

    int w_lo = (coherence_radius > 0) ? max(0, win_idx - coherence_radius) : win_idx;
    int w_hi = (coherence_radius > 0) ? min(num_windows - 1, win_idx + coherence_radius) : win_idx;

    for (int w = w_lo; w <= w_hi; w++) {
        float dw = (float)(w - win_idx);
        float weight = (use_gaussian && coherence_radius > 0)
            ? expf(-0.5f * dw * dw / (gaussian_sigma * gaussian_sigma))
            : 1.0f;

        float2 Lw = left_freq [w * window_size + bin];
        float2 Rw = right_freq[w * window_size + bin];

        float la = sqrtf(Lw.x*Lw.x + Lw.y*Lw.y);
        float ra = sqrtf(Rw.x*Rw.x + Rw.y*Rw.y);
        float amp_sum = la + ra;

        if (amp_sum > 1e-9f) {
            // Left-to-right panning from amplitude ratio
            float w_ltr = ((la / amp_sum) * -2.0f + 1.0f) * widen_factor;
            w_ltr = fminf(1.0f, fmaxf(-1.0f, w_ltr));

            // Front-to-back from phase difference
            float lph = atan2f(Lw.y, Lw.x);
            float rph = atan2f(Rw.y, Rw.x);
            float pd  = fabsf(lph - rph);
            if (pd > PI) pd = TWO_PI - pd;
            float w_btf = pd / PI;

            avg_ltr    += weight * w_ltr;
            avg_btf    += weight * w_btf;
            weight_sum += weight;
        }
    }

    float ltr = (weight_sum > 1e-9f) ? (avg_ltr / weight_sum) : 0.0f;
    float btf = (weight_sum > 1e-9f) ? (avg_btf / weight_sum) : 0.0f;

    // ---- Load current window's L and R ----
    float2 L = left_freq [flat];
    float2 R = right_freq[flat];

    float la = sqrtf(L.x*L.x + L.y*L.y);
    float ra = sqrtf(R.x*R.x + R.y*R.y);

    if (la < minimum_amplitude && ra >= minimum_amplitude) {
        // R only: borrow L's role for phase
        L.x = R.x; L.y = R.y;
    } else if (ra < minimum_amplitude && la >= minimum_amplitude) {
        R.x = L.x; R.y = L.y;
    }

    // ---- Encode to B-format W, X, Y ----
    // W = mono sum, normalised
    // X = front-back component (positive = front)
    // Y = left-right component (positive = left per Ambisonic convention)
    //
    // Source azimuth θ: 0=front, +90=left, ±180=rear
    // ltr: -1=left(+90°), 0=center(0°), +1=right(-90°)
    // btf:  0=front,       1=rear(±180°)
    //
    // Map to azimuth: side component from ltr, front/back from btf
    float side_az  = -ltr * HALF_PI;          // ltr=+1→right→−90°
    float front_az = (ltr >= 0.0f ? 1.0f : -1.0f) * btf * PI;
    float src_az   = wrap(side_az + front_az * 0.5f);

    // B-format signals (complex — preserve phase)
    float amp_total = (la + ra) / (amplitude_adjustment * SQRT2);

    // W: omnidirectional — complex sum scaled
    float Wre = (L.x + R.x) / (amplitude_adjustment * SQRT2);
    float Wim = (L.y + R.y) / (amplitude_adjustment * SQRT2);

    // X, Y: directional — scale W by direction cosines
    float cos_az = cosf(src_az);
    float sin_az = sinf(src_az);

    float Xre = Wre * cos_az * XY_GAIN / W_GAIN;
    float Xim = Wim * cos_az * XY_GAIN / W_GAIN;
    float Yre = Wre * sin_az * XY_GAIN / W_GAIN;
    float Yim = Wim * sin_az * XY_GAIN / W_GAIN;

    // ---- Decode B-format to 5.1 speakers ----
    // g_spk = W_GAIN * W + XY_GAIN * (cos(az)*X + sin(az)*Y)  — but X,Y already encoded
    // Simplified: speaker gain = ambisonic_decode_gain(src_az, spk_az)
    // Applied to W (which carries the signal energy)

    float g_fl  = ambisonic_decode_gain(src_az, AZ_FL);
    float g_fr  = ambisonic_decode_gain(src_az, AZ_FR);
    float g_fc  = ambisonic_decode_gain(src_az, AZ_FC);
    float g_bl  = ambisonic_decode_gain(src_az, AZ_BL)  * rear_adjustment;
    float g_br  = ambisonic_decode_gain(src_az, AZ_BR)  * rear_adjustment;

    // Clamp gains to [0, 2] — they can go slightly negative for sources
    // directly behind a front speaker; clamp prevents phase inversion artefacts
    g_fl = fmaxf(0.0f, g_fl);
    g_fr = fmaxf(0.0f, g_fr);
    g_fc = fmaxf(0.0f, g_fc);
    g_bl = fmaxf(0.0f, g_bl);
    g_br = fmaxf(0.0f, g_br);

    // Apply gains to W (carries correct phase)
    float flre = Wre * g_fl,  flim = Wim * g_fl;
    float frre = Wre * g_fr,  frim = Wim * g_fr;
    float fcre = Wre * g_fc,  fcim = Wim * g_fc;
    float blre = Wre * g_bl,  blim = Wim * g_bl;
    float brre = Wre * g_br,  brim = Wim * g_br;

    // Rear phase shift ±90° (matches soft_matrix convention)
    float lp   = atan2f(L.y, L.x);
    float rp   = atan2f(R.y, R.x);
    float bl_ph = wrap(lp - HALF_PI);
    float br_ph = wrap(rp + HALF_PI);
    float bl_amp = sqrtf(blre*blre + blim*blim);
    float br_amp = sqrtf(brre*brre + brim*brim);
    blre = bl_amp * cosf(bl_ph);
    blim = bl_amp * sinf(bl_ph);
    brre = br_amp * cosf(br_ph);
    brim = br_amp * sinf(br_ph);

    // ---- LFE: low-pass filtered W ----
    float freq_hz  = sample_rate * (float)bin / (float)window_size;
    float lfe_lvl  = 0.0f;
    if      (freq_hz < LFE_FULL)  lfe_lvl = 1.0f;
    else if (freq_hz < LFE_START) lfe_lvl = cosf((freq_hz - LFE_FULL) / LFE_FULL * HALF_PI);
    float lfere = Wre * lfe_lvl;
    float lfeim = Wim * lfe_lvl;

    out_fl [flat] = make_float2(flre,  flim);
    out_fr [flat] = make_float2(frre,  frim);
    out_fc [flat] = make_float2(fcre,  fcim);
    out_lfe[flat] = make_float2(lfere, lfeim);
    out_bl [flat] = make_float2(blre,  blim);
    out_br [flat] = make_float2(brre,  brim);

    if (bin < half_window) {
        out_fl [mirror] = make_float2(flre,  -flim);
        out_fr [mirror] = make_float2(frre,  -frim);
        out_fc [mirror] = make_float2(fcre,  -fcim);
        out_lfe[mirror] = make_float2(lfere, -lfeim);
        out_bl [mirror] = make_float2(blre,  -blim);
        out_br [mirror] = make_float2(brre,  -brim);
    }
}
"#;

pub struct AmbisonicMatrix {
    pub min_amplitude: f32,
    pub widen_factor: f32,
    pub rear_adjustment: f32,
    pub amplitude_adjustment: f32,
    pub coherence_radius: i32,
    pub use_gaussian: bool,
    pub gaussian_sigma: f32,
}

impl AmbisonicMatrix {
    pub fn new(
        min_amplitude: f32,
        widen_factor: f32,
        rear_adjustment: f32,
        amplitude_adjustment: f32,
        coherence_radius: i32,
        use_gaussian: bool,
        gaussian_sigma: f32,
    ) -> Self {
        Self {
            min_amplitude, widen_factor, rear_adjustment, amplitude_adjustment,
            coherence_radius, use_gaussian, gaussian_sigma,
        }
    }
}

impl StereoMatrix for AmbisonicMatrix {
    fn name(&self) -> &str {
        if self.use_gaussian { "ambisonic+gaussian" } else { "ambisonic" }
    }
    fn kernel_src(&self) -> &'static str { KERNEL }
    fn base_params(&self) -> BaseParams {
        BaseParams {
            min_amplitude: self.min_amplitude,
            widen_factor: self.widen_factor,
            rear_adjustment: self.rear_adjustment,
            amplitude_adjustment: self.amplitude_adjustment,
        }
    }
    fn extra_int_params(&self) -> Vec<i32> {
        vec![self.coherence_radius, self.use_gaussian as i32]
    }
    fn extra_float_params(&self) -> Vec<f32> {
        vec![self.gaussian_sigma]
    }
    fn context_windows(&self) -> usize {
        self.coherence_radius as usize
    }
}
