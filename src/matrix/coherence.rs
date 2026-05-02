/// Coherence-weighted steering matrix.
///
/// Extends default steering with per-bin coherence estimation over a
/// temporal neighbourhood.  Optional Gaussian weighting and Wiener
/// filter gain are available as sub-options.
///
/// Extra kernel params (in order after fixed args):
///   int   coherence_radius
///   int   use_gaussian   (0 = flat weights, 1 = Gaussian)
///   int   use_wiener     (0 = raw coherence scale, 1 = Wiener optimal gain)
///   float gaussian_sigma
use super::{BaseParams, StereoMatrix};

const KERNEL: &str = r#"
#define PI        3.14159265358979323846f
#define TWO_PI    6.28318530717958647692f
#define HALF_PI   1.57079632679489661923f
#define CENTER_ADJ  0.707106781186548f
#define LFE_FULL   20.0f
#define LFE_START  40.0f

__device__ void to_polar(float re, float im, float* amp, float* phase) {
    *amp   = sqrtf(re * re + im * im);
    *phase = atan2f(im, re);
}
__device__ void from_polar(float amp, float phase, float* re, float* im) {
    *re = amp * cosf(phase);
    *im = amp * sinf(phase);
}
__device__ float wrap(float p) {
    if (p >  PI) p -= TWO_PI;
    if (p < -PI) p += TWO_PI;
    return p;
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
    int use_wiener,
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

    // ---- Coherence estimation over neighbourhood ----
    float sum_re = 0.0f, sum_im = 0.0f, weight_sum = 0.0f;
    int w_lo = max(0, win_idx - coherence_radius);
    int w_hi = min(num_windows - 1, win_idx + coherence_radius);

    for (int w = w_lo; w <= w_hi; w++) {
        float dw = (float)(w - win_idx);
        float weight = use_gaussian
            ? expf(-0.5f * dw * dw / (gaussian_sigma * gaussian_sigma))
            : 1.0f;

        float2 Lw = left_freq [w * window_size + bin];
        float2 Rw = right_freq[w * window_size + bin];

        // L * conj(R) unit vector
        float cr = Lw.x * Rw.x + Lw.y * Rw.y;
        float ci = Lw.y * Rw.x - Lw.x * Rw.y;
        float cm = sqrtf(cr * cr + ci * ci);

        if (cm > 1e-9f) {
            sum_re     += weight * cr / cm;
            sum_im     += weight * ci / cm;
            weight_sum += weight;
        }
    }

    float coherence = 0.0f;
    if (weight_sum > 1e-9f) {
        float avg_re = sum_re / weight_sum;
        float avg_im = sum_im / weight_sum;
        coherence = sqrtf(avg_re * avg_re + avg_im * avg_im);
    }

    // ---- Steering scale: raw coherence or Wiener optimal gain ----
    float steer_scale;
    if (use_wiener) {
        float snr    = coherence / (1.0f - coherence + 1e-6f);
        steer_scale  = snr / (1.0f + snr);
    } else {
        steer_scale = coherence;
    }

    // ---- Load this window's L and R ----
    float2 L = left_freq [flat];
    float2 R = right_freq[flat];

    float la, lp, ra, rp;
    to_polar(L.x, L.y, &la, &lp);
    to_polar(R.x, R.y, &ra, &rp);

    if (la < minimum_amplitude && ra >= minimum_amplitude) lp = rp;
    else if (ra < minimum_amplitude && la >= minimum_amplitude) rp = lp;

    // ---- Default-matrix steering scaled by steer_scale ----
    float phase_diff = fabsf(lp - rp);
    if (phase_diff > PI) phase_diff = TWO_PI - phase_diff;
    float back_from_phase = phase_diff / PI;

    float amp_sum = la + ra;
    float ltr = 0.0f, btf = 0.0f;
    if (amp_sum > 0.0f) {
        ltr = (la / amp_sum) * -2.0f + 1.0f;
        ltr *= widen_factor;
        float back_from_pan = (fabsf(ltr) > 1.0f) ? (fabsf(ltr) - 1.0f) : 0.0f;
        btf = fminf(1.0f, back_from_phase + back_from_pan);
        ltr = fminf(1.0f, fmaxf(-1.0f, ltr));
    }

    btf *= steer_scale;
    ltr *= steer_scale;

    float ftb    = 1.0f - btf;
    float la_adj = la / amplitude_adjustment;
    float ra_adj = ra / amplitude_adjustment;
    float lf_a   = la_adj * ftb;
    float rf_a   = ra_adj * ftb;
    float bl_a   = la_adj * btf * rear_adjustment;
    float br_a   = ra_adj * btf * rear_adjustment;

    float ltr_abs      = fabsf(ltr);
    float center_scale = (1.0f - ltr_abs) * CENTER_ADJ * 0.5f / amplitude_adjustment;
    float fcre = (L.x + R.x) * center_scale;
    float fcim = (L.y + R.y) * center_scale;
    float fc_a = sqrtf(fcre*fcre + fcim*fcim);
    lf_a = fmaxf(0.0f, lf_a - fc_a);
    rf_a = fmaxf(0.0f, rf_a - fc_a);

    float freq_hz  = sample_rate * (float)bin / (float)window_size;
    float lfe_lvl  = 0.0f;
    if      (freq_hz < LFE_FULL)  lfe_lvl = 1.0f;
    else if (freq_hz < LFE_START) lfe_lvl = cosf((freq_hz - LFE_FULL) / LFE_FULL * HALF_PI);
    float lfe_scale = 0.5f * lfe_lvl / amplitude_adjustment;
    float lfere = (L.x + R.x) * lfe_scale;
    float lfeim = (L.y + R.y) * lfe_scale;

    float bl_ph = wrap(lp - HALF_PI);
    float br_ph = wrap(rp + HALF_PI);

    float flre, flim, frre, frim, blre, blim, brre, brim;
    from_polar(lf_a, lp,    &flre, &flim);
    from_polar(rf_a, rp,    &frre, &frim);
    from_polar(bl_a, bl_ph, &blre, &blim);
    from_polar(br_a, br_ph, &brre, &brim);

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

pub struct CoherenceMatrix {
    pub min_amplitude: f32,
    pub widen_factor: f32,
    pub rear_adjustment: f32,
    pub amplitude_adjustment: f32,
    pub coherence_radius: i32,
    pub use_gaussian: bool,
    pub gaussian_sigma: f32,
    pub use_wiener: bool,
}

impl CoherenceMatrix {
    pub fn new(
        min_amplitude: f32,
        widen_factor: f32,
        rear_adjustment: f32,
        amplitude_adjustment: f32,
        coherence_radius: i32,
        use_gaussian: bool,
        gaussian_sigma: f32,
        use_wiener: bool,
    ) -> Self {
        Self {
            min_amplitude, widen_factor, rear_adjustment, amplitude_adjustment,
            coherence_radius, use_gaussian, gaussian_sigma, use_wiener,
        }
    }
}

impl StereoMatrix for CoherenceMatrix {
    fn name(&self) -> &str {
        match (self.use_gaussian, self.use_wiener) {
            (true,  true)  => "coherence+gaussian+wiener",
            (true,  false) => "coherence+gaussian",
            (false, true)  => "coherence+wiener",
            (false, false) => "coherence",
        }
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
        vec![
            self.coherence_radius,
            self.use_gaussian as i32,
            self.use_wiener   as i32,
        ]
    }
    fn extra_float_params(&self) -> Vec<f32> {
        vec![self.gaussian_sigma]
    }
    fn context_windows(&self) -> usize {
        self.coherence_radius as usize
    }
}
