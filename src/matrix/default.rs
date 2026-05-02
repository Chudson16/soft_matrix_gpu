/// Default steering matrix — direct port of soft_matrix's DefaultMatrix.
///
/// Steers each frequency bin based on:
///   - Phase difference between L and R → front/back pan
///   - Amplitude ratio between L and R  → left/right pan
///
/// Configurable as: default, horseshoe, dolby, qs/rm.
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
    float sample_rate
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int half_window = window_size / 2;
    if (idx >= num_windows * half_window) return;

    int win_idx  = idx / half_window;
    int freq_off = idx % half_window;
    int bin      = freq_off + 1;
    int flat     = win_idx * window_size + bin;
    int mirror   = win_idx * window_size + (window_size - bin);

    float2 L = left_freq[flat];
    float2 R = right_freq[flat];

    float la, lp, ra, rp;
    to_polar(L.x, L.y, &la, &lp);
    to_polar(R.x, R.y, &ra, &rp);

    if (la < minimum_amplitude && ra >= minimum_amplitude) lp = rp;
    else if (ra < minimum_amplitude && la >= minimum_amplitude) rp = lp;

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

    float ftb     = 1.0f - btf;
    float la_adj  = la / amplitude_adjustment;
    float ra_adj  = ra / amplitude_adjustment;
    float lf_a    = la_adj * ftb;
    float rf_a    = ra_adj * ftb;
    float bl_a    = la_adj * btf * rear_adjustment;
    float br_a    = ra_adj * btf * rear_adjustment;

    float ltr_abs = fabsf(ltr);

    // Center and LFE: sum L and R complex values directly to preserve phase.
    // Averaging phases separately causes comb filtering when L and R are
    // out of phase, which produces crackling on loud center-panned material.
    float center_scale = (1.0f - ltr_abs) * CENTER_ADJ * 0.5f / amplitude_adjustment;
    float fcre  = (L.x + R.x) * center_scale;
    float fcim  = (L.y + R.y) * center_scale;
    
    float freq_hz = sample_rate * (float)bin / (float)window_size;
    float lfe_lvl = 0.0f;
    if      (freq_hz < LFE_FULL)  lfe_lvl = 1.0f;
    else if (freq_hz < LFE_START) lfe_lvl = cosf((freq_hz - LFE_FULL) / LFE_FULL * HALF_PI);
    float lfe_scale = 0.5f * lfe_lvl / amplitude_adjustment;
    float lfere = (L.x + R.x) * lfe_scale;
    float lfeim = (L.y + R.y) * lfe_scale;
    
    // Subtract center energy from front channels using consistent complex math
    float fc_a = sqrtf(fcre*fcre + fcim*fcim);
    lf_a = fmaxf(0.0f, lf_a - fc_a);
    rf_a = fmaxf(0.0f, rf_a - fc_a);
    
    float bl_ph = wrap(lp - HALF_PI);
    float br_ph = wrap(rp + HALF_PI);
    
    float flre, flim, frre, frim, blre, blim, brre, brim;
    from_polar(lf_a, lp,     &flre, &flim);
    from_polar(rf_a, rp,     &frre, &frim);
    from_polar(bl_a, bl_ph,  &blre, &blim);
    from_polar(br_a, br_ph,  &brre, &brim);
    
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

pub struct DefaultMatrix {
    pub min_amplitude: f32,
    pub widen_factor: f32,
    pub rear_adjustment: f32,
    pub amplitude_adjustment: f32,
}

impl DefaultMatrix {
    pub fn new(
        min_amplitude: f32,
        widen_factor: f32,
        rear_adjustment: f32,
        amplitude_adjustment: f32,
    ) -> Self {
        Self { min_amplitude, widen_factor, rear_adjustment, amplitude_adjustment }
    }
}

impl StereoMatrix for DefaultMatrix {
    fn name(&self) -> &str { "default" }
    fn kernel_src(&self) -> &'static str { KERNEL }
    fn base_params(&self) -> BaseParams {
        BaseParams {
            min_amplitude: self.min_amplitude,
            widen_factor: self.widen_factor,
            rear_adjustment: self.rear_adjustment,
            amplitude_adjustment: self.amplitude_adjustment,
        }
    }
    // No extra params — DefaultMatrix uses only the fixed scalar args.
}
