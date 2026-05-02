/// RTF/MVDR beamforming matrix.
///
/// For each frequency bin, accumulates the 2×2 cross-spectral matrix Φ
/// over a temporal neighbourhood (with optional Gaussian weighting), then:
///
///   1. Estimates the Relative Transfer Function (RTF) from L to R:
///        h = S_LR / S_LL   (complex scalar)
///
///   2. Analytically inverts the 2×2 Hermitian Φ:
///        det = S_LL·S_RR − |S_LR|²  + ε   (regularised)
///
///   3. Computes MVDR weights and the direct-to-diffuse ratio:
///        SDR = |S_LR|² / (S_LL·S_RR − |S_LR|² + ε)
///
///   4. Routes the direct component to steered positions (direction
///      derived from RTF phase and amplitude ratio).
///      Routes the diffuse component evenly across all channels.
///
/// Extra kernel params (in order after fixed args):
///   int   coherence_radius
///   int   use_gaussian
///   float gaussian_sigma
use super::{BaseParams, StereoMatrix};

const KERNEL: &str = r#"
#define PI        3.14159265358979323846f
#define TWO_PI    6.28318530717958647692f
#define HALF_PI   1.57079632679489661923f
#define CENTER_ADJ  0.707106781186548f
#define LFE_FULL   20.0f
#define LFE_START  40.0f

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

    // ---- Accumulate cross-spectral matrix over neighbourhood ----
    // Φ = [[S_LL, S_LR], [conj(S_LR), S_RR]]
    // S_LL and S_RR are real (power spectra).
    // S_LR is complex (cross-spectrum).
    float S_LL = 0.0f, S_RR = 0.0f;
    float S_LR_re = 0.0f, S_LR_im = 0.0f;
    float weight_sum = 0.0f;

    int w_lo = max(0, win_idx - coherence_radius);
    int w_hi = min(num_windows - 1, win_idx + coherence_radius);

    for (int w = w_lo; w <= w_hi; w++) {
        float dw = (float)(w - win_idx);
        float weight = use_gaussian
            ? expf(-0.5f * dw * dw / (gaussian_sigma * gaussian_sigma))
            : 1.0f;

        float2 Lw = left_freq [w * window_size + bin];
        float2 Rw = right_freq[w * window_size + bin];

        S_LL    += weight * (Lw.x*Lw.x + Lw.y*Lw.y);
        S_RR    += weight * (Rw.x*Rw.x + Rw.y*Rw.y);
        // L * conj(R)
        S_LR_re += weight * (Lw.x*Rw.x + Lw.y*Rw.y);
        S_LR_im += weight * (Lw.y*Rw.x - Lw.x*Rw.y);
        weight_sum += weight;
    }

    if (weight_sum > 1e-9f) {
        S_LL    /= weight_sum;
        S_RR    /= weight_sum;
        S_LR_re /= weight_sum;
        S_LR_im /= weight_sum;
    }

    // ---- Regularised determinant ----
    float S_LR_sq = S_LR_re*S_LR_re + S_LR_im*S_LR_im;
    float eps     = 1e-6f * (S_LL + S_RR + 1e-12f);
    float det     = S_LL * S_RR - S_LR_sq + eps;

    // ---- Direct-to-diffuse ratio from MVDR ----
    // SDR = |S_LR|² / det  (clamped to [0,1])
    float sdr         = fminf(1.0f, S_LR_sq / det);
    float diffuse_mix = 1.0f - sdr;     // fraction that is diffuse

    // ---- RTF-derived direction ----
    // RTF h = S_LR / S_LL gives the complex gain from L to R.
    // Its magnitude encodes amplitude panning, phase encodes time difference.
    float h_re = (S_LL > eps) ? (S_LR_re / S_LL) : 0.0f;
    float h_im = (S_LL > eps) ? (S_LR_im / S_LL) : 0.0f;
    float h_amp = sqrtf(h_re*h_re + h_im*h_im);

    // Left-to-right: derived from RTF amplitude ratio
    // h_amp > 1 → R louder → right-panned; h_amp < 1 → L louder → left-panned
    float ltr = 0.0f;
    if (h_amp + 1.0f > 1e-6f)
        ltr = (h_amp - 1.0f) / (h_amp + 1.0f);   // maps (0,∞) → (-1,1)
    ltr = fminf(1.0f, fmaxf(-1.0f, ltr * widen_factor));

    // Back-to-front: derived from RTF phase (time difference of arrival)
    // Large phase difference → more out-of-phase → more rear
    float rtf_phase   = atan2f(h_im, h_re);
    float phase_abs   = fabsf(rtf_phase);
    float btf_direct  = fminf(1.0f, phase_abs / PI);

    // Blend: direct component steered, diffuse spread evenly
    float btf  = btf_direct * sdr;
    float ftb  = 1.0f - btf;

    // ---- Load current bin's L and R ----
    float2 L = left_freq [flat];
    float2 R = right_freq[flat];

    float la = sqrtf(L.x*L.x + L.y*L.y);
    float ra = sqrtf(R.x*R.x + R.y*R.y);
    float lp = atan2f(L.y, L.x);
    float rp = atan2f(R.y, R.x);

    if (la < minimum_amplitude && ra >= minimum_amplitude) lp = rp;
    else if (ra < minimum_amplitude && la >= minimum_amplitude) rp = lp;

    float la_adj = la / amplitude_adjustment;
    float ra_adj = ra / amplitude_adjustment;

    // ---- Direct component: steered ----
    float lf_direct = la_adj * ftb * sdr;
    float rf_direct = ra_adj * ftb * sdr;
    float bl_direct = la_adj * btf * sdr * rear_adjustment;
    float br_direct = ra_adj * btf * sdr * rear_adjustment;

    // ---- Diffuse component: spread evenly across front and rear ----
    float mono_adj   = (la_adj + ra_adj) * 0.5f;
    float lf_diffuse = mono_adj * diffuse_mix;
    float rf_diffuse = mono_adj * diffuse_mix;
    float bl_diffuse = mono_adj * diffuse_mix * 0.7f;
    float br_diffuse = mono_adj * diffuse_mix * 0.7f;

    float lf_a = lf_direct + lf_diffuse;
    float rf_a = rf_direct + rf_diffuse;
    float bl_a = bl_direct + bl_diffuse;
    float br_a = br_direct + br_diffuse;

    // ---- Center: complex sum, scaled by centerness and SDR ----
    float ltr_abs      = fabsf(ltr);
    float center_scale = (1.0f - ltr_abs) * CENTER_ADJ * 0.5f * sdr / amplitude_adjustment;
    float center_diff  = (1.0f - ltr_abs) * CENTER_ADJ * 0.5f * diffuse_mix / amplitude_adjustment;
    float fcre = (L.x + R.x) * (center_scale + center_diff);
    float fcim = (L.y + R.y) * (center_scale + center_diff);
    float fc_a = sqrtf(fcre*fcre + fcim*fcim);
    lf_a = fmaxf(0.0f, lf_a - fc_a * 0.5f);
    rf_a = fmaxf(0.0f, rf_a - fc_a * 0.5f);

    // ---- LFE ----
    float freq_hz  = sample_rate * (float)bin / (float)window_size;
    float lfe_lvl  = 0.0f;
    if      (freq_hz < LFE_FULL)  lfe_lvl = 1.0f;
    else if (freq_hz < LFE_START) lfe_lvl = cosf((freq_hz - LFE_FULL) / LFE_FULL * HALF_PI);
    float lfe_scale = 0.5f * lfe_lvl / amplitude_adjustment;
    float lfere = (L.x + R.x) * lfe_scale;
    float lfeim = (L.y + R.y) * lfe_scale;

    // ---- Phase shifts ----
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

pub struct MvdrMatrix {
    pub min_amplitude: f32,
    pub widen_factor: f32,
    pub rear_adjustment: f32,
    pub amplitude_adjustment: f32,
    pub coherence_radius: i32,
    pub use_gaussian: bool,
    pub gaussian_sigma: f32,
}

impl MvdrMatrix {
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

impl StereoMatrix for MvdrMatrix {
    fn name(&self) -> &str {
        if self.use_gaussian { "mvdr+gaussian" } else { "mvdr" }
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
