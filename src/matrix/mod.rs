pub mod ambisonic;
pub mod coherence;
pub mod default;
pub mod mvdr;

pub use ambisonic::AmbisonicMatrix;
pub use coherence::CoherenceMatrix;
pub use default::DefaultMatrix;
pub use mvdr::MvdrMatrix;

pub struct BaseParams {
    pub min_amplitude: f32,
    pub widen_factor: f32,
    pub rear_adjustment: f32,
    pub amplitude_adjustment: f32,
}

/// A stereo-to-surround matrix algorithm.
///
/// ## Required kernel signature
///
/// ```c
/// extern "C" __global__ void steer_and_assign(
///     const float2* left_freq,  const float2* right_freq,
///     float2* out_fl, float2* out_fr, float2* out_fc,
///     float2* out_lfe, float2* out_bl, float2* out_br,
///     int window_size, int num_windows,
///     float min_amplitude, float widen_factor,
///     float rear_adjustment, float amplitude_adjustment,
///     float sample_rate
///     /*, extra ints from extra_int_params() */
///     /*, extra floats from extra_float_params() */
/// )
/// ```
pub trait StereoMatrix: Send + Sync {
    fn name(&self) -> &str;
    fn kernel_src(&self) -> &'static str;
    fn base_params(&self) -> BaseParams;
    fn extra_int_params(&self) -> Vec<i32> { vec![] }
    fn extra_float_params(&self) -> Vec<f32> { vec![] }
    fn context_windows(&self) -> usize { 0 }
}
