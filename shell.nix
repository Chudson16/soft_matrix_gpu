{ pkgs ? import <nixpkgs> {
    config.allowUnfree = true;
    config.cudaSupport = true;
  }
}:

let
  cuda = pkgs.cudaPackages;
in
pkgs.mkShell {
  name = "soft-matrix-gpu";

  buildInputs = [
    cuda.cudatoolkit
    cuda.cuda_nvcc
    cuda.libcufft
    pkgs.rustup
    pkgs.pkg-config
    pkgs.gcc
    # FLAC CLI encoder/decoder (libFLAC) — used for FLAC output
    pkgs.flac
  ];

  CUDA_ROOT  = "${cuda.cudatoolkit}";
  CUDA_PATH  = "${cuda.cudatoolkit}";

  LD_LIBRARY_PATH = pkgs.lib.makeLibraryPath [
    cuda.cudatoolkit
    cuda.libcufft
    "/run/opengl-driver"
  ];

  shellHook = ''
    echo "CUDA  : $CUDA_ROOT"
    echo "flac  : $(flac --version 2>/dev/null || echo 'not found')"
    echo "rustc : $(rustc --version 2>/dev/null || echo 'run: rustup install stable')"
  '';
}
