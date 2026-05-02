#!/usr/bin/env bash
# upmix_batch.sh — batch upmix all FLAC files in a directory
#
# Usage: ./upmix_batch.sh <source_dir> <output_dir>

if [ $# -ne 2 ]; then
    echo "Usage: $0 <source_dir> <output_dir>"
    echo "Example: $0 /mnt/NASmedia/Music/Album /mnt/NASmedia/Upmixed/Album"
    exit 1
fi

SOURCE_DIR="$1"
OUTPUT_DIR="$2"
BINARY="$(dirname "$0")/target/release/soft_matrix_gpu"

if [ ! -f "$BINARY" ]; then
    echo "Error: soft_matrix_gpu binary not found at $BINARY"
    echo "Run 'cargo build --release' first"
    exit 1
fi

if [ ! -d "$SOURCE_DIR" ]; then
    echo "Error: source directory not found: $SOURCE_DIR"
    exit 1
fi

mkdir -p "$OUTPUT_DIR"

shopt -s nullglob
files=("$SOURCE_DIR"/*.flac)

if [ ${#files[@]} -eq 0 ]; then
    echo "No FLAC files found in $SOURCE_DIR"
    exit 0
fi

echo "Found ${#files[@]} FLAC file(s) in $SOURCE_DIR"
echo "Output directory: $OUTPUT_DIR"
echo ""

success=0
failed=0

for f in "${files[@]}"; do
    base=$(basename "$f" .flac)
    out="$OUTPUT_DIR/${base}_MVDR_gaussian_upmix.wav"

    echo "Processing: $base"

    if "$BINARY" "$f" "$out" \
        --matrix mvdr \
        --gaussian \
        --gaussian-sigma 1.5 \
        --coherence-radius 6 \
        --widen-factor 3 \
        --min-amplitude 0.005; then
        echo "  → Done: $(basename "$out")"
        success=$((success + 1))
    else
        echo "  → FAILED: $base"
        failed=$((failed + 1))
    fi

    echo ""
done

echo "Batch complete: $success succeeded, $failed failed"
