#!/usr/bin/env bash
# On NixOS, libcuda.so lives in the Nix store (path changes on driver updates).
# Dynamically locate it and add to LD_LIBRARY_PATH before uv runs PyTorch.
CUDA_LIB=$(find /nix/store -maxdepth 3 -name "libcuda.so.1" \
    2>/dev/null | grep -v "32bit\|lib32" \
    | grep "graphics-drivers/lib" | head -1)
if [ -n "$CUDA_LIB" ]; then
    export LD_LIBRARY_PATH="$(dirname "$CUDA_LIB")${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
fi
exec uv run python train.py "$@"
