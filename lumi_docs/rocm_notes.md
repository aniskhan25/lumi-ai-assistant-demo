# ROCm Notes (LUMI Demo)

This is a short demo doc for common ROCm pitfalls.

## Environment
- Make sure ROCm modules are loaded before running GPU workloads.
- Some containers require `--rocm` or `--env` flags to expose devices.

## Common issues
- "hipErrorInvalidDevice": GPU index out of range.
- "HSA runtime" errors can indicate missing device visibility.

## Suggestions
- Check that the container sees GPUs with `rocminfo`.
- Validate driver and runtime compatibility for the container image.
