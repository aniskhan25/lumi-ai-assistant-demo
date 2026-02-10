# GPU Tips (LUMI Demo)

This is a short demo doc for GPU usage tips.

## Performance hints
- Ensure your batch size fits GPU memory.
- Avoid oversubscribing CPU cores relative to GPU count.
- Prefer contiguous reads for large datasets when possible.

## GPU visibility
- When Slurm assigns GPUs, `ROCR_VISIBLE_DEVICES` may be set automatically.
- If you set it manually, keep the indices consistent with Slurm allocation.

## Debugging
- Start with a small batch size to validate correctness.
- If you see OOM, reduce batch size or model sequence length.
