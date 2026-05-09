# patchdataset_worker_cache

Purpose: enable worker-local image/mask caching in PatchDataset for Stage1 patch
training and replay scans.

This is a code/runtime optimization. It does not change Stage2 predictions by itself.
Use configs/stage1_p0_a40_cache.yaml for a Stage1 rerun that enables:

- patch_worker_cache: true
- patch_worker_cache_max_items: 0

Each DataLoader worker keeps its own in-process cache, so repeated patches from the
same source image avoid repeated PIL image/mask reads while preserving original files.
