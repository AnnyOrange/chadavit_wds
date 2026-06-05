# WDS By-Channel Dataset Spec

This spec defines the WebDataset-style tar shard format used for ChAdaViT pretraining when each image channel is stored as a separate TIFF file.

## Goals

- Support images with variable channel counts in the same dataset.
- Store an N-channel image as `ch1`, `ch2`, ..., `chN` files under one sample key.
- Keep per-sample metadata in a paired JSON file.
- Shuffle samples before packing so every shard contains a mixed distribution of datasets, channel counts, and image sources.
- Be directly readable by ChAdaViT's `wds_packed_shards` dataset loader.

## Directory Layout

The dataset root contains tar shards only, plus optional generated cache files.

```text
webds_micro_100k_by_channel_patched_shuffle/
  filtered_mixed_train_w00-000000.tar
  filtered_mixed_train_w00-000001.tar
  filtered_mixed_train_w01-000000.tar
  ...
```

Recommended shard name:

```text
{split_prefix}_w{worker_id:02d}-{shard_id:06d}.tar
```

Example:

```text
filtered_mixed_train_w03-000017.tar
```

Fields:

- `split_prefix`: logical split and filtering stage, for example `filtered_mixed_train`.
- `worker_id`: writer worker id. This is informational and useful for parallel packing.
- `shard_id`: monotonically increasing shard index within the worker or global packing job.

ChAdaViT should usually match all workers with:

```yaml
wds_train_pattern: "filtered_mixed_train_w*.tar"
```

Use a narrower pattern such as `filtered_mixed_train_w00-*.tar` only for debugging or smoke tests.

## Sample Layout Inside A Tar

Each logical image sample is represented by one metadata JSON file and one or more channel TIFF files.

Required naming:

```text
{sample_key}.ch1.tif
{sample_key}.ch2.tif
...
{sample_key}.chN.tif
{sample_key}.meta.json
```

Example:

```text
id000058757_oid20748173_ch3_p7680_11776_crop_0_0.ch1.tif
id000058757_oid20748173_ch3_p7680_11776_crop_0_0.ch2.tif
id000058757_oid20748173_ch3_p7680_11776_crop_0_0.ch3.tif
id000058757_oid20748173_ch3_p7680_11776_crop_0_0.meta.json
```

The `sample_key` is the filename prefix before `.chN.tif` or `.meta.json`.

Rules:

- All files belonging to one sample must use the exact same `sample_key`.
- Channel ids are 1-indexed: `ch1`, `ch2`, ..., `chN`.
- Channel ids must be contiguous for normal samples. If the source image has 3 channels, write `ch1`, `ch2`, and `ch3`.
- Do not write zero-indexed names such as `ch0`.
- Do not store a multi-channel TIFF as a single file for this loader. Split it into one 2D TIFF per channel.
- Write `.meta.json` after the channel files inside the tar. The current loader materializes a sample when it sees the metadata file.
- If a tar has no metadata files, the loader has a fallback path, but metadata should be considered required for this spec.

## Channel TIFF Requirements

Each `{sample_key}.chN.tif` stores one 2D channel image.

Requirements:

- Format: TIFF.
- Shape: `H x W`, single channel.
- Dtype: preferably `uint16`; `uint8` is also supported.
- All channels of the same sample must have identical `H` and `W`.
- Pixel values should use the full meaningful dynamic range of the dtype when possible.

Loader behavior:

- `uint16` is converted to `float32` in `[0, 1]` by dividing by `65535`.
- `uint8` is converted to `float32` in `[0, 1]` by dividing by `255`.
- Floating point or other numeric dtypes are clipped or scaled into `[0, 1]`.

## Metadata JSON Schema

Each `{sample_key}.meta.json` describes the logical sample. The loader currently ignores most metadata fields, but the metadata is required for provenance, validation, and future filtering.

Required fields:

```json
{
  "id": "id000058757_oid20748173_ch3_p7680_11776",
  "dataset_name": "",
  "available_channels": [1, 2, 3],
  "crop_coordinates": [0, 0, 512, 512],
  "patch_shape": [512, 512],
  "original_path": "/data/webds_micro_100k_by_channel_patched/ch3/id000058757_oid20748173_ch3_p7680_11776.tif",
  "source_sample_id": "id000058757_oid20748173_ch3_p7680_11776",
  "source_channel_count": 3,
  "source_image_shape": [3, 512, 512]
}
```

Recommended optional fields:

```json
{
  "original_shape": null,
  "kept_as_full_image": true,
  "variance_value": 199861203.8961,
  "original_image_id": null,
  "source_crop_coordinates": null
}
```

Field definitions:

- `id`: stable sample id before adding `.chN.tif` or `.meta.json`.
- `dataset_name`: source dataset or collection name. Empty string is allowed when unavailable.
- `available_channels`: sorted channel ids actually written in the tar. Must match the `.chN.tif` files.
- `crop_coordinates`: `[x0, y0, width, height]` or the convention used by the patching job. Keep this convention consistent.
- `patch_shape`: `[height, width]` of the stored channel TIFFs.
- `original_path`: source image path before packing.
- `source_sample_id`: stable source id used to trace related crops.
- `source_channel_count`: number of channels in the source image before or at packing time.
- `source_image_shape`: `[channels, height, width]` for the source image represented by this sample.
- `kept_as_full_image`: `true` if the stored patch is the full image rather than a crop.
- `variance_value`: variance or filtering score used by the patch selection step, if applicable.

Consistency checks:

- `available_channels == sorted(actual_channel_ids_from_filenames)`.
- `len(available_channels) == source_channel_count` for normal uncensored samples.
- `patch_shape` must match every channel TIFF shape.
- `source_image_shape[0]` should match `source_channel_count`.

## Packing Order Within Tar

The preferred order for each sample is:

```text
{sample_key}.ch1.tif
{sample_key}.ch2.tif
...
{sample_key}.chN.tif
{sample_key}.meta.json
```

This order is important because the current streaming loader buffers channel files and yields the sample when it reaches `.meta.json`.

Samples may be interleaved only if the packer guarantees all channel files appear before the corresponding metadata file. The simpler and recommended approach is to write all files of a sample contiguously.

## Shuffle Policy

Shuffle at the sample level before tar packing.

Recommended process:

1. Build a manifest where each row is one logical sample, not one channel file.
2. Include at least `sample_key`, source path, channel count, dataset name, and crop information.
3. Apply deterministic random shuffle with a recorded seed.
4. Partition the shuffled sample list into shards.
5. Write each sample's channel files contiguously into the assigned shard.

Do not shuffle channel files independently. Channels from one sample must remain grouped under the same `sample_key` and in the same tar shard.

Recommended deterministic shuffle:

```text
seed = 42
samples = sorted(samples, key=sample_key)
rng = Random(seed)
rng.shuffle(samples)
```

For distributed or multi-worker packing:

- Start from one global shuffled manifest when possible.
- Split the shuffled manifest across writer workers by contiguous ranges or round-robin sample assignment.
- Ensure each output shard receives complete samples only.
- Record the seed and packer version in a run-level manifest if available.

## Shard Size Policy

Use a size target or sample-count target per shard.

Recommended defaults:

- Target shard size: 2 GB to 4 GB uncompressed tar.
- Maximum shard size: keep below filesystem and training infrastructure limits.
- Avoid very small shards except final tail shards.

The current example dataset uses shards around 3.1 GB, which is acceptable for local filesystem streaming.

## Split Policy

For pretraining, the current setup can use only train shards:

```yaml
ssl_val_loss: false
```

If validation loss is required, produce separate validation shards:

```text
filtered_mixed_val_w00-000000.tar
filtered_mixed_val_w00-000001.tar
...
```

Then configure:

```yaml
wds_val_pattern: "filtered_mixed_val_w*.tar"
```

If no validation shards exist, ChAdaViT's current dataset code can fall back to train shards for validation dataset construction, but this should be considered a convenience fallback rather than a true validation split.

## ChAdaViT Config

Recommended training config for this format:

```yaml
data:
  dataset: "wds_packed_shards"
  train_path: "/mnt/huawei_deepcad/webds_micro_100k_by_channel_patched_shuffle"
  val_path: "/mnt/huawei_deepcad/webds_micro_100k_by_channel_patched_shuffle"
  format: "image_folder"
  img_channels: 8
  max_img_channels: 8
  sample_ratio: 1.0

  wds_train_pattern: "filtered_mixed_train_w*.tar"
  wds_val_pattern: "filtered_mixed_val_w*.tar"
  wds_channels: 8
  wds_min_channels: 1
  wds_require_all_channels: false
  wds_count_mode: "estimate"
  wds_estimate_n_shards: 2
  wds_estimated_samples: null

channels_strategy: "multi_channels"
mixed_channels: false
ssl_val_loss: false
```

Config notes:

- `wds_channels` is the maximum channel id accepted by the loader.
- `wds_min_channels: 1` allows 1-channel, 2-channel, 3-channel, ..., N-channel samples in the same dataset.
- `wds_require_all_channels: false` is required for mixed channel-count data.
- `img_channels` and `max_img_channels` should match the model's expected maximum channel count.
- Use `filtered_mixed_train_w*.tar` to include all workers' shards.

## Validation Checklist

Run these checks after packing:

- Every `.meta.json` has at least one matching `.chN.tif`.
- Every sample has `.ch1.tif`.
- No sample has duplicate channel ids.
- Channel ids are positive integers and do not exceed configured `wds_channels`.
- `available_channels` matches actual channel files.
- All channels for one sample have the same shape.
- TIFF dtype is supported, preferably `uint16`.
- Tar member order is channel files first, metadata last for each sample.
- Shard names match the configured glob pattern.
- Shard count and sample count are plausible.

Example shard inspection:

```bash
tar -tf filtered_mixed_train_w00-000000.tar | head -40
```

Expected output shape:

```text
sample_a.ch1.tif
sample_a.meta.json
sample_b.ch1.tif
sample_b.ch2.tif
sample_b.ch3.tif
sample_b.meta.json
```

## Compatibility Contract

This spec is compatible with `src.data.custom_datasets.WDSPackedShards` when:

- The dataset is configured as `dataset: "wds_packed_shards"`.
- The tar glob pattern matches the produced shard names.
- Channel files use `.chN.tif` suffixes.
- Metadata files use `.meta.json` suffixes.
- `wds_require_all_channels` is `false` for mixed channel-count datasets.

