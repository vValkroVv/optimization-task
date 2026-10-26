# Warmup: DeepGEMM

---

## Warmup (generate a cache directory and upload it to S3)

Use these instructions when the existing DeepGEMM cache has been invalidated, for example after migrating to SM100+ or upgrading DeepGEMM.

### Step-by-step instructions

1. Set `USER_LOGIN` and `WARMUP_CONFIG_PATH` in `gift_scripts/submit_warmup.py`. `WARMUP_CONFIG_PATH` points to the warmup configuration, whose `cache_dir` must be an absolute path.
2. Specify the kernels to warm in the warmup configuration. The standard configuration contains six kernels: one forward and two backward kernels for each of the `w1` and `w2` projections.
Run `gift_scripts/submit_warmup.py` to generate the cache directory locally.
3. Upload the warmed cache to S3 and rebuild the Docker image if necessary. The S3 path is defined in the **CI-CD** repository in `scripts/download_deep_gemm_cache_s3.sh`.
