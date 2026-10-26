# Compression

Compression allows us to store and download a small dataset and use a large dataset. Compression is beneficial for text, often compressing shards to a third of the original size, whereas it is marginally helpful for other modalities like images. Compression operates based on shards. We provide several compression algorithms, but in practice, `Zstandard` is a safe bet across the entire time-size Pareto frontier. The higher the quality level, the higher the compression ratio. However, using higher compression levels will impact the compression speed.

Table of supported compression algorithms:

| Name                                          | Code   | Min Level | Default Level | Max Level |
| --------------------------------------------- | ------ | --------- | ------------- | --------- |
| [Brotli](https://github.com/google/brotli)    | br     | 0         | 11            | 11        |
| [Bzip2](https://sourceware.org/bzip2/)        | bz2    | 1         | 9             | 9         |
| [Gzip](https://www.gzip.org/)                 | gz     | 0         | 9             | 9         |
| [Snappy](https://github.com/google/snappy)    | snappy | –         | –             | –         |
| [Zstandard](https://github.com/facebook/zstd) | zstd   | 1         | 3             | 22        |

The compression algorithm to use, if any, is specified by passing `code` or `code:level` as a string to the [Writer](https://docs.mosaicml.com/projects/streaming/en/stable/api_reference/generated/streaming.MDSWriter.html). Decompression happens behind the scenes in the [Stream](https://docs.mosaicml.com/projects/streaming/en/stable/api_reference/generated/streaming.Stream.html) (inside [StreamingDataset](https://docs.mosaicml.com/projects/streaming/en/stable/api_reference/generated/streaming.StreamingDataset.html)) as shards are downloaded. Control whether to keep the compressed version of shards by setting the `keep_zip` flag in the specific Stream’s init or for all streams in StreamingDataset init.

When mixing local sources and remote streams, `cache_limit` applies only to disposable remote caches, including their index files. A stream is a local source when `remote` is unset or its local filesystem path resolves to the same directory as `local`, including symlinks and relative paths. Files in these sources are never evicted, and their size is excluded from cache accounting.

Local sources can contain complete compressed shards, complete uncompressed shards, or both. Reading a compressed local shard writes its uncompressed form into the source directory and retains the compressed files even when `keep_zip=False`. This growth is outside `cache_limit`: provision disk space for both forms of the local dataset. A local source missing both complete forms of a shard is rejected during initialization with a list of missing files.
