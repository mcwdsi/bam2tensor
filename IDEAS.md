# IDEAS.md

Performance and feature ideas for future consideration.

## Performance

### Convert CpG site lists to numpy arrays for faster binary search

`cpg_sites_dict` values are Python lists. `numpy.searchsorted` on a numpy
array is faster than `bisect.bisect_left` on a list for large arrays (chr1
has ~2M CpG sites). Change `embedding.py` to store CpG positions as
`numpy.ndarray` instead of `list[int]`, and replace `bisect` calls in
`functions.py` with `numpy.searchsorted`.

### Use `compressed=False` for `save_npz` when speed matters

`scipy.sparse.save_npz(..., compressed=True)` gzip-compresses the output.
For large BAMs the compression step takes noticeable time. Consider adding
a `--no-compress` CLI flag that passes `compressed=False`, trading disk
space for faster writes. The data is small integers (1, 0, -1) so
compression ratio is good, but on fast NVMe storage the compression CPU
cost may dominate.

### Parallel BAM processing with `concurrent.futures`

When processing a directory of BAMs, each file is independent. A
`--workers N` flag using `concurrent.futures.ProcessPoolExecutor` could
process multiple BAMs concurrently. However, the README already documents
`GNU parallel` as the recommended approach, which gives users more control
over resource allocation (especially on HPC with SLURM). Built-in
parallelism would mainly help casual users who don't have GNU parallel.

## Features

### Bismark support

Bismark is the most widely used bisulfite aligner but uses `XM` tags with
a different encoding (`Z`/`z` for methylated/unmethylated CpG, `H`/`h` for
CHH, etc.) rather than the `YD`/`XB` strand tags. Adding Bismark support
would require parsing the `XM` string to extract CpG methylation states
directly, bypassing the current strand-based C/T logic. This is a larger
change but would significantly expand the user base.
