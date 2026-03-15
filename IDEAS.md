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

## Design Decisions

### Bismark: XM tag takes priority, no strand filtering

Bismark support (added in this codebase) uses the `XM` tag which contains
pre-resolved methylation calls. Key decisions:

- **XM checked first**: If a read has an `XM` tag, the Bismark path is
  used regardless of whether `YD`/`XB` tags are also present. This is
  correct because Bismark's calls are authoritative.
- **No strand filtering**: Unlike Biscuit/bwameth, both reads in a
  Bismark pair carry valid `XM` tags. All reads are processed (no
  parent/daughter strand skip). This means Bismark BAMs produce ~2x
  more rows in the output matrix than Biscuit BAMs for the same data.
- **Non-CpG context**: `XM` characters other than `Z`/`z` at CpG
  reference positions (e.g., `H`, `h`, `X`, `x`, `.`) are recorded as
  -1, consistent with the existing "no data" encoding.
- **Paired-end overlap**: If read1 and read2 in a pair overlap the same
  CpG site, both get separate rows. This is the same behavior as for
  Biscuit. Users should run `deduplicate_bismark` upstream if needed.
