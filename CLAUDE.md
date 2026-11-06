# CLAUDE.md

This file provides guidance for Claude Code when working with the bam2tensor codebase.

## Project Overview

bam2tensor is a Python package that converts BAM files to sparse tensor representations of DNA methylation data for deep learning pipelines. It extracts methylation states from CpG sites and outputs sparse COO matrices as .npz files.

## Key Commands

```bash
# Install dependencies
uv sync

# Run all checks (pre-commit, mypy, tests, typeguard, xdoctest, docs-build)
nox

# Run specific session
nox --session=tests
nox --session=mypy
nox --session=pre-commit

# Run tests directly
uv run pytest

# Run tests with coverage (via nox)
nox --session=tests

# Format code
uv run black src tests

# Lint code
uv run ruff check src tests

# Type check
uv run mypy src
```

## Project Structure

```
src/bam2tensor/
  __init__.py      # Package version (3.0)
  __main__.py      # Click CLI entry point (bam2tensor command)
  inspect.py       # Inspect CLI entry point (bam2tensor-inspect command)
  embedding.py     # GenomeMethylationEmbedding class (FASTA parsing, CpG indexing)
  functions.py     # Core extraction: extract_methylation_data_from_bam()
  metadata.py      # .npz metadata read/write (provenance info in output files)
  reference.py     # Reference genome download and caching utilities

tests/
  conftest.py         # Runs every test in its own temporary working directory
  paths.py            # Absolute paths to the bundled fixtures (TEST_BAM, TEST_FASTA, ...)
  simulate.py         # Bisulfite read simulator with known per-CpG truth
  data/sarscov2/      # Real Bismark and bwa-meth output for the same reads (+ reference, licence)
  test_main.py        # CLI tests
  test_functions.py   # Core function tests
  test_embedding.py   # Embedding class tests
  test_filters.py     # Non-converted / EM over-conversion filter tests
  test_parallel.py    # --threads parallel extraction tests
  test_simulated.py   # End-to-end tests against simulated reads with known truth
  test_fragments.py   # Hand-built read pairs: one test per fragment-assembly rule
  test_real_aligners.py  # Real Bismark / bwa-meth output: tag conventions, cross-aligner identity
  test_duplication.py  # Read duplication bug tests
  test_inspect.py     # Inspect CLI tests
  test_metadata.py    # Metadata read/write/round-trip tests
  test_reference.py   # Reference download/caching tests
  test.bam, test.bam.bai, test_fasta.fa  # Test fixtures
```

## Code Style

- **Formatter**: Black (88 char line length)
- **Linter**: Ruff (E, F rules)
- **Type checker**: mypy
- **Docstrings**: Google-style, validated by darglint
- **Indentation**: 4 spaces for Python, 2 spaces for YAML/JSON
- **Line endings**: LF

Pre-commit hooks enforce these automatically. Run `nox --session=pre-commit -- install` to set up.

### Docstring Requirements (darglint)

Darglint validates docstrings against actual code. Key rules:

- **Only document explicitly raised exceptions**: In the `Raises:` section, only list exceptions that are raised with an explicit `raise` statement in that function. Do NOT document:
  - Exceptions raised by called functions (even if they propagate up)
  - Implicit exceptions (e.g., `KeyError` from dict access, `IndexError` from list access)
- **Match Args to parameters**: Every parameter must be documented, and documented args must exist
- **Match Returns to return type**: Return documentation must match actual return statements

Run `nox --session=pre-commit` to validate docstrings before committing.

### Doctest Requirements (xdoctest)

xdoctest validates code examples in docstrings. Important rules:

- **Run xdoctest before committing**: `nox --session=xdoctest` or `nox` (runs all sessions)
- **Skip non-executable examples**: If an example references files that don't exist or can't be executed in isolation, add `# xdoctest: +SKIP` as the first line of the example block:

  ```python
  Example:
      >>> # xdoctest: +SKIP
      >>> embedding = GenomeMethylationEmbedding(
      ...     genome_name="hg38",
      ...     fasta_source="reference.fa",  # File doesn't exist in tests
      ... )
  ```

- **Executable examples**: If an example can actually be executed (no external files needed), don't skip it - xdoctest will validate the output matches

## Testing Guidelines

- Test framework: pytest
- Tests are in `tests/` directory
- Test fixtures: `test.bam`, `test.bam.bai`, `test_fasta.fa`
- **Tests run in a temporary working directory** (autouse fixture in `tests/conftest.py`), so cache files and default outputs never land in the checkout. Refer to bundled fixtures through the absolute-path constants in `tests/paths.py` (`TEST_BAM`, `TEST_BAM_INDEX`, `TEST_FASTA`), never by a path relative to the repo root.
- **`test.bam` was not aligned to `test_fasta.fa`**: it is real bwa-meth EM-seq on hg38 chr1, and `test_fasta.fa` is an unrelated 700 kb stand-in. The extractor rightly refuses the combination (chromosome lengths differ), so tests that pair them are plumbing smoke tests only and must pass `allow_length_mismatch=True` (API) or `--allow-length-mismatch` (CLI); one read survives extraction. For tests that need many reads with known answers, use `tests/simulate.py`: `simulate_bam()` writes an indexed Biscuit- or Bismark-style BAM from `test_fasta.fa` and returns the per-read ground truth, including base qualities, improper-pair flags and fragment ends. `tests/test_simulated.py` states the extractor's contract in `_expected()` / `_read_calls()`; when a rule changes, change it there rather than loosening assertions.
- **Hand-built BAMs must be self-consistent**: the `@SQ` length must equal the FASTA sequence length, and a read only yields a call where it shows an intact CpG (the helpers in `tests/test_fragments.py` fill in the CpG's other base for you).
- **Real aligner output** lives in `tests/data/sarscov2/` (paths in `tests/paths.py`): the same 557 read pairs aligned by Bismark and by bwa-meth. Simulated and hand-built reads only encode what we *believe* aligners write; when changing strand or tag handling, `tests/test_real_aligners.py` is the check against what they actually write. The two files must always extract to identical matrices.
- Run tests across Python 3.11, 3.12, 3.13 via nox
- Coverage reporting enabled (minimum threshold: 10%)
- **Typeguard**: `nox --session=typeguard` runs runtime type checking. All function arguments must match their type annotations exactly — pass `str`, not `Path`, when the signature says `str`. This catches real bugs that mypy misses (e.g., passing `pathlib.PosixPath` to a `str` parameter).

## Key Technical Details

### Data Structure
- Output: scipy sparse COO matrix saved as .npz
- Rows = sequenced DNA fragments: the two mates of a read pair merged into one row; a single-end read (or a read whose mate is filtered, missing, on another chromosome, or more than 100 kb away) is its own row. Ordered by chromosome (in `expected_chromosomes` order, the same as the columns: `cpg_sites_dict` is kept in that order whatever the FASTA's), then by where each fragment's first CpG-covering read appears in the BAM.
- Mates are matched by query name **and** position: a waiting fragment is keyed by (name, its first mate's start, the start it expects of the second), so unrelated pairs sharing a name are not merged.
- Columns = CpG sites (ordered by genomic position, determined by reference genome)
- Values: 1 (methylated), 0 (unmethylated), -1 (covered but no usable call: the read does not show an intact CpG there, the base carrying the call is below the minimum base quality, the CpG is in the masked 3' end of the fragment, or the two mates disagree)
- A CpG covered by both mates is stored once: agreement keeps the call, a lone call beats -1, a methylated/unmethylated disagreement becomes -1 (`_merge_mate_calls`).
- `0` (unmethylated) is stored explicitly; uncovered CpGs are absent. Never densify to test for unmethylated calls, and never call `eliminate_zeros()`.
- **Output format version**: `metadata.json` carries `format_version` (currently 2) and `row_unit` (`fragment`), defined in `metadata.py` (`OUTPUT_FORMAT_VERSION`). Files without it are format 1 (one row per read). Bump `OUTPUT_FORMAT_VERSION`, and describe the new version in the `metadata.py` docstring and README, whenever the meaning of rows, columns, values or `tlen` changes.
- Each .npz file contains a `metadata.json` entry with provenance info (genome name, version, CpG index CRC32, expected chromosomes, input BAM name, detected aligner, every filter setting under `filters`, and the counts in `extraction_stats`). Read via `bam2tensor.metadata.read_npz_metadata()`. A new extraction option must be recorded there.
- Each .npz file contains a `tlen.npy` entry with the length of each fragment (absolute BAM TLEN, 0 if unknown) as int32, one per row. Read via `bam2tensor.metadata.read_npz_tlen()`. Returns `None` for files from older versions.
- `extract_methylation_data_from_bam()` returns an `ExtractionResult` NamedTuple with `.matrix` (sparse COO), `.tlen` (numpy int32 array) and `.stats` (dict of counts: reads dropped as improper pairs, CpG observations, and how many were discarded for each reason).
- Matrix values are int64 and row/col indices int32 (int64 only past 2**31 rows or columns). These dtypes are part of the on-disk format; keep them stable.
- Column of a CpG = `embedding.chromosome_offsets[chrom]` + its index in the sorted `embedding.cpg_sites_dict[chrom]`. There is deliberately no per-position lookup dict (it cost several GB for a human genome); use `genomic_position_to_embedding()` (a binary search) or the index arithmetic directly.
- The `cpg_index_crc32` stored in every output is defined over the text `chrom\tpos1,pos2,...` per chromosome joined by newlines. Its value is pinned by a test; never change the definition.

### Methylation Strand Detection
A fragment derives from the top or the bottom strand of the original DNA, and **both mates of a proper pair report on that same strand** (they carry the same strand tag; in real bwa-meth output the only pairs whose mates differ are improper pairs, which are dropped by default). BAM stores reads in top-strand orientation, so for a CpG whose C is at reference position p:
- top-strand fragment: the call is on the C at p (C = methylated, T = unmethylated)
- bottom-strand fragment: the call is on the G at p + 1 (G = methylated, A = unmethylated)

Every read is decoded at its fragment's informative base. The read's own alignment orientation (flag 0x10) is irrelevant and must never be used to filter reads: doing so discards one mate of every pair.
- Bismark aligner: `XM` call string (Z/z = methylated/unmethylated CpG) read at the informative base; strand from `XG` (CT = top, GA = bottom). Bismark writes the call on the C for top-strand alignments and on the G for bottom-strand ones. Without `XG`, the strand is inferred from the bases the calls sit on.
- Biscuit aligner: `YD` tag (f = top, r = bottom); call read from the base
- bwameth aligner: `YD` tag (same format as Biscuit)
- gem3/blueprint aligner: `XB` tag (C = top, G = bottom); not verified against real gem3 output
- The strand tag is trusted. It is correct for directional libraries. bwa-meth mis-tags most complementary-strand (CTOT/CTOB) pairs of a non-directional library, so such data decodes wrongly: a mis-tagged read is read on the base its strand never converts, which always looks methylated; its unmethylated CpGs fail the intact-CpG rule (below) and become -1, so only methylated calls survive and methylation is overstated.
- **Strand-tag check**: for reads decoded from their bases, the extractor also looks at each CpG's *other* base, which the read's strand never converts (G for top-strand reads, C for bottom-strand), wherever the read covers it with sufficient base quality. A conversion product there (A at the G, T at the C) contradicts the tag. Counts are summed per BAM; `--verbose` prints them, and a `RuntimeWarning` fires when more than 2% of at least 1,000 checked CpGs contradict (`_STRAND_CONTRADICTION_WARNING_RATE`, `_STRAND_CHECK_MIN_CPGS` in `functions.py`). It warns only; it cannot repair the calls. `tests/data/sarscov2/bwameth_nondirectional.bam` holds real mis-tagged pairs (15.9% contradictions); `bwameth.bam` holds only directional pairs (0%); real human EM-seq (`tests/test.bam` against hg38 chr1) sits at 0.70%.

### Which Calls Are Kept
These rules are applied per read from its own bases and qualities, identically for the Bismark (`XM`) and base-decoding paths. That identity is what keeps the Bismark and bwa-meth fixtures extracting to the same matrix, so never implement one of them for a single path.
- **Proper pairs only**: a paired read without the proper-pair flag (0x2) is skipped (`keep_improper_pairs` / `--keep-improper-pairs` keeps it). Single-end reads are kept.
- **Intact CpG**: a call is kept only if the CpG's other base, where the read covers it, is present, adjacent in the read, and the reference base (G after the C for top-strand reads, C before the G for bottom-strand reads). Otherwise -1. This removes the false "unmethylated" call a CpA/TpG variant molecule produces (its lone C is converted as non-CpG). A variant on the informative base itself (top-strand read of a TpG molecule) is indistinguishable from conversion and is *not* caught.
- **Base quality**: the informative base needs Phred >= `min_base_quality` (default 20; 0 = off; reads without stored qualities are not filtered). Measured on real EM-seq: bases below Q20 are wrong 14.7% of the time, against under 1% above.
- **3' end mask** (`mask_3prime`, default 0 = off): calls within N bases of the 3' end of the fragment's original strand become -1, to remove end-repair fill-in (unmethylated dCTP). The end is the right-hand end for top-strand fragments (the reverse-aligned mate's reference end) and the left-hand end for bottom-strand ones (the forward-aligned mate's start). It is applied per fragment when the row is emitted, from the mates' own coordinates (`_three_prime_end`, `emit()` in `_extract_chromosome`), because TLEN differs between aligners when mates extend past each other (Bismark: outer span; bwa: distance between 5' ends). TLEN is only the fallback for a top-strand forward read whose mate is unavailable; single-end reads use their own end. Off by default because pipelines often trim for this before alignment.
- **EM over-conversion filter** ignores -1 entries: they neither count towards `min_cpgs` nor shield a fragment.
- **Majority warnings**: the proper-pair and base-quality filters rely on the BAM's own flag and qualities. A `RuntimeWarning` fires when either discards more than half of at least 1,000 otherwise usable reads / calls (`_FILTER_MAJORITY_WARNING_RATE`, `_FILTER_MAJORITY_MIN_COUNT`), so an aligner that never sets 0x2, or placeholder qualities, cannot silently empty the output.

### Reference Check
- `extract_methylation_data_from_bam()` raises `ValueError` when a chromosome's length in the BAM header differs from `embedding.chromosome_lengths` (`check_chromosome_lengths`); `allow_length_mismatch` / `--allow-length-mismatch` turns it off.
- A `RuntimeWarning` fires when more than 10% of at least 1,000 decoded CpGs mismatch the reference (`_REFERENCE_MISMATCH_WARNING_RATE`, `_REFERENCE_MISMATCH_MIN_CPGS`). The count (`cpg_reference_mismatch`) leaves out CpGs whose only fault is the other strand's conversion product at the other base, which belong to the strand-tag check. Real data: 2.1%; a wrong reference of the same length: 78%.

### Dependencies
- pysam: BAM file handling
- biopython: FASTA parsing
- scipy: Sparse matrices (>= 1.15, for `scipy.sparse.get_index_dtype`)
- click: CLI framework
- numpy, tqdm

## CI/CD

- GitHub Actions runs tests on Ubuntu and macOS with Python 3.12
- Automated docs deployment to GitHub Pages

## Common Tasks

**Adding a new CLI option**: Edit `src/bam2tensor/__main__.py`, use Click decorators

**Modifying extraction logic**: Edit `src/bam2tensor/functions.py`, update `extract_methylation_data_from_bam()`

**Changing CpG indexing**: Edit `src/bam2tensor/embedding.py`, update `GenomeMethylationEmbedding` class

**Running the tool**:
```bash
uv run bam2tensor --input-path input.bam --reference-fasta ref.fa
# Or with auto-download:
uv run bam2tensor --input-path input.bam --download-reference hg38
# Inspect an output file:
uv run bam2tensor-inspect output.methylation.npz
```

### Parallel extraction (`--threads`)

The per-chromosome extraction loop is parallelizable. `--threads N` fans
out one chromosome per worker via `concurrent.futures.ProcessPoolExecutor`
with `multiprocessing.get_context("spawn")` (safe for pysam on macOS).
Output is bitwise-identical to `--threads 1` because per-chromosome
results are concatenated in `cpg_sites_dict` order, whatever order the
workers finish in. Each task carries only that chromosome's CpG positions
and column offset; the embedding object itself is never sent to a worker.
Per-chromosome results are typed arrays (column index, int8 call, CpGs per
fragment, TLEN); row indices are rebuilt once at the end from the
per-fragment counts. Mates are only merged within a chromosome, so the
per-chromosome workers assemble fragments exactly as the sequential path
does. Callers using `threads > 1` from a script need the usual
`if __name__ == "__main__":` guard (spawn re-imports the main module).

When recommending CPU counts:
- Small (region-targeted) BAMs: stay at `--threads 1` — starting worker
  interpreters (and importing numpy/scipy/pysam in each) dominates for
  sub-minute jobs.
- Whole-genome WGBS/EM-seq BAMs (≥ 10 GB): `--threads 8` is a good
  default; diminishing returns past `--threads 24` since chr1/chr2
  dominate wall time and that's the longest pole.
- Memory budget: ~1.5 GB per GB of input BAM, plus ~1 GB per worker, is
  the documented conservative rule. Actual use is now several-fold lower
  (about 25 bytes per CpG observation at peak in synthetic benchmarks;
  workers hold only the chromosome they are processing), but that has not
  been re-measured on real whole-genome BAMs.
- OS-level threads (`OMP_NUM_THREADS`, `threading`) do NOT help — the
  hot loop is Python and GIL-bound. Always use `--threads`.

### Reference Genome Downloads
- Downloaded references are cached in `~/.cache/bam2tensor/`
- Available genomes: hg38, hg19, mm10, T2T-CHM13
- The `reference.py` module manages downloads from UCSC/NCBI

### Chromosome Naming
- BAM chromosome names must match the reference FASTA chromosome names
- UCSC style: `chr1`, `chr2`, ... (used by hg38, hg19 from UCSC)
- Ensembl style: `1`, `2`, ... (used by GRCh38 from Ensembl)
- bam2tensor detects mismatches and provides actionable error messages
- Use `--expected-chromosomes` to match your naming convention

## AI Agent Guidelines

When working on this codebase as an AI coding agent:

- **Always run `nox`** (or at minimum `nox --session=pre-commit` and `nox --session=tests`) before committing changes
- **Darglint long strictness**: Every parameter, return value, and explicitly-raised exception must be documented in Google-style docstrings
- **Do not add exceptions to `Raises:`** that are not explicitly raised with a `raise` statement in that function
- **Run `nox --session=xdoctest`** to validate docstring examples; use `# xdoctest: +SKIP` for examples requiring external files
- **Test BAM**: `tests/test.bam` has `chr1`, `chr2`, `chr3` in its header, like `tests/test_fasta.fa`, but with hg38 lengths: it was not aligned to that FASTA (see Testing Guidelines)
- **Creating test BAMs**: Follow the pattern in `tests/test_duplication.py` (pysam to create BAMs, then `pysam.index()`) for hand-built edge cases; use `tests/simulate.py` when you need realistic read sets with known truth
- **CLI exit status**: `bam2tensor` exits 1 if any BAM in the run failed (it still processes the others), and refuses up front a run in which two BAMs would share one output file (same file name under `--output-dir`). `bam2tensor-inspect` exits 1 if any file could not be read. Check the exit status, not just the output, when scripting them
- **Dependency management**: Uses `uv` (not pip/poetry)
- **Type checking**: mypy with `strict = false` but `check_untyped_defs = true`
- **Do not modify** `noxfile.py` (excluded from ruff checks)
- **New CLI options**: Add to both Click decorators and the `main()` function signature in `__main__.py`
- **Output files** are written by `write_output()` in `__main__.py`: assembled under a temporary name and renamed into place, so a partial `.methylation.npz` never exists. Keep any new entry added to the archive inside that function.
- **Cache files**: `.cache.json.gz` files are written to the current working directory. A cache is validated against the FASTA by path + size + modification time when those are trustworthy (same resolved path as the cache was built from), and by SHA-256 otherwise; it is rejected (and regenerated) on a version or content mismatch or if it lacks chromosome lengths, and raises `ValueError` if it was built for a different chromosome list. `expected_chromosomes` must not name a chromosome twice (`ValueError`)

### Using bam2tensor efficiently when invoked from an agent

If a user asks you to run `bam2tensor` on a sample, default to:

- **Whole-genome BAM** (≥ 10 GB): `--threads 8` (or `os.cpu_count() // 2`).
  Single-threaded extraction wastes most of a modern node's cores; the
  parallel path is bitwise-equivalent.
- **Small/region BAM** (< 1 GB): leave `--threads 1`. Worker spawn cost
  exceeds the savings.
- **Cohort run on SLURM**: prefer one array task per BAM with
  `--cpus-per-task=8` and `--threads 8`. See the SLURM template in
  README.md. Don't `srun` a single huge job — array tasks parallelize
  across BAMs and let the scheduler recover from individual failures.
- **`--filter-non-converted` is not replayable from the `.npz`**: if a
  downstream consumer might need conversion-corrected output, enable
  the flag at extraction time (don't skip it expecting to filter later).
- **Re-extracting the same BAM**: check for an existing `.methylation.npz`
  first — `--overwrite` is off by default for a reason.
