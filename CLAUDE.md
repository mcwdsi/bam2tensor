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
  __init__.py      # Package version (2.1)
  __main__.py      # Click CLI entry point
  embedding.py     # GenomeMethylationEmbedding class (FASTA parsing, CpG indexing)
  functions.py     # Core extraction: extract_methylation_data_from_bam()
  reference.py     # Reference genome download and caching utilities

tests/
  test_main.py        # CLI tests
  test_functions.py   # Core function tests
  test_embedding.py   # Embedding class tests
  test_duplication.py  # Read duplication bug tests
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
- Run tests across Python 3.11, 3.12, 3.13 via nox
- Coverage reporting enabled (minimum threshold: 10%)
- **Typeguard**: `nox --session=typeguard` runs runtime type checking. All function arguments must match their type annotations exactly — pass `str`, not `Path`, when the signature says `str`. This catches real bugs that mypy misses (e.g., passing `pathlib.PosixPath` to a `str` parameter).

## Key Technical Details

### Data Structure
- Output: scipy sparse COO matrix saved as .npz
- Rows = unique reads (primary alignments)
- Columns = CpG sites
- Values: 1 (methylated), 0 (unmethylated), -1 (no data/indels/SNVs)

### Methylation Strand Detection
- Bismark aligner: XM tag (Z/z for methylated/unmethylated CpG; no strand filtering needed)
- Biscuit aligner: YD tag
- bwameth aligner: YD tag (same format as Biscuit)
- gem3/blueprint aligner: XB tag

### Dependencies
- pysam: BAM file handling
- biopython: FASTA parsing
- scipy: Sparse matrices
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
```

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
- **Test BAM**: `tests/test.bam` uses chromosomes `chr1`, `chr2`, `chr3` matching `tests/test_fasta.fa`
- **Creating test BAMs**: Follow the pattern in `tests/test_duplication.py` (pysam to create BAMs, then `pysam.index()`)
- **Dependency management**: Uses `uv` (not pip/poetry)
- **Type checking**: mypy with `strict = false` but `check_untyped_defs = true`
- **Do not modify** `noxfile.py` (excluded from ruff checks)
- **New CLI options**: Add to both Click decorators and the `main()` function signature in `__main__.py`
- **Cache files**: `.cache.json.gz` files are written to the current working directory
