# CLAUDE.md

This file provides guidance for Claude Code when working with the bam2tensor codebase.

## Project Overview

bam2tensor is a Python package that converts BAM files to sparse tensor representations of DNA methylation data for deep learning pipelines. It extracts methylation states from CpG sites and outputs sparse COO matrices as .npz files.

## Key Commands

```bash
# Install dependencies
poetry install

# Run all checks (pre-commit, mypy, tests, typeguard, xdoctest, docs-build)
nox

# Run specific session
nox --session=tests
nox --session=mypy
nox --session=pre-commit

# Run tests directly
poetry run pytest

# Run tests with coverage
poetry run pytest --cov

# Format code
poetry run black src tests

# Lint code
poetry run ruff check src tests

# Type check
poetry run mypy src
```

## Project Structure

```
src/bam2tensor/
  __init__.py      # Package version (1.4)
  __main__.py      # Click CLI entry point
  embedding.py     # GenomeMethylationEmbedding class (FASTA parsing, CpG indexing)
  functions.py     # Core extraction: extract_methylation_data_from_bam()

tests/
  test_main.py       # CLI tests
  test_functions.py  # Core function tests
  test_embedding.py  # Embedding class tests
  test_duplication.py
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

## Testing Guidelines

- Test framework: pytest
- Tests are in `tests/` directory
- Test fixtures: `test.bam`, `test.bam.bai`, `test_fasta.fa`
- Run tests across Python 3.11, 3.12, 3.13 via nox
- Coverage reporting enabled (minimum threshold: 10%)

## Key Technical Details

### Data Structure
- Output: scipy sparse COO matrix saved as .npz
- Rows = unique reads (primary alignments)
- Columns = CpG sites
- Values: 1 (methylated), 0 (unmethylated), -1 (no data/indels/SNVs)

### Methylation Strand Detection
- Biscuit aligner: YD tag
- gem3/blueprint aligner: XB tag

### Dependencies
- pysam: BAM file handling
- biopython: FASTA parsing
- scipy: Sparse matrices
- click: CLI framework
- numpy, tqdm

## CI/CD

- GitHub Actions runs tests on Ubuntu and macOS with Python 3.12
- SonarCloud integration for quality metrics
- Automated docs deployment to GitHub Pages

## Common Tasks

**Adding a new CLI option**: Edit `src/bam2tensor/__main__.py`, use Click decorators

**Modifying extraction logic**: Edit `src/bam2tensor/functions.py`, update `extract_methylation_data_from_bam()`

**Changing CpG indexing**: Edit `src/bam2tensor/embedding.py`, update `GenomeMethylationEmbedding` class

**Running the tool**:
```bash
poetry run bam2tensor --input-path input.bam --reference-fasta ref.fa
```
