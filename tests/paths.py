"""Absolute paths to the bundled test fixtures.

Tests run in a throwaway working directory (see ``conftest.py``), so the
fixtures must be referenced by absolute path rather than relative to the
repository root.
"""

from pathlib import Path

TESTS_DIR = Path(__file__).resolve().parent

# Real aligner output: experimental EM-seq aligned to hg38 by bwa-meth, all of
# it on chr1. It was not aligned to the 700 kb test FASTA below, so only a
# handful of its reads fall within that FASTA at all, and those are decoded
# against the wrong sequence: tests that pair the two only exercise the
# plumbing, and must pass ``allow_length_mismatch=True`` /
# ``--allow-length-mismatch`` to get past the reference check. See
# ``simulate.py`` for a fixture whose reads really belong to the test FASTA.
TEST_BAM = str(TESTS_DIR / "test.bam")
TEST_BAM_INDEX = str(TESTS_DIR / "test.bam.bai")

# chr1, chr2, chr3 (and an extra ``chrIGNORED`` record), 700 kb each.
TEST_FASTA = str(TESTS_DIR / "test_fasta.fa")

# Real aligner output for the same 557 read pairs (a subset of the nf-core
# test datasets; see tests/data/sarscov2/README.md): one BAM from Bismark, one
# from bwa-meth, and their SARS-CoV-2 reference.
SARSCOV2_DIR = TESTS_DIR / "data" / "sarscov2"
SARSCOV2_FASTA = str(SARSCOV2_DIR / "genome.fasta")
SARSCOV2_CHROMOSOME = "MT192765.1"
SARSCOV2_BISMARK_BAM = str(SARSCOV2_DIR / "bismark.bam")
SARSCOV2_BWAMETH_BAM = str(SARSCOV2_DIR / "bwameth.bam")

# 634 further read pairs that only bwa-meth aligned: complementary-strand pairs
# of the (non-directional) library, most of which bwa-meth gives a strand tag
# that contradicts their bases.
SARSCOV2_BWAMETH_NONDIRECTIONAL_BAM = str(SARSCOV2_DIR / "bwameth_nondirectional.bam")
