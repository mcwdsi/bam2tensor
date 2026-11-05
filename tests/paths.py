"""Absolute paths to the bundled test fixtures.

Tests run in a throwaway working directory (see ``conftest.py``), so the
fixtures must be referenced by absolute path rather than relative to the
repository root.
"""

from pathlib import Path

TESTS_DIR = Path(__file__).resolve().parent

# Real aligner output (Biscuit, hg38 coordinates). Only a handful of its reads
# overlap the 700 kb test FASTA; see ``simulate.py`` for a fixture whose reads
# actually cover it.
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
