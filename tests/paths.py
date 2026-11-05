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
