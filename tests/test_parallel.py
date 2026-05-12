"""Tests for the --threads parallel extraction path."""

import numpy as np
import pysam
import pytest
from bam2tensor import embedding
from bam2tensor import functions

# Shared embedding for tests that hit the bundled multi-chromosome fixture.
TEST_EMBEDDING = embedding.GenomeMethylationEmbedding(
    "test_genome",
    expected_chromosomes=["chr1", "chr2", "chr3"],
    fasta_source="tests/test_fasta.fa",
    window_size=150,
    skip_cache=False,
)


def _assert_results_equal(a, b):
    """Assert two ExtractionResult values are bitwise-identical."""
    assert a.matrix.shape == b.matrix.shape
    # Compare COO triplets directly — both paths build COO from the same
    # iteration order, so triplets should match position-for-position.
    assert a.matrix.nnz == b.matrix.nnz
    assert (a.matrix.row == b.matrix.row).all()
    assert (a.matrix.col == b.matrix.col).all()
    assert (a.matrix.data == b.matrix.data).all()
    assert a.tlen.dtype == b.tlen.dtype
    assert np.array_equal(a.tlen, b.tlen)


def test_threads_2_matches_threads_1_on_test_bam():
    """threads=2 produces bitwise-identical output to threads=1 on test.bam."""
    serial = functions.extract_methylation_data_from_bam(
        input_bam="tests/test.bam",
        genome_methylation_embedding=TEST_EMBEDDING,
        quality_limit=20,
        threads=1,
    )
    parallel = functions.extract_methylation_data_from_bam(
        input_bam="tests/test.bam",
        genome_methylation_embedding=TEST_EMBEDDING,
        quality_limit=20,
        threads=2,
    )
    _assert_results_equal(serial, parallel)


def test_threads_4_matches_threads_1_on_test_bam():
    """threads=4 (more workers than chromosomes is fine) matches threads=1."""
    serial = functions.extract_methylation_data_from_bam(
        input_bam="tests/test.bam",
        genome_methylation_embedding=TEST_EMBEDDING,
        quality_limit=20,
        threads=1,
    )
    parallel = functions.extract_methylation_data_from_bam(
        input_bam="tests/test.bam",
        genome_methylation_embedding=TEST_EMBEDDING,
        quality_limit=20,
        threads=4,
    )
    _assert_results_equal(serial, parallel)


def test_threads_invalid_value():
    """threads=0 raises ValueError before any work happens."""
    with pytest.raises(ValueError, match="threads must be >= 1"):
        functions.extract_methylation_data_from_bam(
            input_bam="tests/test.bam",
            genome_methylation_embedding=TEST_EMBEDDING,
            threads=0,
        )


def _make_multichrom_bam(tmp_path):
    """Build a 3-chromosome FASTA and BAM with reads on each chromosome.

    The reads carry YD=f (Biscuit/bwameth forward strand) so all of them
    pass strand filtering and land in the matrix. We seed enough reads
    per chromosome that the cumulative-offset logic in the parallel
    path is exercised.
    """
    fasta_path = tmp_path / "ref.fa"
    seq_len = 200
    # CpGs at 1-based positions 10, 21 (matches the helper pattern).
    seq = "N" * 9 + "CG" + "N" * 9 + "CG" + "N" * (seq_len - 22)
    with open(fasta_path, "w") as f:
        for chrom in ["chr1", "chr2", "chr3"]:
            f.write(f">{chrom}\n{seq}\n")

    emb = embedding.GenomeMethylationEmbedding(
        "test_multichrom",
        expected_chromosomes=["chr1", "chr2", "chr3"],
        fasta_source=str(fasta_path),
        skip_cache=True,
    )

    bam_path = tmp_path / "multichrom.bam"
    header = {
        "HD": {"VN": "1.0"},
        "SQ": [
            {"LN": seq_len, "SN": "chr1"},
            {"LN": seq_len, "SN": "chr2"},
            {"LN": seq_len, "SN": "chr3"},
        ],
    }
    # 2 reads on chr1, 3 on chr2, 4 on chr3 — enough to expose any
    # cross-chromosome row-numbering bug.
    reads_per_chrom = {"chr1": 2, "chr2": 3, "chr3": 4}
    read_seq = list("N" * seq_len)
    read_seq[9] = "C"  # methylated
    read_seq[20] = "T"  # unmethylated
    read_seq_str = "".join(read_seq)
    with pysam.AlignmentFile(bam_path, "wb", header=header) as out_bam:
        for ref_id, chrom in enumerate(["chr1", "chr2", "chr3"]):
            for read_idx in range(reads_per_chrom[chrom]):
                a = pysam.AlignedSegment()
                a.query_name = f"{chrom}_read{read_idx}"
                a.query_sequence = read_seq_str
                a.flag = 0
                a.reference_id = ref_id
                a.reference_start = 0
                a.mapping_quality = 60
                a.cigartuples = [(0, seq_len)]
                a.template_length = 100 + read_idx  # distinct TLENs per read
                a.set_tag("MD", str(seq_len))
                a.set_tag("YD", "f")
                out_bam.write(a)
    pysam.index(str(bam_path))
    return emb, str(bam_path)


def test_multichrom_threads_2_matches_serial(tmp_path):
    """Multi-chromosome BAM: parallel result is bitwise-identical to serial."""
    emb, bam_path = _make_multichrom_bam(tmp_path)
    serial = functions.extract_methylation_data_from_bam(
        input_bam=bam_path, genome_methylation_embedding=emb, threads=1
    )
    parallel = functions.extract_methylation_data_from_bam(
        input_bam=bam_path, genome_methylation_embedding=emb, threads=2
    )
    _assert_results_equal(serial, parallel)
    # Sanity: we expect 9 reads (2 + 3 + 4) × 2 CpGs each.
    assert serial.matrix.shape[0] == 9
    assert serial.matrix.nnz == 18


def test_multichrom_threads_3_preserves_row_order(tmp_path):
    """With threads=3 (one worker per chromosome), row order still
    follows cpg_sites_dict order (chr1 reads first, then chr2, then chr3)."""
    emb, bam_path = _make_multichrom_bam(tmp_path)
    result = functions.extract_methylation_data_from_bam(
        input_bam=bam_path, genome_methylation_embedding=emb, threads=3
    )
    # Rows 0-1 are chr1 reads, 2-4 are chr2, 5-8 are chr3.
    # All CpGs land in chr1 columns vs chr2 columns vs chr3 columns; we
    # use that to verify the chromosome-order of rows is preserved.
    csr = result.matrix.tocsr()
    chr1_cpgs = len(emb.cpg_sites_dict["chr1"])
    chr2_cpgs = len(emb.cpg_sites_dict["chr2"])

    def row_chrom(row_idx):
        cols = csr.getrow(row_idx).indices
        if (cols < chr1_cpgs).all():
            return "chr1"
        if ((cols >= chr1_cpgs) & (cols < chr1_cpgs + chr2_cpgs)).all():
            return "chr2"
        return "chr3"

    chroms = [row_chrom(i) for i in range(result.matrix.shape[0])]
    assert chroms == ["chr1"] * 2 + ["chr2"] * 3 + ["chr3"] * 4


def test_multichrom_tlen_order_preserved(tmp_path):
    """TLEN values are in chromosome-then-read order (matches serial path)."""
    emb, bam_path = _make_multichrom_bam(tmp_path)
    serial = functions.extract_methylation_data_from_bam(
        input_bam=bam_path, genome_methylation_embedding=emb, threads=1
    )
    parallel = functions.extract_methylation_data_from_bam(
        input_bam=bam_path, genome_methylation_embedding=emb, threads=3
    )
    assert np.array_equal(serial.tlen, parallel.tlen)
    # Expected: chr1 (100, 101), chr2 (100, 101, 102), chr3 (100, 101, 102, 103)
    expected = np.array([100, 101, 100, 101, 102, 100, 101, 102, 103], dtype=np.int32)
    assert np.array_equal(parallel.tlen, expected)


def test_threads_with_missing_chromosome(tmp_path):
    """Parallel path handles chromosomes that are absent from the BAM."""
    fasta_path = tmp_path / "ref.fa"
    seq = "N" * 9 + "CG" + "N" * 139
    with open(fasta_path, "w") as f:
        f.write(">chr1\n" + seq + "\n")
        f.write(">chr2\n" + seq + "\n")

    emb = embedding.GenomeMethylationEmbedding(
        "test_missing_chrom_parallel",
        expected_chromosomes=["chr1", "chr2"],
        fasta_source=str(fasta_path),
        skip_cache=True,
    )

    bam_path = tmp_path / "only_chr1.bam"
    header = {"HD": {"VN": "1.0"}, "SQ": [{"LN": 150, "SN": "chr1"}]}
    with pysam.AlignmentFile(bam_path, "wb", header=header) as out_bam:
        a = pysam.AlignedSegment()
        a.query_name = "read1"
        a.query_sequence = "N" * 150
        a.flag = 0
        a.reference_id = 0
        a.reference_start = 0
        a.mapping_quality = 60
        a.cigartuples = [(0, 150)]
        a.set_tag("MD", "150")
        a.set_tag("YD", "f")
        out_bam.write(a)
    pysam.index(str(bam_path))

    serial = functions.extract_methylation_data_from_bam(
        input_bam=str(bam_path), genome_methylation_embedding=emb, threads=1
    )
    parallel = functions.extract_methylation_data_from_bam(
        input_bam=str(bam_path), genome_methylation_embedding=emb, threads=2
    )
    _assert_results_equal(serial, parallel)
    assert serial.matrix.shape[0] == 1


def test_threads_with_filters_matches_serial(tmp_path):
    """Filters (non-converted + EM over-conversion) work identically in parallel."""
    emb, bam_path = _make_multichrom_bam(tmp_path)
    common = dict(
        input_bam=bam_path,
        genome_methylation_embedding=emb,
        filter_non_converted=True,
        non_converted_threshold=3,
        filter_em_overconversion=True,
        em_overconversion_min_cpgs=2,
    )
    serial = functions.extract_methylation_data_from_bam(**common, threads=1)
    parallel = functions.extract_methylation_data_from_bam(**common, threads=3)
    _assert_results_equal(serial, parallel)


def test_threads_dtype_is_int32(tmp_path):
    """TLEN dtype stays int32 in parallel path."""
    emb, bam_path = _make_multichrom_bam(tmp_path)
    result = functions.extract_methylation_data_from_bam(
        input_bam=bam_path, genome_methylation_embedding=emb, threads=2
    )
    assert result.tlen.dtype == np.int32
