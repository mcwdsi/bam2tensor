"""Extraction from real Bismark and bwa-meth output.

``tests/data/sarscov2`` holds the same 557 read pairs aligned by Bismark and
by bwa-meth. Hand-built and simulated reads can only encode what we believe
an aligner writes; these files are what two aligners actually wrote, so they
are the check on those beliefs — in particular on where each aligner puts a
bottom-strand call and on which strand tag each mate carries.
"""

import collections

import numpy as np
import pysam
import pytest
from click.testing import CliRunner

from bam2tensor import __main__, embedding, functions
from bam2tensor.functions import detect_aligner
from bam2tensor.metadata import read_npz_metadata
from tests.paths import (
    SARSCOV2_BISMARK_BAM,
    SARSCOV2_BWAMETH_BAM,
    SARSCOV2_CHROMOSOME,
    SARSCOV2_FASTA,
)

ALIGNMENTS = [
    pytest.param(SARSCOV2_BISMARK_BAM, id="bismark"),
    pytest.param(SARSCOV2_BWAMETH_BAM, id="bwameth"),
]


@pytest.fixture(scope="module")
def sarscov2_embedding():
    """Embedding of the SARS-CoV-2 reference (one chromosome, 438 CpGs)."""
    return embedding.GenomeMethylationEmbedding(
        "sarscov2",
        expected_chromosomes=[SARSCOV2_CHROMOSOME],
        fasta_source=SARSCOV2_FASTA,
        skip_cache=True,
    )


def _is_bottom_strand(read) -> bool:
    """The fragment's strand, from the aligner's own tag."""
    if read.has_tag("XG"):  # Bismark
        return read.get_tag("XG") == "GA"
    return read.get_tag("YD") == "r"  # bwa-meth


def _expected_rows(bam_path, emb):
    """Fragment rows worked out directly from the BAM and the reference.

    A deliberately plain re-statement of the extractor's contract: decode
    every read at its informative base, merge mates by name, keep fragments
    in the order they first appear.
    """
    cpg_columns = {
        position - 1: column
        for column, position in enumerate(emb.cpg_sites_dict[SARSCOV2_CHROMOSOME])
    }
    fragments: dict[str, dict[int, int]] = {}
    lengths: dict[str, int] = {}
    with pysam.AlignmentFile(bam_path) as bam:
        for read in bam.fetch(SARSCOV2_CHROMOSOME):
            bottom = _is_bottom_strand(read)
            read_position = {
                ref: query for query, ref in read.get_aligned_pairs(matches_only=True)
            }
            calls = {}
            for cpg, column in cpg_columns.items():
                informative = cpg + 1 if bottom else cpg
                if not read.reference_start <= informative < read.reference_end:
                    continue
                query = read_position.get(informative)
                if query is None:
                    state = -1
                elif read.has_tag("XM"):
                    state = {"Z": 1, "z": 0}.get(str(read.get_tag("XM"))[query], -1)
                else:
                    methylated, unmethylated = ("G", "A") if bottom else ("C", "T")
                    base = str(read.query_sequence)[query]
                    state = (
                        1 if base == methylated else 0 if base == unmethylated else -1
                    )
                calls[column] = state
            if not calls:
                continue
            name = str(read.query_name)
            fragment = fragments.setdefault(name, {})
            lengths.setdefault(name, abs(read.template_length))
            for column, state in calls.items():
                seen = fragment.get(column)
                if seen is None or seen == -1:
                    fragment[column] = state
                elif state != -1 and state != seen:
                    fragment[column] = -1
    return [dict(sorted(calls.items())) for calls in fragments.values()], list(
        lengths.values()
    )


def _rows(result):
    """The matrix as a list of ``{column: state}`` dicts, one per row."""
    rows: list[dict[int, int]] = [{} for _ in range(result.matrix.shape[0])]
    for row, col, state in zip(
        result.matrix.row.tolist(),
        result.matrix.col.tolist(),
        result.matrix.data.tolist(),
    ):
        rows[row][col] = state
    return rows


# ---------------------------------------------------------------------------
# What the aligners write (guards the assumptions the extractor relies on)
# ---------------------------------------------------------------------------


def test_bismark_writes_each_call_on_the_base_that_carries_it() -> None:
    """Top-strand calls sit on the C of a CpG, bottom-strand calls on the G.

    This is the convention the extractor depends on. Reading every call at
    the C — as releases up to 2.8 did — finds '.' for every bottom-strand
    alignment.
    """
    reference = "".join(
        line.strip() for line in open(SARSCOV2_FASTA) if not line.startswith(">")
    ).upper()
    where: collections.Counter = collections.Counter()
    with pysam.AlignmentFile(SARSCOV2_BISMARK_BAM) as bam:
        for read in bam.fetch():
            calls = str(read.get_tag("XM"))
            for query, ref in read.get_aligned_pairs(matches_only=True):
                if calls[query] in "zZ":
                    on_c = reference[ref : ref + 2] == "CG"
                    on_g = reference[ref - 1 : ref + 1] == "CG"
                    where[
                        (read.get_tag("XG"), "C" if on_c else "G" if on_g else "?")
                    ] += 1

    assert set(where) == {("CT", "C"), ("GA", "G")}
    assert where[("CT", "C")] > 500
    assert where[("GA", "G")] > 500


@pytest.mark.parametrize(
    "bam_path, strand_tag, values",
    [
        pytest.param(SARSCOV2_BISMARK_BAM, "XG", {"CT", "GA"}, id="bismark"),
        pytest.param(SARSCOV2_BWAMETH_BAM, "YD", {"f", "r"}, id="bwameth"),
    ],
)
def test_both_mates_carry_the_same_strand_tag(bam_path, strand_tag, values) -> None:
    """The strand tag describes the fragment: read 1 and read 2 always agree.

    So a read's own alignment orientation says nothing about which strand it
    reports on — the forward and the reverse mate of a pair report on the
    same one.
    """
    tags = collections.defaultdict(set)
    orientations = collections.defaultdict(set)
    with pysam.AlignmentFile(bam_path) as bam:
        for read in bam.fetch():
            tags[read.query_name].add(read.get_tag(strand_tag))
            orientations[read.query_name].add(read.is_reverse)

    assert len(tags) == 557
    assert all(len(strands) == 1 for strands in tags.values())
    assert {strand for strands in tags.values() for strand in strands} == values
    # ...while the two mates of every pair align in opposite orientations.
    assert all(orientation == {True, False} for orientation in orientations.values())


# ---------------------------------------------------------------------------
# What the extractor makes of them
# ---------------------------------------------------------------------------


def test_detect_aligner_on_real_output() -> None:
    """Each file is recognised by its tags."""
    assert detect_aligner(SARSCOV2_BISMARK_BAM) == "Bismark (XM tag)"
    assert detect_aligner(SARSCOV2_BWAMETH_BAM) == "Biscuit/bwameth (YD tag)"


@pytest.mark.parametrize("bam_path", ALIGNMENTS)
def test_extraction_matches_calls_read_straight_from_the_bam(
    bam_path, sarscov2_embedding
) -> None:
    """Rows, columns, calls and fragment lengths match an independent decode."""
    expected_rows, expected_lengths = _expected_rows(bam_path, sarscov2_embedding)
    result = functions.extract_methylation_data_from_bam(
        input_bam=bam_path, genome_methylation_embedding=sarscov2_embedding
    )
    assert _rows(result) == expected_rows
    assert result.tlen.tolist() == expected_lengths


@pytest.mark.parametrize("bam_path", ALIGNMENTS)
def test_real_output_summary(bam_path, sarscov2_embedding) -> None:
    """The headline numbers for this fixture, the same for both aligners."""
    result = functions.extract_methylation_data_from_bam(
        input_bam=bam_path, genome_methylation_embedding=sarscov2_embedding
    )
    # 557 pairs; 481 of them cover at least one of the genome's 438 CpGs.
    assert result.matrix.shape == (481, 438)
    assert collections.Counter(result.matrix.data.tolist()) == {1: 1108, 0: 244, -1: 65}
    assert (result.tlen > 0).all()

    # Both strands contribute calls. Before bottom-strand Bismark calls were
    # decoded, half of this file's calls were -1 and half its rows held
    # nothing else.
    rows = _rows(result)
    assert sum(1 for row in rows if set(row.values()) == {-1}) < 20
    assert (result.matrix.data == -1).mean() < 0.06


@pytest.mark.parametrize("threads", [1, 2])
def test_bismark_and_bwameth_alignments_give_the_same_matrix(
    sarscov2_embedding, threads
) -> None:
    """The same reads, aligned by two aligners, yield one and the same matrix.

    One file is decoded from Bismark's call string and ``XG`` tag, the other
    from the read bases and bwa-meth's ``YD`` tag. Agreement on every one of
    the 1,417 calls shows the two paths treat strand, position and mates
    identically.
    """
    from_bismark = functions.extract_methylation_data_from_bam(
        input_bam=SARSCOV2_BISMARK_BAM,
        genome_methylation_embedding=sarscov2_embedding,
        threads=threads,
    )
    from_bwameth = functions.extract_methylation_data_from_bam(
        input_bam=SARSCOV2_BWAMETH_BAM,
        genome_methylation_embedding=sarscov2_embedding,
        threads=threads,
    )
    assert from_bismark.matrix.nnz == 1417
    for attribute in ("row", "col", "data"):
        assert np.array_equal(
            getattr(from_bismark.matrix, attribute),
            getattr(from_bwameth.matrix, attribute),
        ), attribute


@pytest.mark.parametrize("bam_path", ALIGNMENTS)
def test_filters_run_on_real_output(bam_path, sarscov2_embedding) -> None:
    """Both opt-in filters run on real tags and remove some, not all, fragments."""
    unfiltered = functions.extract_methylation_data_from_bam(
        input_bam=bam_path, genome_methylation_embedding=sarscov2_embedding
    )
    filtered = functions.extract_methylation_data_from_bam(
        input_bam=bam_path,
        genome_methylation_embedding=sarscov2_embedding,
        filter_non_converted=True,
        filter_em_overconversion=True,
    )
    assert 0 < filtered.matrix.shape[0] < unfiltered.matrix.shape[0]
    assert len(filtered.tlen) == filtered.matrix.shape[0]


def test_cli_on_real_bismark_output(tmp_path) -> None:
    """End to end: the CLI writes fragment rows and records the format."""
    result = CliRunner().invoke(
        __main__.main,
        [
            "--input-path",
            SARSCOV2_BISMARK_BAM,
            "--reference-fasta",
            SARSCOV2_FASTA,
            "--genome-name",
            "sarscov2",
            "--expected-chromosomes",
            SARSCOV2_CHROMOSOME,
            "--output-dir",
            str(tmp_path / "out"),
        ],
    )
    assert result.exit_code == 0, result.output
    assert "Aligner:       Bismark (XM tag)" in result.output
    assert "481 fragments x 438 CpG sites (1,417 data points)" in result.output

    metadata = read_npz_metadata(str(tmp_path / "out" / "bismark.methylation.npz"))
    assert metadata is not None
    assert metadata["format_version"] == 2
    assert metadata["row_unit"] == "fragment"


def test_retained_cytosine_count_matches_the_reference_on_real_bwameth_output(
    sarscov2_embedding,
) -> None:
    """On real bwa-meth output the filter's count equals a count from the FASTA.

    bwa-meth writes its ``MD`` tag against a converted reference, so this is
    the check that the tag still yields the right answer for the directional
    read pairs bwa-meth is designed for.
    """
    reference = "".join(
        line.strip() for line in open(SARSCOV2_FASTA) if not line.startswith(">")
    ).upper()
    cpgs = {
        position - 1
        for position in sarscov2_embedding.cpg_sites_dict[SARSCOV2_CHROMOSOME]
    }
    reads = with_retained = 0
    with pysam.AlignmentFile(SARSCOV2_BWAMETH_BAM) as bam:
        for read in bam.fetch():
            bottom = _is_bottom_strand(read)
            sequence = str(read.query_sequence)
            from_fasta = 0
            for query, ref in read.get_aligned_pairs(matches_only=True):
                if bottom:
                    retained = reference[ref] == "G" and sequence[query] == "G"
                    in_cpg = reference[ref - 1] == "C"
                else:
                    retained = reference[ref] == "C" and sequence[query] == "C"
                    in_cpg = reference[ref + 1 : ref + 2] == "G"
                from_fasta += retained and not in_cpg
            assert (
                functions.count_non_cpg_retained_reference(read, bottom, cpgs)
                == from_fasta
            ), read.query_name
            reads += 1
            with_retained += from_fasta > 0

    assert reads == 1114
    assert with_retained > 500  # the comparison is not vacuous
