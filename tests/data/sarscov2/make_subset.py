"""Rebuild the bundled SARS-CoV-2 fixtures from the nf-core test datasets.

The fixtures are a subset of two alignments of the same paired-end FASTQs,
published in nf-core/test-datasets (``modules`` branch, MIT licence) under
``data/genomics/sarscov2/``:

* ``illumina/bam/test.paired_end.methylated.bam`` — aligned with Bismark
  v0.23.0 (unsorted);
* ``illumina/bam/test.paired_end.methylated.sorted.bam`` — aligned with
  bwa-meth 0.2.0;
* ``genome/genome.fasta`` — the reference (MT192765.1), bundled unchanged.

Usage::

    python make_subset.py <bismark.bam> <bwameth.sorted.bam> <output directory>

Every ``STEP``-th fragment (by name) that both aligners placed is kept, with
both of its mates, in both files (``bismark.bam``, ``bwameth.bam``). Bismark
leaves the FASTQ's ``_R1`` suffix on read names and bwa-meth strips it, so
names are compared without it.

The upstream reads are a non-directional library. Bismark, in its default
directional mode, placed only the pairs from the original top and bottom
strands, so the two files above hold directional pairs only. Every
``NONDIRECTIONAL_STEP``-th fragment that *only* bwa-meth placed — the
complementary-strand pairs, most of which bwa-meth gives a strand tag that
contradicts their bases — goes to ``bwameth_nondirectional.bam``.
"""

import os
import sys

import pysam

STEP = 9
NONDIRECTIONAL_STEP = 6


def base_name(name: str | None) -> str:
    """Read name without the mate suffix some aligners keep."""
    assert name is not None
    return name[:-3] if name.endswith(("_R1", "_R2")) else name


def mapped_pairs(path: str) -> set[str]:
    """Names of fragments with both mates mapped as primary alignments."""
    counts: dict[str, int] = {}
    with pysam.AlignmentFile(path, "rb", check_sq=False) as bam:
        for read in bam.fetch(until_eof=True):
            if not read.flag & (0x4 | 0x100 | 0x800):
                name = base_name(read.query_name)
                counts[name] = counts.get(name, 0) + 1
    return {name for name, count in counts.items() if count == 2}


def write_subset(source: str, keep: set[str], final_path: str) -> None:
    """Write the primary alignments of the named fragments, sorted and indexed."""
    unsorted_path = final_path + ".unsorted"
    with pysam.AlignmentFile(source, "rb", check_sq=False) as bam:
        with pysam.AlignmentFile(unsorted_path, "wb", header=bam.header) as out:
            for read in bam.fetch(until_eof=True):
                if base_name(read.query_name) in keep and not read.flag & (
                    0x4 | 0x100 | 0x800
                ):
                    out.write(read)
    pysam.sort("-o", final_path, unsorted_path)
    os.remove(unsorted_path)
    pysam.index(final_path)
    print(f"wrote {final_path}")


def main() -> None:
    bismark_path, bwameth_path, out_dir = sys.argv[1:4]
    in_bismark = mapped_pairs(bismark_path)
    in_bwameth = mapped_pairs(bwameth_path)

    shared = sorted(in_bismark & in_bwameth)
    keep = set(shared[::STEP])
    print(f"{len(shared)} fragments placed by both aligners; keeping {len(keep)}")
    write_subset(bismark_path, keep, os.path.join(out_dir, "bismark.bam"))
    write_subset(bwameth_path, keep, os.path.join(out_dir, "bwameth.bam"))

    bwameth_only = sorted(in_bwameth - in_bismark)
    keep = set(bwameth_only[::NONDIRECTIONAL_STEP])
    print(f"{len(bwameth_only)} fragments placed by bwa-meth only; keeping {len(keep)}")
    write_subset(
        bwameth_path, keep, os.path.join(out_dir, "bwameth_nondirectional.bam")
    )


if __name__ == "__main__":
    main()
