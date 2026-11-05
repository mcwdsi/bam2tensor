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
both of its mates, in both files. Bismark leaves the FASTQ's ``_R1`` suffix on
read names and bwa-meth strips it, so names are compared without it.
"""

import os
import sys

import pysam

STEP = 9


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


def main() -> None:
    bismark_path, bwameth_path, out_dir = sys.argv[1:4]
    shared = sorted(mapped_pairs(bismark_path) & mapped_pairs(bwameth_path))
    keep = set(shared[::STEP])
    print(f"{len(shared)} fragments placed by both aligners; keeping {len(keep)}")

    for label, source in (("bismark", bismark_path), ("bwameth", bwameth_path)):
        unsorted_path = os.path.join(out_dir, f"{label}.unsorted.bam")
        final_path = os.path.join(out_dir, f"{label}.bam")
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


if __name__ == "__main__":
    main()
