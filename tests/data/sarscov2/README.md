# SARS-CoV-2 bisulfite alignments (real aligner output)

Small paired-end test data aligned by two different bisulfite aligners. The
reads are simulated, but the BAM files are genuine aligner output, so they pin
down each aligner's tag conventions in a way hand-built reads cannot.

| File | What it is |
|---|---|
| `genome.fasta` | Reference, SARS-CoV-2 isolate MT192765.1 (29,829 bp, 438 CpGs) |
| `bismark.bam` | 557 read pairs aligned with Bismark v0.23.0 (`XM`, `XR`, `XG` tags) |
| `bwameth.bam` | The same 557 read pairs aligned with bwa-meth 0.2.0 (`YD`, `YC` tags) |
| `make_subset.py` | Script that produced the two BAMs from the upstream files |

Both BAMs contain the same fragments, so extracting either must give the same
matrix. See `tests/test_real_aligners.py`.

## Source and licence

Subset of files from the [nf-core/test-datasets] repository (`modules`
branch, `data/genomics/sarscov2/`), redistributed under its MIT licence; see
`LICENSE` in this directory.

- `illumina/bam/test.paired_end.methylated.bam` (Bismark)
- `illumina/bam/test.paired_end.methylated.sorted.bam` (bwa-meth)
- `genome/genome.fasta` (bundled unchanged)

Every ninth fragment that both aligners placed was kept; `make_subset.py`
records exactly how.

## Things worth knowing about this data

- Bismark leaves the FASTQ's `_R1` suffix on the name of both mates; bwa-meth
  strips it. Mates still share a name within each file.
- The reads were simulated with the two mates of a pair converted
  independently, so mates disagree at overlapping CpGs far more often than
  real mates do (which come from one molecule). That is why about 5% of the
  merged calls here are `-1`.
- The two aligners report different `TLEN` for dovetailed pairs (fragment
  shorter than the read length), so fragment lengths are not compared across
  them.

[nf-core/test-datasets]: https://github.com/nf-core/test-datasets
