import pysam
from bam2tensor import embedding, functions


def test_duplication_bug(tmp_path):
    """Test that reads overlapping multiple 'potential windows' are only counted once."""
    # 1. Create a dummy reference fasta
    fasta_path = tmp_path / "ref.fa"
    with open(fasta_path, "w") as f:
        f.write(">chr1\n")
        # Sequence: 150bp.
        # CpGs at 10 (CG), 20 (CG), 100 (CG), 110 (CG).
        seq = (
            "N" * 9
            + "CG"
            + "N" * 9
            + "CG"
            + "N" * 79
            + "CG"
            + "N" * 9
            + "CG"
            + "N" * 40
        )
        f.write(seq + "\n")

    # 2. Create the embedding with small window size
    # Even though we removed window generation, we pass the param to ensure no error
    emb = embedding.GenomeMethylationEmbedding(
        "test_genome",
        expected_chromosomes=["chr1"],
        fasta_source=str(fasta_path),
        window_size=50,
        skip_cache=True,
        verbose=True,
    )

    # 3. Create a dummy BAM with one read spanning both CpG clusters
    bam_path = tmp_path / "test.bam"
    header = {"HD": {"VN": "1.0"}, "SQ": [{"LN": len(seq), "SN": "chr1"}]}
    with pysam.AlignmentFile(bam_path, "wb", header=header) as out_bam:
        a = pysam.AlignedSegment()
        a.query_name = "read1"
        a.query_sequence = "N" * len(seq)
        a.flag = 0
        a.reference_id = 0
        a.reference_start = 0
        a.mapping_quality = 60
        a.cigartuples = [(0, len(seq))]  # 150M
        # Add tags to pass filters
        a.set_tag("MD", "150")
        a.set_tag("YD", "f")  # Bisulfite strand
        out_bam.write(a)

    pysam.index(str(bam_path))

    # 4. Run extraction
    matrix = functions.extract_methylation_data_from_bam(
        input_bam=str(bam_path), genome_methylation_embedding=emb, debug=True
    )

    print(f"Matrix shape: {matrix.shape}")
    # We expect exactly 1 row because there is only 1 read
    assert matrix.shape[0] == 1, f"Expected 1 row (read), got {matrix.shape[0]}"
