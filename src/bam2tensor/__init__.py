"""bam2tensor: Convert BAM files to sparse tensor representations of DNA methylation.

bam2tensor is a Python package for converting bisulfite-sequencing BAM files
to sparse tensor representations of DNA methylation data. It extracts read-level
methylation states from CpG sites and outputs efficient sparse COO matrices
as .npz files, ready for deep learning pipelines.

Main Components:
    GenomeMethylationEmbedding: Class for managing CpG site positions and
        coordinate conversions between genomic positions and matrix indices.
    extract_methylation_data_from_bam: Core function for extracting methylation
        data from a BAM file into a sparse matrix.

Example:
    Command-line usage::

        $ bam2tensor --input-path sample.bam --reference-fasta ref.fa --genome-name hg38

    Python API usage::

        from bam2tensor.embedding import GenomeMethylationEmbedding
        from bam2tensor.functions import extract_methylation_data_from_bam

        # Create genome embedding (parses FASTA for CpG sites)
        embedding = GenomeMethylationEmbedding(
            genome_name="hg38",
            expected_chromosomes=["chr1", "chr2", ...],
            fasta_source="/path/to/reference.fa",
        )

        # Extract methylation data
        sparse_matrix = extract_methylation_data_from_bam(
            input_bam="/path/to/sample.bam",
            genome_methylation_embedding=embedding,
        )

        # Save to file
        import scipy.sparse
        scipy.sparse.save_npz("output.npz", sparse_matrix)

Output Format:
    The output is a SciPy sparse COO matrix where:
    - Rows represent unique reads (primary alignments)
    - Columns represent CpG sites (ordered by genomic position)
    - Values are: 1 (methylated), 0 (unmethylated), -1 (no data/indel/SNV)

See Also:
    - README.md for installation and usage instructions
    - https://mcwdsi.github.io/bam2tensor for full documentation
"""

__version__ = "1.5"
