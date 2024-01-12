"""Stores methylation embedding information for a given genome."""

import os


class GenomeMethylationEmbedding:
    """Stores methylation embedding information for a given genome.

    This class helps identify CpG sites in a given .fasta file and stores
    methylation embedding information for each CpG site for analyzing .bam files.

    Stored embedding information cached and stored in gzipped json files.

    Attributes:
        genome_name: The name of the genome.
        fasta_source: The source of the fasta file.
        expected_chromosomes: The expected chromosomes in the genome.
        methylation_embedding: The methylation embedding for the genome.

    """

    genome_name = None
    fasta_source = None
    expected_chromosomes = None

    def __init__(self, genome_name, fasta_source, expected_chromosomes):
        self.genome_name = genome_name
        self.fasta_source = fasta_source
        self.expected_chromosomes = expected_chromosomes

        self.methylation_embedding = {}

        self.embedding_loaded = False

        # Check that we can read the fasta file
        if not os.access(self.fasta_source, os.R_OK):
            raise Exception("Cannot read fasta file: " + self.fasta_source)

        # Check that the expected chromosomes are not empty
        if len(self.expected_chromosomes) == 0:
            raise Exception("Expected chromosomes cannot be empty")

    def add_methylation_embedding(self, methylation_embedding):
        self.methylation_embedding = methylation_embedding

    def get_methylation_embedding(self):
        return self.methylation_embedding

    def get_genome_name(self):
        return self.genome_name

    def get_methylation_embedding_size(self):
        return len(self.methylation_embedding)

    def get_methylation_embedding_vector(self, position):
        return self.methylation_embedding[position]

    def get_methylation_embedding_vector_size(self, position):
        return len(self.methylation_embedding[position])

    def get_methylation_embedding_vector_at_position(self, position):
        return self.methylation_embedding[position]

    def get_methylation_embedding_vector_at_position_size(self, position):
        return len(self.methylation_embedding[position])
