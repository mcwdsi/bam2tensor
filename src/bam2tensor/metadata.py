"""Metadata utilities for bam2tensor .npz output files.

This module provides functions to embed and retrieve provenance metadata
inside the ``.methylation.npz`` files produced by bam2tensor.  The metadata
is stored as a ``metadata.json`` entry appended to the ZIP archive that
underlies every ``.npz`` file.  ``scipy.sparse.load_npz`` silently ignores
this extra entry, so existing downstream code is unaffected.

Example:
    Writing metadata (done automatically by the CLI)::

        >>> # xdoctest: +SKIP
        >>> from bam2tensor.metadata import write_npz_metadata, read_npz_metadata
        >>> write_npz_metadata("sample.methylation.npz", {
        ...     "bam2tensor_version": "2.2",
        ...     "genome_name": "hg38",
        ... })
        >>> read_npz_metadata("sample.methylation.npz")
        {'bam2tensor_version': '2.2', 'genome_name': 'hg38'}

    Reading metadata from an existing file::

        >>> # xdoctest: +SKIP
        >>> meta = read_npz_metadata("sample.methylation.npz")
        >>> if meta is not None:
        ...     print(meta["genome_name"])
        hg38
"""

import io
import json
import zipfile
import zlib

import numpy as np

from bam2tensor.embedding import GenomeMethylationEmbedding


def compute_cpg_index_crc32(embedding: GenomeMethylationEmbedding) -> str:
    """Compute a CRC32 checksum over the CpG site positions in an embedding.

    The checksum captures the exact column mapping of the sparse matrix:
    which chromosomes are included, in what order, and which genomic
    positions are CpG sites within each chromosome.  Two embeddings with
    the same checksum will produce identical column semantics.

    Args:
        embedding: A fully initialised GenomeMethylationEmbedding whose
            ``cpg_sites_dict`` is populated.

    Returns:
        The CRC32 checksum as an 8-character lowercase hexadecimal string.

    Example:
        >>> # xdoctest: +SKIP
        >>> from bam2tensor.embedding import GenomeMethylationEmbedding
        >>> emb = GenomeMethylationEmbedding(
        ...     genome_name="hg38",
        ...     expected_chromosomes=["chr1"],
        ...     fasta_source="ref.fa",
        ... )
        >>> compute_cpg_index_crc32(emb)
        'a1b2c3d4'
    """
    # Build a deterministic byte representation:
    #   chrom\tpos1,pos2,...\n  (one line per chromosome, in order)
    parts: list[str] = []
    for chrom in embedding.expected_chromosomes:
        positions = embedding.cpg_sites_dict.get(chrom, [])
        parts.append(chrom + "\t" + ",".join(str(p) for p in positions))
    payload = "\n".join(parts).encode("utf-8")
    return format(zlib.crc32(payload) & 0xFFFFFFFF, "08x")


def write_npz_metadata(
    npz_path: str,
    metadata: dict,
) -> None:
    """Append a ``metadata.json`` entry to an existing ``.npz`` file.

    The metadata is serialised as compact JSON and appended to the ZIP
    archive.  ``scipy.sparse.load_npz`` ignores unrecognised entries, so
    the file remains fully compatible with existing code.

    Args:
        npz_path: Path to the ``.npz`` file (must already exist).
        metadata: A JSON-serialisable dictionary of metadata to embed.

    Example:
        >>> # xdoctest: +SKIP
        >>> write_npz_metadata("out.npz", {"genome_name": "hg38"})
    """
    with zipfile.ZipFile(npz_path, "a") as zf:
        zf.writestr("metadata.json", json.dumps(metadata, indent=2))


def read_npz_metadata(npz_path: str) -> dict | None:
    """Read the ``metadata.json`` entry from a ``.npz`` file.

    Args:
        npz_path: Path to the ``.npz`` file.

    Returns:
        The metadata dictionary, or ``None`` if the file does not contain
        a ``metadata.json`` entry (e.g. files produced by older versions).

    Example:
        >>> # xdoctest: +SKIP
        >>> meta = read_npz_metadata("sample.methylation.npz")
        >>> meta["genome_name"]
        'hg38'
    """
    with zipfile.ZipFile(npz_path, "r") as zf:
        if "metadata.json" in zf.namelist():
            return json.loads(zf.read("metadata.json"))
    return None


def write_npz_tlen(npz_path: str, tlen: np.ndarray) -> None:
    """Append a ``tlen.npy`` entry to an existing ``.npz`` file.

    The array is serialised with :func:`numpy.save` and appended to the
    ZIP archive.  ``scipy.sparse.load_npz`` ignores this extra entry, so
    the file remains compatible with existing sparse-matrix code.

    Args:
        npz_path: Path to the ``.npz`` file (must already exist).
        tlen: A 1-D numpy array of per-read template lengths.

    Example:
        >>> # xdoctest: +SKIP
        >>> write_npz_tlen("out.npz", np.array([300, -300, 0], dtype=np.int32))
    """
    buf = io.BytesIO()
    np.save(buf, tlen)
    with zipfile.ZipFile(npz_path, "a") as zf:
        zf.writestr("tlen.npy", buf.getvalue())


def read_npz_tlen(npz_path: str) -> np.ndarray | None:
    """Read the ``tlen.npy`` entry from a ``.npz`` file.

    Args:
        npz_path: Path to the ``.npz`` file.

    Returns:
        The per-read template-length array, or ``None`` if the file does
        not contain a ``tlen.npy`` entry (e.g. files produced by older
        versions).

    Example:
        >>> # xdoctest: +SKIP
        >>> tlen = read_npz_tlen("sample.methylation.npz")
        >>> if tlen is not None:
        ...     print(f"Median fragment length: {np.median(np.abs(tlen)):.0f}")
    """
    with zipfile.ZipFile(npz_path, "r") as zf:
        if "tlen.npy" in zf.namelist():
            buf = io.BytesIO(zf.read("tlen.npy"))
            return np.load(buf, allow_pickle=False)
    return None
