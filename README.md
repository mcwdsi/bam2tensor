# Bam2Tensor

[![PyPI](https://img.shields.io/pypi/v/bam2tensor.svg)][pypi status]
[![Status](https://img.shields.io/pypi/status/bam2tensor.svg)][pypi status]
[![Python Version](https://img.shields.io/pypi/pyversions/bam2tensor)][pypi status]
[![License](https://img.shields.io/pypi/l/bam2tensor)][license]

[![Documentation](https://github.com/mcwdsi/bam2tensor/actions/workflows/docs.yml/badge.svg)][documentation]
[![Tests](https://github.com/mcwdsi/bam2tensor/actions/workflows/tests.yml/badge.svg)][tests]
[![Coverage](https://sonarcloud.io/api/project_badges/measure?project=mcwdsi_bam2tensor&metric=coverage)][sonarcov]
[![Quality Gate Status](https://sonarcloud.io/api/project_badges/measure?project=mcwdsi_bam2tensor&metric=alert_status)][sonarquality]

[![pre-commit](https://img.shields.io/badge/pre--commit-enabled-brightgreen?logo=pre-commit&logoColor=white)][pre-commit]
[![Black](https://img.shields.io/badge/code%20style-black-000000.svg)][black]
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
[![Poetry](https://img.shields.io/endpoint?url=https://python-poetry.org/badge/v0.json)][poetry]

[pypi status]: https://pypi.org/project/bam2tensor/
[documentation]: https://mcwdsi.github.io/bam2tensor
[tests]: https://github.com/mcwdsi/bam2tensor/actions?workflow=Tests

[sonarcov]: https://sonarcloud.io/summary/overall?id=mcwdsi_bam2tensor
[sonarquality]: https://sonarcloud.io/summary/overall?id=mcwdsi_bam2tensor
[pre-commit]: https://github.com/pre-commit/pre-commit
[black]: https://github.com/psf/black
[poetry]: https://python-poetry.org/

bam2tensor is a Python package for converting .bam files to dense representations of methylation data (as .npz NumPy arrays). It is designed to evaluate all CpG sites and store methylation states for loading into other deep learning pipelines.

## Features
- Parses .bam files using [pysam](https://github.com/pysam-developers/pysam)
- Extracts methylation data from all CpG sites
- Easily parallelizable
- Supports any genome (Hg38, T2T-CHM13, mm10, etc.)
- Stores methylation data as .npz NumPy arrays
- Stores data in sparse format (COO matrix) for efficient loading

## Requirements

- Python 3.8+
- pysam, numpy, scipy, tqdm

## Installation

You can install _Bam2Tensor_ via [pip] from [PyPI]:

```console
pip install bam2tensor
```

## Usage

Please see the [Reference Guide] for details.

## Contributing

Contributions are welcome! Please see the [Contributor Guide].

## License

Distributed under the terms of the [MIT license][license],
_Bam2Tensor_ is free and open source.

## Issues

If you encounter any problems,
please [file an issue] along with a detailed description.

## Credits

This project is developed and maintained by [Nick Semenkovich (@semenko)], as part of the Medical College of Wisconsin's [Data Science Institute].

This project was generated from [Statistics Norway]'s [SSB PyPI Template].

[Nick Semenkovich (@semenko)]: https://nick.semenkovich.com/
[Data Science Institute]: https://www.mcw.edu/departments/data-science-institute
[statistics norway]: https://www.ssb.no/en
[pypi]: https://pypi.org/
[ssb pypi template]: https://github.com/statisticsnorway/ssb-pypitemplate
[file an issue]: https://github.com/mcwdsi/bam2tensor/issues
[pip]: https://pip.pypa.io/

<!-- github-only -->

[license]: https://github.com/mcwdsi/bam2tensor/blob/main/LICENSE
[contributor guide]: https://github.com/mcwdsi/bam2tensor/blob/main/CONTRIBUTING.md
[reference guide]: https://mcwdsi.github.io/bam2tensor/reference.html
