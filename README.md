# Repository Coverage

[Full report](https://htmlpreview.github.io/?https://github.com/mcwdsi/bam2tensor/blob/python-coverage-comment-action-data/htmlcov/index.html)

| Name                           |    Stmts |     Miss |   Branch |   BrPart |   Cover |   Missing |
|------------------------------- | -------: | -------: | -------: | -------: | ------: | --------: |
| src/bam2tensor/\_\_init\_\_.py |        1 |        0 |        0 |        0 |    100% |           |
| src/bam2tensor/\_\_main\_\_.py |      153 |       16 |       60 |        8 |     87% |70-72, 184, 187, 331-337, 347, 351, 370, 483 |
| src/bam2tensor/embedding.py    |       98 |        0 |       42 |        0 |    100% |           |
| src/bam2tensor/functions.py    |      173 |       17 |       92 |       11 |     87% |87, 95-96, 99, 316-320, 357, 366, 390->403, 416->430, 425->430, 489, 514-516, 518-521 |
| src/bam2tensor/reference.py    |       63 |       29 |       16 |        2 |     53% |77-79, 177->179, 181-225 |
| tests/\_\_init\_\_.py          |        0 |        0 |        0 |        0 |    100% |           |
| tests/test\_duplication.py     |       27 |        0 |        0 |        0 |    100% |           |
| tests/test\_embedding.py       |       69 |        0 |        4 |        0 |    100% |           |
| tests/test\_functions.py       |      291 |        0 |       14 |        2 |     99% |139->155, 517->532 |
| tests/test\_main.py            |       92 |        0 |        0 |        0 |    100% |           |
| tests/test\_reference.py       |       67 |        0 |        4 |        0 |    100% |           |
| **TOTAL**                      | **1034** |   **62** |  **232** |   **23** | **92%** |           |


## Setup coverage badge

Below are examples of the badges you can use in your main branch `README` file.

### Direct image

[![Coverage badge](https://raw.githubusercontent.com/mcwdsi/bam2tensor/python-coverage-comment-action-data/badge.svg)](https://htmlpreview.github.io/?https://github.com/mcwdsi/bam2tensor/blob/python-coverage-comment-action-data/htmlcov/index.html)

This is the one to use if your repository is private or if you don't want to customize anything.

### [Shields.io](https://shields.io) Json Endpoint

[![Coverage badge](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/mcwdsi/bam2tensor/python-coverage-comment-action-data/endpoint.json)](https://htmlpreview.github.io/?https://github.com/mcwdsi/bam2tensor/blob/python-coverage-comment-action-data/htmlcov/index.html)

Using this one will allow you to [customize](https://shields.io/endpoint) the look of your badge.
It won't work with private repositories. It won't be refreshed more than once per five minutes.

### [Shields.io](https://shields.io) Dynamic Badge

[![Coverage badge](https://img.shields.io/badge/dynamic/json?color=brightgreen&label=coverage&query=%24.message&url=https%3A%2F%2Fraw.githubusercontent.com%2Fmcwdsi%2Fbam2tensor%2Fpython-coverage-comment-action-data%2Fendpoint.json)](https://htmlpreview.github.io/?https://github.com/mcwdsi/bam2tensor/blob/python-coverage-comment-action-data/htmlcov/index.html)

This one will always be the same color. It won't work for private repos. I'm not even sure why we included it.

## What is that?

This branch is part of the
[python-coverage-comment-action](https://github.com/marketplace/actions/python-coverage-comment)
GitHub Action. All the files in this branch are automatically generated and may be
overwritten at any moment.