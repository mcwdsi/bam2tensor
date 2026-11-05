# Repository Coverage

[Full report](https://htmlpreview.github.io/?https://github.com/mcwdsi/bam2tensor/blob/python-coverage-comment-action-data/htmlcov/index.html)

| Name                           |    Stmts |     Miss |   Branch |   BrPart |   Cover |   Missing |
|------------------------------- | -------: | -------: | -------: | -------: | ------: | --------: |
| src/bam2tensor/\_\_init\_\_.py |        1 |        0 |        0 |        0 |    100% |           |
| src/bam2tensor/\_\_main\_\_.py |      180 |        2 |       70 |        2 |     98% |  424, 633 |
| src/bam2tensor/embedding.py    |      160 |        0 |       74 |        0 |    100% |           |
| src/bam2tensor/functions.py    |      321 |       15 |      156 |       12 |     94% |149, 153-155, 419, 432-435, 578, 586, 604, 610, 640-\>652, 647-\>652, 677, 720, 779, 1107-\>1109, 1110-\>1112 |
| src/bam2tensor/inspect.py      |       75 |        9 |       38 |        4 |     87% |124, 127, 132-134, 169-171, 175 |
| src/bam2tensor/metadata.py     |       41 |        0 |       10 |        0 |    100% |           |
| src/bam2tensor/reference.py    |       70 |        0 |       20 |        0 |    100% |           |
| **TOTAL**                      |  **848** |   **26** |  **368** |   **18** | **96%** |           |


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