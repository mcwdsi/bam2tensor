# Repository Coverage

[Full report](https://htmlpreview.github.io/?https://github.com/mcwdsi/bam2tensor/blob/python-coverage-comment-action-data/htmlcov/index.html)

| Name                           |    Stmts |     Miss |   Branch |   BrPart |   Cover |   Missing |
|------------------------------- | -------: | -------: | -------: | -------: | ------: | --------: |
| src/bam2tensor/\_\_init\_\_.py |        1 |        0 |        0 |        0 |    100% |           |
| src/bam2tensor/\_\_main\_\_.py |      217 |        2 |       90 |        2 |     99% |  540, 796 |
| src/bam2tensor/embedding.py    |      174 |        0 |       78 |        0 |    100% |           |
| src/bam2tensor/functions.py    |      554 |       16 |      268 |       16 |     96% |231, 235-237, 601, 614-617, 781, 885-\>892, 1112, 1180, 1202-\>1211, 1233, 1241, 1245, 1281, 1290-\>1298, 1337, 1505-\>1119, 1960-\>1970, 2090-\>2092, 2093-\>2095 |
| src/bam2tensor/inspect.py      |       91 |        2 |       48 |        4 |     96% |130-\>132, 134-\>140, 156, 205 |
| src/bam2tensor/metadata.py     |       43 |        0 |       10 |        0 |    100% |           |
| src/bam2tensor/reference.py    |       70 |        0 |       20 |        0 |    100% |           |
| **TOTAL**                      | **1150** |   **20** |  **514** |   **22** | **97%** |           |


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