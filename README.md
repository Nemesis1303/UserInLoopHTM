# UserInLoopHTM

[![PyPI - License](https://img.shields.io/badge/license-MIT-green.svg)](https://github.com/Nemesis1303/UserInLoopHTM/blob/main/LICENSE)

This GitHub repository hosts two novel implementations for the construction of hierarchical topic models (HTMs), HTM with word selection (HTM-WS) and HTM with document selection (HTM-DS), which can be used to build hierarchical extensions of existing topic modeling algorithms.

## Overview

Both methods follow a three-step process:

1. Construction of a level-1 topic model with any existing topic modeling algorithm and selection of a topic for expansion.
2. Construction of the synthetic training corpus for the level-2 topic model according to HTM-WS or HTM-DS principles.
3. Training of the synthetic corpus to generate the level-2 model.

At the time being, this repository provides implementations for:

|                     Name                     |                         Source code                         |
| :------------------------------------------: | :---------------------------------------------------------: |
|    **CTMs** `(Bianchi et al. 2020, 2021)`    | <https://github.com/MilaNLProc/contextualized-topic-models> |
|  **LDA** `(Blei 2003)`  |       <https://mimno.github.io/Mallet/topics.html>       |


