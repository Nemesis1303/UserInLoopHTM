# evaluateNN

This directory contains a submodule of the [OCTIS](https://github.com/MIND-Lab/OCTIS) repository, with some modifications to adapt it to the specific purposes of UserInLoopHTM. In particular, the following modifications have been made:

1. In ``OCTIS/octis/dataset/dataset.py``, the following functions have been added:

   * ``load_custom_dataset_from_parquet`` to create a Dataset object created from a corpus given in parquet format.
   * ``get_partitioned_embeddings`` to get the corpus corresponding partitions of the documents if provided in the input parquet file.
   * ``get_embeddings`` to get the array of embeddings,
2. ``OCTIS/octis/models/CTM.py`` has been modified so as to include two dropounts as hyperparameters: one for the thetas, and another one for the encoder. Changes in the CTM module has been added accordingly in order to support the latter.

The script ``evaluate.py`` carries out the evaluation, which can be customized as needed.
