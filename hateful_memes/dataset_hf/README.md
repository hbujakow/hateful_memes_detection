### Firstly prepare `data_hf` folder using `data_prep_hf.py` script.

### Then update `data_dir` variable inside `dataset_hateful_memes.py` to point to `data_hf` folder.

### Then you can load dataset using `load_dataset` from `datasets` library.

```python
from datasets import load_dataset


path_to_dataset_hateful_memes = ".../dataset_hf/dataset_hateful_memes.py"
dataset = load_dataset(path_to_dataset_hateful_memes, )
```

----------------------------------------------------------------

Dataset loading script was created using tutorials:
- [this](https://huggingface.co/docs/datasets/dataset_script#create-a-dataset-loading-script)
- [this](https://huggingface.co/datasets/cifar10/blob/main/cifar10.py)
- [this](https://huggingface.co/datasets/squad/blob/main/squad.py)