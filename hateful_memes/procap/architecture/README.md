## Setup

### Config

Inside the config there are hardcoded paths to the data. (captions)

Despite Pro-Cap captions you need generic captions (there are also inside my folder `/home2/faculty/mgalkowski/memes_analysis/data/hateful_memes/captions`).

You also need data from `hateful_memes/Pro-Cap/Data/mem*`. I have put them inside `/home2/faculty/mgalkowski/memes_analysis/data/hateful_memes/data/hateful_memes/domain_splits/`. It is important that they are inside `domain_splits` folder.

## Training

#### Before running training script you need to generate captions for the images.

Modify paths in `generate_captions.py` and run it.

```bash
sbatch generate_captions.sh
```

### Modify those files before running training script

`config.py` - modify it to your needs.

`train_model.sh` - modify it to your needs.

### Run training script

```bash
sbatch train_model.sh
```
