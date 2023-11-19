# TODO - Refactor -> ja to mogę później ogarnąć

### Config

Inside the config there are hardcoded paths to the data. (captions)

Despite Pro-Cap captions you need generic captions (there are also inside my folder `/home2/faculty/mgalkowski/memes_analysis/data/hateful_memes/captions`).

You also need data from `hateful_memes/Pro-Cap/Data/mem*`. I have put them inside `/home2/faculty/mgalkowski/memes_analysis/data/hateful_memes/data/hateful_memes/domain_splits/`. It is important that they are inside `domain_splits` folder. [CHECK WHY HERE 75 line inside `dataset.py` file]

### Training

`train_model.sh` - modify it to your needs.