import json
import os
from pathlib import Path

import datasets
import pandas as pd

# based on this:https://github.com/huggingface/datasets/blob/main/templates/new_dataset_script.py
# https://huggingface.co/docs/datasets/dataset_script

_DESCRIPTION = """\
This dataset is Hateful Memes dataset. It is a multimodal dataset containing image and text.
"""

# TODO: Add the licence for the dataset here if you can find it
# _LICENSE = ""

# # TODO: Add link to the official dataset URLs here
# # The HuggingFace Datasets library doesn't host the datasets but only points to the original files.
# # This can be an arbitrary nested dict/list of URLs (see below in `_split_generators` method)
# _URLS = {
#     "first_domain": "https://huggingface.co/great-new-dataset-first_domain.zip",
#     "second_domain": "https://huggingface.co/great-new-dataset-second_domain.zip",
# }


class HatefulMemesDataset(datasets.GeneratorBasedBuilder):
    # """TODO: Short description of my dataset."""

    VERSION = datasets.Version("1.1.0")
    # This is an example of a dataset with multiple configurations.
    # If you don't want/need to define several sub-sets in your dataset,
    # just remove the BUILDER_CONFIG_CLASS and the BUILDER_CONFIGS attributes.

    # If you need to make complex sub-parts in the datasets with configurable options
    # You can create your own builder configuration class to store attribute, inheriting from datasets.BuilderConfig
    # BUILDER_CONFIG_CLASS = MyBuilderConfig

    # You will be able to load one or the other configurations in the following list with
    # data = datasets.load_dataset('my_dataset', 'first_domain')
    # data = datasets.load_dataset('my_dataset', 'second_domain')
    BUILDER_CONFIGS = [
        datasets.BuilderConfig(
            name="first_domain",
            version=VERSION,
            description="This part of my dataset covers a first domain",
        ),
        # datasets.BuilderConfig(
        #     name="second_domain",
        #     version=VERSION,
        #     description="This part of my dataset covers a second domain",
        # ),
    ]

    DEFAULT_CONFIG_NAME = "first_domain"  # It's not mandatory to have a default configuration. Just use one if it make sense.

    def _info(self):
        if (
            self.config.name == "first_domain"
        ):  # This is the name of the configuration selected in BUILDER_CONFIGS above
            features = datasets.Features(
                {
                    "image": datasets.Image(),
                    "caption": datasets.Value("string"),
                }
            )
        # else:  # This is an example to show how to have different features for "first_domain" and "second_domain"
        # features = datasets.Features(
        # {
        # "sentence": datasets.Value("string"),
        # "option2": datasets.Value("string"),
        # "second_domain_answer": datasets.Value("string")
        # These are the features of your dataset like images, labels ...
        # }
        # )
        return datasets.DatasetInfo(
            # This is the description that will appear on the datasets page.
            description=_DESCRIPTION,
            # This defines the different columns of the dataset and their types
            features=features,  # Here we define them above because they are different between the two configurations
            # If there's a common (input, target) tuple from the features, uncomment supervised_keys line below and
            # specify them. They'll be used if as_supervised=True in builder.as_dataset.
            # supervised_keys=("sentence", "label"),
            # Homepage of the dataset for documentation
            # homepage=_HOMEPAGE,
            # License for the dataset if available
            # license=_LICENSE,
            # Citation for the dataset
            # citation=_CITATION,
        )

    def _split_generators(self, dl_manager):
        # TODO: This method is tasked with downloading/extracting the data and defining the splits depending on the configuration
        # If several configurations are possible (listed in BUILDER_CONFIGS), the configuration selected by the user is in self.config.name

        # dl_manager is a datasets.download.DownloadManager that can be used to download and extract URLS
        # It can accept any type or nested list/dict and will give back the same structure with the url replaced with path to local files.
        # By default the archives will be extracted and a path to a cached folder where they are extracted is returned instead of the archive
        # urls = _URLS[self.config.name]
        data_dir = Path("o:\\memes_analysis\\data\data_hf")
        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                # These kwargs will be passed to _generate_examples
                gen_kwargs={
                    "filepath": data_dir / "train",
                    "split": "train",
                },
            ),
            datasets.SplitGenerator(
                name=datasets.Split.VALIDATION,
                # These kwargs will be passed to _generate_examples
                gen_kwargs={
                    "filepath": data_dir / "dev",
                    "split": "dev",
                },
            ),
            datasets.SplitGenerator(
                name=datasets.Split.TEST,
                # These kwargs will be passed to _generate_examples
                gen_kwargs={
                    "filepath": data_dir / "test",
                    "split": "test",
                },
            ),
        ]

    # method parameters are unpacked from `gen_kwargs` as given in `_split_generators`
    def _generate_examples(self, filepath, split):
        # TODO: This method handles input defined in _split_generators to yield (key, example) tuples from the dataset.

        metadata = pd.read_csv(filepath / "metadata.tsv", sep="\t")
        for key, row in metadata.iterrows():
            data = row
            yield key, {
                "image": str(filepath / data["file_name"]),
                "caption": data["img_text"],
            }
            # else:
            # yield key, {
            #     "sentence": data["sentence"],
            #     "option2": data["option2"],
            #     "second_domain_answer": ""
            #     if split == "test"
            #     else data["second_domain_answer"],
            # }
