import pytest
from dataset import MultiModalData
from datetime import datetime


class Config:
    def __init__(self) -> None:
        self.DATASET: str = "mem"
        self.CAP_TYPE: str = "vqa"
        self.MODEL_NAME: str = "roberta-large"
        self.DATA: str = "."
        self.CAPTION_PATH: str = "./captions"
        self.LOG_PATH: str = ""
        self.MODEL_PATH: str = ""
        self.NUM_LABELS: int = 2
        self.POS_WORD: str = "good"
        self.NEG_WORD: str = "bad"
        self.MULTI_QUERY: bool = True
        self.USE_DEMO: bool = True
        self.NUM_QUERIES: int = 4
        self.WEIGHT_DECAY: float = 0.01
        self.LR_RATE: float = 1e-5
        self.EPS: float = 1e-8
        self.BATCH_SIZE: int = 16
        self.FIX_LAYERS: int = 2
        self.NUM_SAMPLE: int = 1
        self.LENGTH: int = 65
        self.ASK_CAP: str = "race,gender,country,animal,valid_disable,religion"
        self.CAP_LENGTH: int = 12
        self.PRETRAIN_DATA: str = "conceptual"
        self.IMG_VERSION: str = "clean"
        self.ADD_ENT: bool = False
        self.ADD_DEM: bool = False
        self.DEBUG: bool = False
        self.SAVE: bool = True
        self.SAVE_NUM: str = datetime.now().strftime("%Y%m%d_%H%M")
        self.EPOCHS: int = 10
        self.SEED: int = 1111

    """
    Config class for the dataset.
    """


@pytest.fixture
def dataset():
    opt = Config()
    return MultiModalData(opt, mode="train")


def test_load_entries(dataset: MultiModalData):
    """
    Test if the entries are loaded correctly.
    """
    entries = dataset.load_entries("train")
    assert len(entries) > 0
    sample_entry = entries[0]
    assert "img" in sample_entry
    assert "label" in sample_entry
    assert "meme_text" in sample_entry
    assert "cap" in sample_entry


def test_select_context(dataset: MultiModalData):
    """
    Test if the context is selected correctly.
    """
    context_examples = [
        {"cap": "random caption 1", "meme_text": "text 1", "label": 0, "img": "1.png"},
        {"cap": "random caption 2", "meme_text": "text 2", "label": 0, "img": "2.png"},
        {"cap": "random caption 3", "meme_text": "text 3", "label": 1, "img": "3.png"},
        {"cap": "random caption 4", "meme_text": "text 4", "label": 1, "img": "4.png"},
    ]
    selection = dataset.select_context(context_examples)
    assert len(selection) == dataset.num_sample * 2
    labels_count = {"0": 0, "1": 0}
    for sample in selection:
        labels_count[str(sample["label"])] += 1
    assert len(set(labels_count.values())) == 1


def test_process_prompt(dataset: MultiModalData):
    """
    Test if the prompt is processed correctly and the test text is returned.
    """
    examples = [
        {
            "cap": "a man in a suit and a man in a suit . a black man in a suit and a white man in a suit . a man in a suit and a woman in a suit and a t-shirt with a shaved head . the person in the image is from the united states and the person in the image is from the philippines and the person in the image . a black man in a suit and a white man in a suit and a black man in a suit and a white",
            "meme_text": "its their character not their color that matters",
            "label": 0,
            "img": "42953.png",
        },
        {
            "cap": "a close up of a small animal with a leaf in its mouth . a small animal is sitting on the ground with a leaf in it's mouth . he is a buddhist and he is wearing a tibetan tibetan",
            "meme_text": "my girlfriend just freaked me out she gave me blowjob but insisted on roleplaying as a 14 year old fucking weird and gross i was like youre going to be 14 in a couple of years anyway what is the rush",
            "label": 0,
            "img": "16407.png",
        },
        {
            "cap": "a woman sitting in the driver's seat of a car . asian woman in a car with a smile on her face and a cell phone in her hand . a woman is sitting in a car with a smile on her face and a cell phone in her hand . south korea -             . a woman in a car with a tatoo on her head",
            "meme_text": "when your uber driver arrives but you are probably safer driving drunk",
            "label": 1,
            "img": "37628.png",
        },
    ]
    concate_sent, test_text = dataset.process_prompt(examples)
    assert concate_sent == [
        "its their character not their color that matters .  It was <mask> . a man in a suit and a man in a suit . a black man in a suit and a white man in a suit . a man in a suit and a woman in a suit and a t-shirt with a shaved head . the person in the image is from the united states and the person in the image is from the philippines and the person in the image . a black man in a suit and a white man in a suit and a black man in a suit and a white",
        "my girlfriend just freaked me out she gave me blowjob but insisted on roleplaying as a 14 year old fucking weird and gross i was like youre going to be 14 in a couple of years anyway what is the rush .  It was good . a close up of a small animal with a leaf in its mouth . a small animal is sitting on the ground with a leaf in it's mouth . he is a buddhist and he is wearing a tibetan tibetan",
        "when your uber driver arrives but you are probably safer driving drunk .  It was bad . a woman sitting in the driver's seat of a car . asian woman in a car with a smile on her face and a cell phone in her hand . a woman is sitting in a car with a smile on her face and a cell phone in her hand . south korea -             . a woman in a car with a tatoo on her head",
    ]
    assert (
        test_text
        == "its their character not their color that matters . a man in a suit and a man in a suit . a black man in a suit and a white man in a suit . a man in a suit and a woman in a suit and a t-shirt with a shaved head . the person in the image is from the united states and the person in the image is from the philippines and the person in the image . a black man in a suit and a white man in a suit and a black man in a suit and a white"
    )


def test_len(dataset: MultiModalData):
    """
    Test if the length of the dataset is correct.
    """
    length = dataset.__len__()
    assert length >= 0
