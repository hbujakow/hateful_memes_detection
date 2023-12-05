import pytest
from config import parse_opt
from dataset import MultiModalData


@pytest.fixture
def dataset():
    opt = parse_opt()
    return MultiModalData(opt, mode="train")


def test_load_entries(dataset):
    entries = dataset.load_entries("train")
    assert len(entries) > 0
    sample_entry = entries[0]
    assert "img" in sample_entry
    assert "label" in sample_entry
    assert "meme_text" in sample_entry
    assert "cap" in sample_entry


def test_select_context(dataset):
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


def test_process_prompt(dataset):
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


def test_len(dataset):
    length = dataset.__len__()
    assert length >= 0
