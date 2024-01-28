from pbm import PromptHateModel


def test_prompt_hate_model_forward():
    """
    Tests the forward method of the PromptHateModel class.
    """
    model = PromptHateModel()

    all_texts = [
        "This is a test sentence, its meaning is hateful. It was <mask>",
        "This is a second test sentence, its meaning is non hateful. It was <mask>",
    ]

    logits = model.forward(all_texts)

    assert logits.shape == (len(all_texts), 2)
