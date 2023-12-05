import pytest
import easyocr
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from Inpainter import ImageConverter

inpainter = ImageConverter()


@pytest.fixture
def sample_image():
    """
    Creates sample image with text.
    Returns:
        PIL.Image.Image: Sample image.
    """
    # create a sample image
    sample_image = Image.new("RGB", (200, 200), color="white")

    # add sample text to the image
    drawer = ImageDraw.Draw(sample_image)
    font = ImageFont.truetype("arial.ttf", size=16)
    drawer.text((10, 10), "Sample meme text", fill=(0, 0, 0), font=font, size=100)

    return sample_image


def test_upload_img(sample_image):
    """
    Tests if the image is uploaded correctly to the inpainter.
    """
    inpainter.upload_img(sample_image)
    assert inpainter.image is not None
    assert isinstance(inpainter.image, Image.Image)


def test_image_inpainting(sample_image):
    """
    Tests if the image is inpainted correctly by the inpainter,
    i.e. there is no text on the inpainted image.
    """
    inpainter.upload_img(sample_image)
    inpainted_image = inpainter.inpaint_image()

    text_on_inpainted_image = easyocr.Reader(["en"]).readtext(np.array(inpainted_image))
    assert text_on_inpainted_image == []


def test_extracted_text(sample_image):
    """
    Tests if the extracted text from the image is correct.
    """
    inpainter.upload_img(sample_image)
    text = inpainter.retrieve_text()
    assert text == "Sample meme text"
