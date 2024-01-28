import argparse
import os
from pathlib import Path
import easyocr
import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision.transforms as T
from PIL import Image, ImageDraw
from tqdm import tqdm
from model.networks import Generator
from typing import List, Tuple


DATAPATH = Path(__file__).resolve().parent.parent / "data"
plt.rcParams["figure.facecolor"] = "white"


class ImageConverter:
    def __init__(
        self,
        model_path: Path = Path(__file__).resolve().parent
        / "pretrained"
        / "states_pt_places2.pth",
        device: str = None,
    ):
        """
        Class for inpainting images and extracting text using OCR.
        Args:
            model_path (Path): Path to the model checkpoint.
            device (str): Device to use for inference.
        """

        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = device
        self.inpainting_model = Generator(
            checkpoint=model_path, return_flow=True, device=self.device
        ).to(self.device)
        self.reader = easyocr.Reader(["en"])

    def upload_img(self, image: Image.Image = None, image_path: str = None):
        """
        Uploads image to the inpainter.
        Args:
            image (Image.Image): PIL image.
            image_path (str): Path to image.

        """
        self.text = None

        if image is not None:
            self.image = image
        elif image_path is not None:
            self.image = Image.open(image_path)
        else:
            raise ValueError("Image or image path must be provided.")

    def retrieve_text(self) -> str:
        """
        Retrieves text extracted by OCR from the image.
        """
        if self.text is None:
            self.create_mask()
        return self.text

    def create_bounding_boxes(
        self, ocr_mask: List[Tuple[float, ...], Tuple[float, ...], Tuple[float, ...]]
    ) -> Image.Image:
        """
        Creates bounding boxes around the text in the image.
        Args:
            ocr_mask (List[Tuple[float, ...], Tuple[float, ...], Tuple[float, ...]]): OCR mask.
        Returns:
            Image.Image: Image with bounding boxes.
        """
        print("Creating bounding boxes...")
        res = Image.new("L", self.image.size, 0)

        draw = ImageDraw.Draw(res)
        all_text = []

        for box, _, _ in ocr_mask:
            box = [
                int(coord) for point in box for coord in point
            ]  # Flatten the list of tuples
            draw.polygon(box, outline=255, fill=255)

            left, upper, right, lower = (
                min(box[0::2]),
                min(box[1::2]),
                max(box[0::2]),
                max(box[1::2]),
            )
            region = self.image.crop((left, upper, right, lower))
            text = self.reader.readtext(np.array(region))
            if text != []:
                all_text.append(text[0][1])

        self.text = " ".join(all_text) if all_text != [] else ""
        return res

    def create_mask(self) -> Image.Image:
        """
        Creates mask for the image.
        """
        print("Creating mask...")
        ocr_result = self.reader.readtext(np.array(self.image))
        image_mask = self.create_bounding_boxes(ocr_result)
        return image_mask

    def inpaint_image(self) -> Image.Image:
        """
        Inpaints the image using the inpainting model.
        Returns:
            Image.Image: Inpainted image.
        """

        print("Inpainting image...")
        ocr_result = self.reader.readtext(np.array(self.image))
        img_pil = self.image.convert("RGB")
        mask = self.create_bounding_boxes(ocr_result)
        img = T.ToTensor()(img_pil).to(self.device)
        mask = T.ToTensor()(mask).to(self.device)
        inpainted_img = self.inpainting_model.infer(
            img, mask, return_vals=["inpainted"]
        )
        return Image.fromarray(inpainted_img)


def generate_inpainted_images(
    img_dir: str | Path, inpainted_img_dir: str | Path, device: str = None
):
    """
    Generates inpainted images from the images in the img_dir. Saves the inpainted images to the inpainted_img_dir.
    Args:
        img_dir (str | Path): Path to the directory with images.
        inpainted_img_dir (str | Path): Path to the directory where the inpainted images will be saved.
        device (str): Device to use for inference.
    """
    if not os.path.exists(inpainted_img_dir):
        os.mkdir(inpainted_img_dir)

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    for filename in tqdm(os.listdir(img_dir)):
        img = Image.open(img_dir / filename)
        converter = ImageConverter(device=device, image=img)
        inpainted_img = converter.inpaint_image()
        inpainted_img.save(inpainted_img_dir / filename, "PNG")
        print(f"Saved {filename} to {inpainted_img_dir}.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--img_dir", type=str, default=DATAPATH / "img/")
    parser.add_argument(
        "--inpainted_img_dir", type=str, default=DATAPATH / "inpainted_img/"
    )
    parser.add_argument("--device", type=str, default=None)
    args = parser.parse_args()
    generate_inpainted_images(args.img_dir, args.inpainted_img_dir)
    # img = Image.open(args.img_dir / '01358.png')
    # converter = ImageConverter(device=args.device, image=img)
    # print(converter.retrieve_text())
