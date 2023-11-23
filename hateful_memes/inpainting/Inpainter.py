import torch
import torchvision.transforms as T
import numpy as np
from PIL import Image, ImageDraw
import os
import matplotlib.pyplot as plt
from tqdm import tqdm
import easyocr
from pathlib import Path
import argparse

from model.networks import Generator

DATAPATH = Path(__file__).resolve().parent.parent / 'data'

class ImageConverter:

    def __init__(self, image=None, image_path=None, device=None):
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = device

        plt.rcParams['figure.facecolor'] = 'white'
        self.reader = easyocr.Reader(['en'])
        self.text = None

        if image is not None:
            self.image = image
        elif image_path is not None:
            self.image = Image.open(image_path)
        else:
            raise ValueError("Image or image path must be provided.")

    def retrieve_text(self):
        if self.text is None:
            self.create_mask()
        return self.text
        # ocr_result = self.reader.readtext(np.array(self.image))
        # return ocr_result

    def create_bounding_boxes(self, ocr_mask):
        print("Creating bounding boxes...")
        res = Image.new("L", self.image.size, 0)

        draw = ImageDraw.Draw(res)
        all_text = []

        for box, _, _ in ocr_mask:
            box = [int(coord) for point in box for coord in point]  # Flatten the list of tuples
            draw.polygon(box, outline=255, fill=255)

            left, upper, right, lower = min(box[0::2]), min(box[1::2]), max(box[0::2]), max(box[1::2])
            region = self.image.crop((left, upper, right, lower))
            text = self.reader.readtext(np.array(region))
            all_text.append(text[0][1])

        self.text = ' '.join(all_text)
        return res

    def create_mask(self):
        print("Creating mask...")
        ocr_result = self.reader.readtext(np.array(self.image))
        image_mask = self.create_bounding_boxes(ocr_result)
        return image_mask


    def inpaint_image(self, model_path = 'pretrained/states_pt_places2.pth'):
        print("Inpainting image...")
        generator = Generator(checkpoint=model_path, return_flow=True, device = self.device).to(self.device)
        ocr_result = self.reader.readtext(np.array(self.image))
        img_pil = self.image.convert('RGB')
        mask = self.create_bounding_boxes(ocr_result)
        img = T.ToTensor()(img_pil).to(self.device)
        mask = T.ToTensor()(mask).to(self.device)
        inpainted_img = generator.infer(img, mask, return_vals=['inpainted'])

        return Image.fromarray(inpainted_img)



def generate_inpainted_images(img_dir, inpainted_img_dir, device=None):
    if not os.path.exists(inpainted_img_dir):
        os.mkdir(inpainted_img_dir)

    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    for filename in tqdm(os.listdir(img_dir)):
        img = Image.open(img_dir / filename)
        converter = ImageConverter(device=device, image=img)
        inpainted_img = converter.inpaint_image()
        inpainted_img.save(inpainted_img_dir / filename, 'PNG')
        print(f"Saved {filename} to {inpainted_img_dir}.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--img_dir', type=str, default=DATAPATH / 'img/')
    parser.add_argument('--inpainted_img_dir', type=str, default=DATAPATH / 'inpainted_img/')
    parser.add_argument('--device', type=str, default=None)
    args = parser.parse_args()
    generate_inpainted_images(args.img_dir, args.inpainted_img_dir)
    # img = Image.open(args.img_dir / '01358.png')
    # converter = ImageConverter(device=args.device, image=img)
    # print(converter.retrieve_text())