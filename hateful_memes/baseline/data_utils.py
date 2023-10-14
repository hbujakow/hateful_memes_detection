import pandas as pd
import torch
from PIL import Image


class HatefulMemesDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        data_path,
        img_dir,
        image_transform,
        text_transform,
        balance=False,
        dev_limit=None,
        random_state=0,
    ):
        self.samples_frame = pd.read_json(data_path, lines=True)
        self.dev_limit = dev_limit

        if balance:
            neg = self.samples_frame[self.samples_frame.label.eq(0)]
            pos = self.samples_frame[self.samples_frame.label.eq(1)]
            self.samples_frame = pd.concat(
                [neg.sample(pos.shape[0], random_state=random_state), pos]
            )

        if self.dev_limit:
            if self.samples_frame.shape[0] > self.dev_limit:
                self.samples_frame = self.samples_frame.sample(
                    dev_limit, random_state=random_state
                )
        self.samples_frame = self.samples_frame.reset_index(drop=True)
        self.samples_frame.img = self.samples_frame.apply(
            lambda row: (img_dir / row.img), axis=1
        )

        # https://github.com/drivendataorg/pandas-path
        # if not self.samples_frame.img.path.exists().all():
        # raise FileNotFoundError
        # if not self.samples_frame.img.path.is_file().all():
        # raise TypeError

        self.image_transform = image_transform
        self.text_transform = text_transform

    def __len__(self):
        """This method is called when you do len(instance)
        for an instance of this class.
        """
        return len(self.samples_frame)

    def __getitem__(self, idx):
        """This method is called when you do instance[key]
        for an instance of this class.
        """
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_id = self.samples_frame.loc[idx, "id"]

        image = Image.open(self.samples_frame.loc[idx, "img"]).convert("RGB")
        image = self.image_transform(image)

        text = torch.Tensor(
            self.text_transform.get_sentence_vector(self.samples_frame.loc[idx, "text"])
        ).squeeze()

        if "label" in self.samples_frame.columns:
            label = (
                torch.Tensor([self.samples_frame.loc[idx, "label"]]).long().squeeze()
            )
            sample = {"id": img_id, "image": image, "text": text, "label": label}
        else:
            sample = {"id": img_id, "image": image, "text": text}

        return sample
