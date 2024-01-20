import os
import random

import config
import numpy as np
import torch
from dataset import MultiModalData
from pbm import PromptHateModel
from torch.utils.data import DataLoader
from train import train_for_epoch


def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


if __name__ == "__main__":
    opt = config.parse_opt()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    torch.cuda.set_device(device)
    set_seed(opt.SEED)

    train_set = MultiModalData(opt, "train")
    dev_set = MultiModalData(opt, "dev")
    test_set = MultiModalData(opt, "test")

    max_length = opt.LENGTH + opt.CAP_LENGTH
    if opt.ASK_CAP != "":
        num_ask_cap = len(opt.ASK_CAP.split(","))
        print("Number of asking captions", num_ask_cap)
        all_cap_len = opt.CAP_LENGTH * num_ask_cap  # default, 12*5=60
        max_length += all_cap_len
    if opt.USE_DEMO:
        max_length *= opt.NUM_SAMPLE * opt.NUM_LABELS + 1

    label_words = [opt.POS_WORD, opt.NEG_WORD]

    model = PromptHateModel(label_words, max_length, model_name=opt.MODEL_NAME).cuda()

    train_loader = DataLoader(train_set, opt.BATCH_SIZE, shuffle=True, num_workers=2)
    dev_loader = DataLoader(dev_set, opt.BATCH_SIZE, shuffle=False, num_workers=2)
    test_loader = DataLoader(test_set, opt.BATCH_SIZE, shuffle=False, num_workers=2)
    train_for_epoch(opt, model, train_loader, dev_loader, test_loader)

    exit(0)
