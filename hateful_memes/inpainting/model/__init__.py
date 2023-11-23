# FROM https://github.com/nipponjo/deepfillv2-pytorch/blob/master/model/__init__.py

import torch
from model.networks import Generator

def load_model(path, device='cuda'):  
    try:  
        gen_sd = torch.load(path)['G']
    except FileNotFoundError:
        return None

    gen = Generator(cnum_in=5, cnum=48, return_flow=False)
    gen = gen.to(device)
    gen.eval()

    gen.load_state_dict(gen_sd, strict=False)
    return gen
