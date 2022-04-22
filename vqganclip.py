# Add vqgan and CLIP directory to the path
import sys
import os
sys.path.append(os.path.abspath('./taming-transformers'))
sys.path.append(os.path.abspath('./utilities.py'))
sys.path.append(os.path.abspath('./prompt.py'))
# Torch stuff
import torch
import torchvision
import torchvision.transforms as transforms
from torchvision.transforms import functional as TF

device = "cpu" #torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)
# Some utilities
import numpy as np
import yaml
from utilities import load_vqgan, load_vqgan_config, vector_quantize, random_noise
from tqdm.notebook import trange, tqdm

# Prompt Model
from prompts import Prompt, clamp_with_grad, replace_grad, MakeCutouts
# For vqgan
from omegaconf import OmegaConf
from taming.models.vqgan import VQModel
from PIL import Image

# For clip
import clip

# Synthesize the image from z
def synth(z, model):
    z_q = vector_quantize(z.movedim(1, 3), model.quantize.embedding.weight).movedim(3, 1)
    return clamp_with_grad(model.decode(z_q).add(1).div(2), 0, 1)

normalize = torchvision.transforms.Normalize(mean=[0.48145466, 0.4578275, 0.40821073],
                                            std = [0.26862954, 0.26130258, 0.27577711])
def ascend_txt(i, out, prompt_models, perceptor, make_cutouts):
    iii = perceptor.encode_image(normalize(make_cutouts(out))).float()
    results = []
    for prompt in prompt_models:
        results.append(prompt(iii))
    return results

def train(config, locked_image, opt, z, prompts, model, perceptor):
    make_cutouts = MakeCutouts(config.cut_size, 64, cut_pow=1)
    for i in trange(config.max_iterations):
        opt.zero_grad()
        out = synth(z, model)
        loss_all = ascend_txt(i, out, prompts, perceptor, make_cutouts)
        loss = sum(loss_all)
        loss.backward()
        opt.step()
        
        with torch.no_grad():
            if i % 1 == 0:
                image = TF.to_pil_image(out[0].cpu())
                image.save(config.output_dir + "/" + "iter" + str(i) + ".png")
                if locked_image is not None:
                    locked_image.locked_update(image)
                losses_str = ', '.join(f'{l.item():g}' for l in loss_all)
                tqdm.write(f'i: {i}, loss: {sum(loss_all).item():g}, losses: {losses_str}')
            z.copy_(z.maximum(config.z_min).minimum(config.z_max))

class Config:
    def __init__(self, img_path, img_width, img_height, step_size, prompts, cut_size, z_min, z_max, max_iterations, output_dir):
        self.img_path = img_path
        self.img_width = img_width
        self.img_height = img_height
        self.step_size = step_size
        self.prompts = prompts
        self.cut_size = cut_size
        self.max_iterations = max_iterations
        self.output_dir = output_dir
        self.z_min = z_min
        self.z_max = z_max

def generate_image(text_prompts, locked_image):
    with open('config.yaml', 'r') as file:
        config = yaml.safe_load(file)

    img_path = config["image"]["path"]
    img_width = config["image"]["width"]
    img_height = config["image"]["height"]
    clip_model = config["clip"]["model"]
    vqgan_model = config["vqgan"]["model"]
    vqgan_ckpt = config["vqgan"]["checkpoint"]
    step_size = config["model"]["step_size"]
    prompts_arr = config["model"]["prompts"]
    max_iterations = config["model"]["iterations"]
    output_dir = config["model"]["output_dir"]
    # Load the VQGAN model
    vqgan_config = load_vqgan_config(vqgan_model)
    model = load_vqgan(vqgan_config, ckpt_path=vqgan_ckpt).requires_grad_(False)
    model = model.to(device)

    # Load the CLIP model, called perceptor
    perceptor, preprocess = clip.load(clip_model, device=device)
    perceptor.requires_grad_(False).to(device)
    cut_size = perceptor.visual.input_resolution

    # We need to make an initial guess for vqgan to start from
    # If the user provided an image in config, use that
    # otherwise, fill with random integers!
    z = None
    e_dim = 256
    z_min = model.quantize.embedding.weight.min(dim=0).values[None, :, None, None]
    z_max = model.quantize.embedding.weight.max(dim=0).values[None, :, None, None]

    if img_path is not None:
        pil_img = Image.open(img_path)
        pil_img = pil_img.resize((img_width, img_height), Image.LANCZOS)
        pil_img = transforms.ToTensor()(pil_img)
        z, *_ = model.encode(pil_img.to(device).unsqueeze(0) * 2 - 1)
    else:
        pil_img = random_noise(img_width, img_height)
        pil_img = pil_img.resize((img_width, img_height), Image.LANCZOS)
        pil_img = transforms.ToTensor()(pil_img)
        z, *_ = model.encode(pil_img.to(device).unsqueeze(0) * 2 - 1)
    z_orig = z.clone()
    z = z.to(device)
    z.requires_grad_(True)


    opt = torch.optim.Adam([z], lr=step_size)
    # Make the configuration for training
    config = Config(img_path, img_width, img_height, step_size, prompts_arr, cut_size, z_min, z_max, max_iterations, output_dir)

    # Load the prompts for the image
    prompt_models = []
    for prompt in text_prompts:
        print("Prompt", prompt)
        embed = perceptor.encode_text(clip.tokenize(prompt).to(device)).float()
        prompt_models.append(Prompt(embed))
    train(config, locked_image, opt, z, prompt_models, model, perceptor)

if __name__ == "__main__":
    generate_image(["new york city art deco"], None)