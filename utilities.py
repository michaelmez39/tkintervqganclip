from omegaconf import OmegaConf
import yaml
from taming.models.vqgan import VQModel
from taming.models import cond_transformer
import torch
import sys
import os
sys.path.append(os.path.abspath('./prompts.py'))
from prompts import replace_grad



def vector_quantize(x, codebook):
  d = x.pow(2).sum(dim=-1, keepdim=True) + codebook.pow(2).sum(dim=1) - 2 * x @ codebook.T
  indices = d.argmin(-1)
  x_q = torch.nn.functional.one_hot(indices, codebook.shape[0]).to(d.dtype) @ codebook
  return replace_grad(x_q, x)

def load_vqgan(config, ckpt_path=None):
  if config.model.target == 'taming.models.cond_transformer.Net2NetTransformer':
    parent_model = cond_transformer.Net2NetTransformer(**config.model.params)
    parent_model.eval().requires_grad_(False)
    parent_model.init_from_ckpt(ckpt_path)
    model = parent_model.first_stage_model
  else:
    model = VQModel(**config.model.params)
    model.eval().requires_grad_(False)
    model.init_from_ckpt(ckpt_path)

  del model.loss
  return model

def load_vqgan_config(config_path, display=False):
  config = OmegaConf.load(config_path)
  if display:
    print(yaml.dump(OmegaConf.to_container(config)))
  return config