import torch

from .base import GPTBase, LayerNorm
from .gpt import GPTBase as gptmoe
from .llama import Llama, RMSNorm

BLACKLIST_WEIGHT_MODULES = (
    torch.nn.LayerNorm,
    LayerNorm,
    RMSNorm,
    torch.nn.Embedding,
)


def get_model(args):
    """Return the right model"""
    if args.model == "base":
        model = GPTBase(args)
        return model
    elif args.model == "llama2":
        model = Llama(args)
        return model
    elif args.model == "gpt":
        model = gptmoe(args)
        return model
    else:
        raise KeyError(f"Unknown model '{args.model}'.")
