import torch

def multinomial_select(probas: torch.Tensor):
    return torch.multinomial(probas, 1).cpu().numpy()