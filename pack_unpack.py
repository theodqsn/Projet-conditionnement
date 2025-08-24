import pickle
import torch
import numpy as np

def pack(obj, filename):
    """
    Sauvegarde un objet Python sérialisable (liste, dict, numpy array, tensor, etc.)
    dans un fichier binaire.
    """
    # Pour PyTorch : on convertit en numpy si possible (plus léger)
    if isinstance(obj, torch.Tensor):
        obj = ("torch", obj.detach().cpu().numpy())
    else:
        obj = ("raw", obj)
    
    with open(filename, "wb") as f:
        pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)


def unpack(filename):
    """
    Charge un objet Python depuis un fichier binaire.
    """
    with open(filename, "rb") as f:
        type_str, data = pickle.load(f)
    
    if type_str == "torch":
        return torch.from_numpy(data)
    else:
        return data
    
import torch

def clone_mutables(obj):
    """
    Clone récursivement un objet mutable contenant éventuellement
    des tensors PyTorch, en détachant les tensors du graphe d'autograd.
    
    - torch.Tensor → clone().detach(), conserve requires_grad
    - list → nouvelle liste clonée élément par élément
    - tuple → nouveau tuple cloné élément par élément
    - dict → nouveau dict cloné valeur par valeur
    - set → nouveau set cloné élément par élément
    - autres objets → retour inchangé (référence identique)
    """
    if isinstance(obj, torch.Tensor):
        return obj.clone().detach().requires_grad_(obj.requires_grad)
    
    elif isinstance(obj, list):
        return [clone_mutables(x) for x in obj]
    
    elif isinstance(obj, tuple):
        return tuple(clone_mutables(x) for x in obj)
    
    elif isinstance(obj, dict):
        return {k: clone_mutables(v) for k, v in obj.items()}
    
    elif isinstance(obj, set):
        return {clone_mutables(x) for x in obj}
    
    else:
        return obj

