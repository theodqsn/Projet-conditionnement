import torch  
import torch.nn as nn  
import torch.nn.functional as f 
import torch.optim as optim  
from torch.distributions import Normal 
from tqdm import tqdm  # barre de progression
import matplotlib.pyplot as plt  
import random
import numpy as np 
import torch
import os
import time
import copy

import arrangement_legitimite
import outils_courants
import outils_dessin
import pertes_2d
import perturbations
import calcul_marges
import gestion_arrangements
import penalisation_pertes
import outils_pertes
import final_monos
import minimiseur
import bibos
import minimiseur_bibos
import reglages_bibos
import gestion_nn
import gestion_reglages
import minimiseur_bibos_nouvelle_generation
import pack_unpack
import NN_ancienne_version


# === Imports étoilés ===
from arrangement_legitimite import * 
from outils_courants import *
from outils_dessin import *
from pertes_2d import *
from perturbations import *
from scipy.optimize import minimize, approx_fprime
from calcul_marges import *
from gestion_arrangements import *
from penalisation_pertes import *
from outils_pertes import * 
from final_monos import *
from minimiseur import *
from bibos import *
from minimiseur_bibos import *
from gestion_nn import *
from gestion_reglages import *
from minimiseur_bibos_nouvelle_generation import *
from pack_unpack import *
from NN_ancienne_version import *


#fonctions cibles

def wrapper_evalue_simple(args):
    entree, dims, petit_rayon, k, reg = args
    return evalue_simple(args)
  
def evalue_simple(args):
  entree, dims, petit_rayon, n_explos ,reglages_min = args
  # rapide à évaluer, découpe l'espace en 3 zones
  entree = entree.detach().numpy() 
  x,y = dims
  separation = x/y
  
  if separation >= 1.2:
    base = np.array([0.8]*len(entree))
    return(-100*np.linalg.norm(base - entree) + 100)
  elif separation <= 0.8:
    base = np.array([0.2]*len(entree))
    return(-100*np.linalg.norm(base - entree) + 100)
  else : 
    base = np.array([0]*len(entree))
    return(-100*np.linalg.norm(base - entree)+100)
  
def evalue(entree, dims, petit_rayon, n_explos ,reglages_min):
    reglages_minimisation = copy.deepcopy(reglages_min)
    axe = [0,0,1]
    valeur =  0.5
    x,y = dims
    centres_orientations = interpretation(entree, dims, n_explos)
    centres, orientations = t_2_d_bibos( injection(centres_orientations, axe, valeur))
    # On renormalise les centres afin qu'ils se trouvent tous dans dims
    arrangement = { 'nom' :'évaluation'}
    arrangement['dimensions du carton'] = [x,y,1]
    arrangement['petit rayon'] = petit_rayon
    arrangement['centres'] = centres
    arrangement['orientations'] = orientations
    arrangement['axe'] = [0,0,1]
    arrangement['grand rayon'] = 0
    arrangement['longueur']= 1
    sortie = inch_allah_lisse_bibos (arrangement, reglages_minimisation)
    return sortie['grand rayon']

def wrapper_evalue(args):
    entree, dims, petit_rayon, k , reg = args
    return evalue(entree, dims, petit_rayon, k, reg)

def peche_arrangement():
  peche_arrangement.liste_arrangements = []

def evalue_arrg(entree, dims, petit_rayon, n_explos ,reglages_min):
    reglages_minimisation = copy.deepcopy(reglages_min)
    axe = [0,0,1]
    valeur =  0.5
    x,y = dims
    centres_orientations = interpretation(entree, dims, n_explos)
    centres, orientations = t_2_d_bibos( injection(centres_orientations, axe, valeur))
    # On renormalise les centres afin qu'ils se trouvent tous dans dims
    arrangement = { 'nom' :'évaluation'}
    arrangement['dimensions du carton'] = [x,y,1]
    arrangement['petit rayon'] = petit_rayon
    arrangement['centres'] = centres
    arrangement['orientations'] = orientations
    arrangement['axe'] = [0,0,1]
    arrangement['grand rayon'] = 0
    arrangement['longueur']= 1
    sortie = inch_allah_lisse_bibos (arrangement, reglages_minimisation)
    evalue_arrg.arrg = sortie
    return sortie['grand rayon']

















