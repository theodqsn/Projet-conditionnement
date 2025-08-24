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
import NN_fonctions_a_evaluer 
import NN_construction_entrainement 
import NN_ancienne_version 
from collections import deque
from concurrent.futures import ProcessPoolExecutor
import threading


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
from NN_fonctions_a_evaluer import *
from NN_construction_entrainement import *
from NN_ancienne_version import *

class TesterReseau:
    def __init__(self, nom_reseau, fonction_evaluation, nktbqreg=None):
      if isinstance(nom_reseau, str):
        self.nom_reseau = nom_reseau
        self.reglages = None
        self.reseau = None
      else : #on part du principe qu'on a direct donné le réseau
        self.reseau = nom_reseau
        self.nom_reseau = None
        self.reglages = nktbqreg
      self.entrees = None
      self.fonction_evaluation = fonction_evaluation
      self.sortie_brutes = None
      self.resultats = []
       
    def recuperer_reseau(self):
      if self.nom_reseau is not None:
        reseau = charger_nn(self.nom_reseau)
        lien_reglages = charger_nn.lien_reglages
        n = reglages_internes('nombre de rouleaux', lien_reglages)
        k = reglages_internes('nombre explorateurs')
        regmin = reglages_internes('reglages minimisation')
        taille_batch = None
        q = None
        self.reglages = n,k,taille_batch, q, regmin
        self.reseau = reseau
        
    def ajouter_entrees(self, entrees):
      entrees = torch.tensor(entrees)
      if len(entrees.shape) == 1:
        entrees = torch.stack([entrees])
      self.entrees = entrees
      n,k,taille_batch, q, regmin = self.reglages
      if taille_batch is None:
        taille_batch = entrees.shape[0]
      if q is None :
        q = 1
      self.reglages = n,k,taille_batch, q, regmin

    def construire_strategie(self):
      n,k,taille_batch,q, regmin = self.reglages
      reseau  = self.reseau
      strategie = Strategie(n,k,taille_batch,reseau)
      strategie.variance_a_la_main = True
      strategie.variances = 1e-5 *torch.ones((taille_batch, k,3*n))
      strategie.ajouter_echantillon_entrees(self.entrees)
      strategie.digerer()
      self.strategie = strategie

    def effectuer_evaluations(self):
      strategie = self.strategie
      n,k,taille_batch,q, regmin = self.reglages
      resultats_evaluation =  ResultatsEvaluation(strategie=strategie, evaluations_par_entree=q, fonction_evaluation= self.fonction_evaluation, reglages_min = regmin)
      resultats_evaluation.creer_eval_entrees()
      resultats_evaluation.executer_evaluations()
      self.resultats_evaluation = resultats_evaluation

    def evaluer(self, entrees):
      self.recuperer_reseau()
      self.ajouter_entrees(entrees)
      self.construire_strategie()
      self.effectuer_evaluations()
      entrees = self.entrees.tolist()
      resultats = [rec.evaluations[0] for rec in self.resultats_evaluation.evaluations_entrees]
      self.resultats = resultats
      return entrees, resultats



      