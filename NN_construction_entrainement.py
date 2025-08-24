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
from NN_ancienne_version import *

class ArgumentsVariance :

    def __init__(self):
        self.liste_arguments = []

    def ajouter_argument(self, nom, argument):
        setattr(self, nom, argument)
        self.liste_arguments.append(nom)

class Strategie :

    def __init__(self, n,k,q, reseau):
        self.n = n
        self.k = k
        self.reseau = reseau
        self.moyennes = torch.zeros((q,k, 3*n)) # Pour chaque entrée, pour chaque explorateur, on a une moyenne de taille 3*n (2n pour les coordonnées et n pour les orientations)
        self.variances = torch.zeros((q,k, 3*n))
        self.probas = torch.zeros((q,k)) #Pour chaque entrée, on a un vecteur de probabilités de taille k
        self.strategies_individuelles = []
        self.variance_a_la_main = False
        self.echantillon_entrees = None

    def ajouter_echantillon_entrees(self, echantillon_entree):
        self.echantillon_entrees = echantillon_entree
    
    def ajouter_arguments_variance(self, args_variance):
        self.args_variance = args_variance

    def proposer_strategie(self, entree, indice = None):
        sortie_brute = self.reseau(entree)
        probas, moyennes = decode_nn(sortie_brute)
        variance, indice_alt = self.obtenir_variance_pour_une_entree(entree)
        if indice is None:
            indice = indice_alt
        strat_indiv = {'indice': indice, 'probas': probas, 'moyenne': moyennes, 'variance': variance, 'entree': entree}
        self.strategies_individuelles.append(strat_indiv)

    def obtenir_variance_pour_une_entree(self, entree):
        indice_entree = (self.echantillon_entrees == entree).nonzero(as_tuple=True)[0][0].item()
        return(self.variances[indice_entree], indice_entree)

    def maj_variance(self):
        if not self.variance_a_la_main:
            self.variances = gestion_variance(self.args_variance)

    def calculer_strats(self):
        # calcule les stratégies proposées par entrée 
        for entree in self.echantillon_entrees:
            self.proposer_strategie(entree)
        
    def digerer(self):
        self.maj_variance()
        self.calculer_strats()
        moyennes = []
        variances = []
        probas = []
        for strat in self.strategies_individuelles: #On a bien une strat par entrée
            moyennes.append(strat['moyenne'])
            variances.append(strat['variance'])
            probas.append(strat['probas'])
        self.moyennes = torch.stack(moyennes)
        self.variances = torch.stack(variances)
        self.probas = torch.stack(probas)

    def clean(self):
        self.strategies_individuelles = []

class ResultatsEvaluation: # Gros objet qui va se trainer tous les résultats d'évaluation de la strategie

    def __init__(self, strategie, evaluations_par_entree, fonction_evaluation, reglages_min = None):
        self.strategie = strategie
        self.nombre_d_evaluations_par_entree = evaluations_par_entree
        self.echantillon_entrees = strategie.echantillon_entrees # De taille 'taille_batch'
        self.evaluations_entrees = []
        self.fonction_evaluation = fonction_evaluation
        self.reglages_min = reglages_min
        
    def creer_eval_entrees(self):
        for strats in self.strategie.strategies_individuelles:
            self.evaluations_entrees.append(EvalEntree(strats['entree'], strats, self.nombre_d_evaluations_par_entree, self.strategie.n, self.fonction_evaluation ,self.reglages_min ))    

    def executer_evaluations(self):
        for eval_entree in self.evaluations_entrees :
            eval_entree.evaluer()
    
    def clean(self):
        evaluations_entree = []
        
class EvalEntree : # Petit objet qui va contenir les évaluations d'une seule entrée

    def __init__(self, entree, strategie, nombre_d_evaluations_par_entree, n, fonction_evaluation, reglages_min = None):
        self.entree = entree
        self.n = n
        self.k = len(strategie['probas'])
        self.fonction_evaluation = fonction_evaluation # Quasi systematiquement wrapper_evalue
        self.reglages_min = reglages_min
        self.echantillons_bruts = torch.tensor([]) # De taille NEPE (nombre d'evaluations par entrée, en gros 3 ou 4)
        self.echantillons_raffines = torch.tensor([]) # De taille NEPE
        self.evaluations = torch.tensor([])
        self.nepe = nombre_d_evaluations_par_entree
        self.strategie = strategie
        self.indices_modes = None

    def echantillonner(self):
        # Tirage des indices de mode selon la loi catégorielle
        indices_modes = torch.multinomial(self.strategie['probas'], self.nepe, replacement=True)
        # Tirage dans chaque dimension selon la N(m, sigma)
        tirages = []
        for i in range(self.nepe):
            tirages.append(torch.normal(self.strategie['moyenne'][indices_modes[i]], self.strategie['variance'][indices_modes[i]]))
        echantillons = torch.stack(tirages)
        self.echantillons_bruts = echantillons

    def raffiner(self):
        x, y, petit_rayon = self.entree
        petit_rayon = float(petit_rayon.numpy())
        dims = (x.item(), y.item())
        echantillons_raffines = []
        for e in self.echantillons_bruts:
            echantillons_raffines.append( recentrer_une_entree(e, dims, petit_rayon, self.k))
        self.echantillons_raffines = torch.stack(echantillons_raffines) # de taille NEPE

    def evaluer(self):
        self.echantillonner()
        self.raffiner()
        reg = self.reglages_min
        x, y, petit_rayon = self.entree
        petit_rayon = float(petit_rayon.numpy())
        dims = (x.item(), y.item())
        args = [(elem.clone().detach(), dims, petit_rayon, self.k, reg) for elem in self.echantillons_raffines] # On prépare les arguments sous la bonne forme pour se faire miamer par une fonction adaptée à la parallélisation
        with ProcessPoolExecutor(max_workers=None) as executor:
            recompenses = list(executor.map(self.fonction_evaluation, args))
        recompenses_tensor = torch.tensor(recompenses, dtype=torch.float32) # De taille NEPE
        self.evaluations = recompenses_tensor

class GestionRecompenses : # Gros objet qui va collecter les infos des recompenses de chacune des entréees pour calculer la loss
    def __init__(self, resultats_evaluation):
        self.resultats_evaluation = resultats_evaluation
        self.RUEs = [] # RUE = Récompense Une Entrée, petit objet qui aura les infos importantes pour le calcul de la loss liée à une entrée
        self.loss = None

    def creer_RUEs(self):
        for eval_entree in self.resultats_evaluation.evaluations_entrees:
            self.RUEs.append(RecompenseUneEntree(eval_entree)) # Pour RUE on associe l'objet EvalEntree correspondant

    def calcul_log_probs(self):
        self.creer_RUEs()
        for rue in self.RUEs:
            rue.calculer_log_probs()

    def calcul_loss(self):
        loss = 0
        self.calcul_log_probs()
        for rue in  self.RUEs:
            #rue.calculer_log_probs()
            rue.calculer_recompenses()
            loss -= (rue.recompenses*rue.log_probs).mean()
        self.loss = loss
        
class RecompenseUneEntree : # Petit objet qui aura les infos importantes pour le calcul de la loss liée à une entrée

    def __init__(self, EvalEntree):
        self.eval_entree = EvalEntree
        self.log_probs = None
        self.recompenses = None

    def calculer_recompenses(self):
        recompenses_brute = self.eval_entree.evaluations
        recompenses_raffine = recompenses_brute - torch.mean(recompenses_brute) # On ajoute une baseline 
        self.recompenses = recompenses_raffine

    def calculer_log_probs(self):
        probas = self.eval_entree.strategie['probas']
        moyenne = self.eval_entree.strategie['moyenne']
        variance = self.eval_entree.strategie['variance']
        log_probs = []
        echantillons = self.eval_entree.echantillons_bruts
        for i,e in enumerate(echantillons) :
            log_probs.append(log_prob_mixture_gaussienne(e, probas, moyenne, variance))
        self.log_probs = torch.stack(log_probs)

