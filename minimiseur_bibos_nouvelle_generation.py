import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math
from math import *
import scipy
from scipy import stats
import time
import copy
import random
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from IPython.display import HTML
import os
from matplotlib.patches import Rectangle, Circle
import matplotlib.animation as animation
matplotlib.rcParams['animation.embed_limit'] = 100 * 1024 * 1024  # 100 Mo
from matplotlib.animation import PillowWriter
import importlib
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.animation as animation

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

def inv_chelou(d, d_min ):
  if d >= 1.1 * d_min:
    return 0
  elif d <= d_min :
    return 1
  else:
    return 1/ (1 +( ( d - d_min)/(0.02*d_min)))
  
def inv(d, d_min ):
  if d >= 1.1 * d_min:
    return 0
  elif d <= d_min :
    return 1e8
  else:
    return 1/ (d-d_min)

def quad(x, portee_demi_valeur):
    a = portee_demi_valeur/(1- 1/np.sqrt(2))
    if x <= a :
        return (x-a)**2
    else:
        return 0

def perte_inch_lisse_v1(arrangement, reglages= None):

  # Initialisation des objets
    arrangement['petits centres'] = petits_centres(arrangement)
    n= len(arrangement['petits centres'])
    rayon_mandrin = arrangement['petit rayon']
    def init_bibo():
        bibo =  {
        
               'distance minimale rouleau': float('inf'),
               'deuxieme distance minimale rouleau': float('inf'),
               'point min rouleau' : None,
               'gradient min rouleau' : np.zeros(2),

               'distance minimale mandrin': float('inf'),
               'deuxieme distance minimale mandrin': float('inf'),
               'point min mandrin' : None,
               'gradient min mandrin' : np.zeros(2),

               'autres grads' : [],
               'distances' : [],
               'points' : []
               
               }
        return(bibo)
    
    bibos = [init_bibo() for i in range(n)]

  # Remplissage des infos sur les gradients et les distances
    for i in range(n):
        for j in range(i+1,n) : 
            dist_grad_2_bibos(i,j,arrangement,bibos, reglages)
        dist_grad_bibo_bord(i, arrangement, bibos, reglages)
    
  # Initialisation des objets 'gradient' et "e_thetas"
    gradient = np.zeros(n * 3)
    e_thetas = np.zeros((n,2))
    for i in range(n):
        e_thetas[i] = np.array([-np.sin(arrangement['orientations'][i]), np.cos(arrangement['orientations'][i])])

  # Calcul des poids 
    poids = calcul_poids(bibos, rayon_mandrin, reglages)

  # Calcul du gradient
    gradient = maj_gradient(gradient, bibos, e_thetas, reglages)

  # Retour
    perte_inch_lisse.bibos = bibos
    if hasattr(inch_allah_lisse_bibos, 'hist_bibos'):
        inch_allah_lisse_bibos.hist_bibos.append(bibos)
    return -min(maj_gradient.d_min_rouleau,maj_gradient.d_min_mandrin) , gradient

def dist_grad_2_bibos(i,j,arrangement,bibos, reglages= None):
    distances = transformer_distances_2_bibos(i,j,arrangement)
    directions_gradient = transformer_distances_2_bibos.directions_gradient
    indicatrice_grand_petit = [['g','g'],['g','p'],['p','g'],['p','p']]
    for k, dist in enumerate(distances) :

     # On traite distances/gradients relatifs aux rouleaux
      if indicatrice_grand_petit[k][0] == 'g':

        if dist <= bibos[i]['distance minimale rouleau'] and bibos[i]['distance minimale rouleau'] < float('inf') :

          # On met les anciennes valeurs de min dans le stockage               
            bibos[i]['distances'].append(bibos[i]['distance minimale rouleau'])
            bibos[i]['points'].append(bibos[i]['point min rouleau'])
            bibos[i]['autres grads'].append( bibos[i]['gradient min rouleau'])
            bibos[i]['deuxieme distance minimale rouleau'] = bibos[i]['distance minimale rouleau']

          # On met à jour les nouvelles valeurs de min
            bibos[i]['distance minimale rouleau'] = dist
            bibos[i]['point min rouleau'] = indicatrice_grand_petit[k][0] # 'g' ou 'p'
            bibos[i]['gradient min rouleau'] = directions_gradient[k]

        elif bibos[i]['distance minimale rouleau'] == float('inf') : 

          # On met à jour les nouvelles valeurs de min
            bibos[i]['distance minimale rouleau'] = dist
            bibos[i]['point min rouleau'] = indicatrice_grand_petit[k][0]
            bibos[i]['gradient min rouleau'] = directions_gradient[k]

        else:

          # On met à jour le stockage               
            bibos[i]['distances'].append(dist)
            bibos[i]['points'].append(indicatrice_grand_petit[k][0])
            bibos[i]['autres grads'].append( directions_gradient[k])

          # Si c'est une meilleure deuxième distance minimale, on met cette derniere à jour
            if dist < bibos[i]['deuxieme distance minimale rouleau'] and dist > bibos[i]['distance minimale rouleau'] :
                bibos[i]['deuxieme distance minimale rouleau'] = dist

     # On traite distances/gradients relatifs aux mandrins
      else :
        if dist <= bibos[i]['distance minimale mandrin'] and bibos[i]['distance minimale mandrin'] < float('inf') :

          # On met les anciennes valeurs de min dans le stockage               
            bibos[i]['distances'].append(bibos[i]['distance minimale mandrin'])
            bibos[i]['points'].append(bibos[i]['point min mandrin'])
            bibos[i]['autres grads'].append( bibos[i]['gradient min mandrin'])
            bibos[i]['deuxieme distance minimale mandrin'] = bibos[i]['distance minimale mandrin']

          # On met à jour les nouvelles valeurs de min
            bibos[i]['distance minimale mandrin'] = dist
            bibos[i]['point min mandrin'] = indicatrice_grand_petit[k][0] # 'g' ou 'p'
            bibos[i]['gradient min mandrin'] = directions_gradient[k]

        elif bibos[i]['distance minimale mandrin'] == float('inf') : 

          # On met à jour les nouvelles valeurs de min
            bibos[i]['distance minimale mandrin'] = dist
            bibos[i]['point min mandrin'] = indicatrice_grand_petit[k][0]
            bibos[i]['gradient min mandrin'] = directions_gradient[k]

        else:

          # On met à jour le stockage               
            bibos[i]['distances'].append(dist)
            bibos[i]['points'].append(indicatrice_grand_petit[k][0])
            bibos[i]['autres grads'].append( directions_gradient[k])

          # Si c'est une meilleure deuxième distance minimale, on met cette derniere à jour
            if dist < bibos[i]['deuxieme distance minimale mandrin'] and dist > bibos[i]['distance minimale mandrin'] :
                bibos[i]['deuxieme distance minimale mandrin'] = dist


      # mandrins pour j
      if indicatrice_grand_petit[k][1] == 'p':
        if dist <= bibos[j]['distance minimale mandrin'] and bibos[j]['distance minimale mandrin'] < float('inf') :

          # On met les anciennes valeurs de min dans le stockage               
            bibos[j]['distances'].append(bibos[j]['distance minimale mandrin'])
            bibos[j]['points'].append(bibos[j]['point min mandrin'])
            bibos[j]['autres grads'].append( bibos[j]['gradient min mandrin'])
            bibos[j]['deuxieme distance minimale mandrin'] = bibos[j]['distance minimale mandrin']

          # On met à jour les nouvelles valeurs de min
            bibos[j]['distance minimale mandrin'] = dist
            bibos[j]['point min mandrin'] = indicatrice_grand_petit[k][1] # 'g' ou 'p'
            bibos[j]['gradient min mandrin'] = -directions_gradient[k]

        elif bibos[j]['distance minimale mandrin'] == float('inf') : 

          # On met à jour les nouvelles valeurs de min
            bibos[j]['distance minimale mandrin'] = dist
            bibos[j]['point min mandrin'] = indicatrice_grand_petit[k][1]
            bibos[j]['gradient min mandrin'] = -directions_gradient[k]

        else:

          # On met à jour le stockage               
            bibos[j]['distances'].append(dist)
            bibos[j]['points'].append(indicatrice_grand_petit[k][1])
            bibos[j]['autres grads'].append( -directions_gradient[k])

          # Si c'est une meilleure deuxième distance minimale, on met cette derniere à jour
            if dist < bibos[j]['deuxieme distance minimale mandrin'] and dist > bibos[j]['distance minimale mandrin']  :
                bibos[j]['deuxieme distance minimale mandrin'] = dist

        # Rouleaux pour j
      if indicatrice_grand_petit[k][1] == 'g':
        if dist <= bibos[j]['distance minimale rouleau'] and bibos[j]['distance minimale rouleau'] < float('inf') :

          # On met les anciennes valeurs de min dans le stockage               
            bibos[j]['distances'].append(bibos[j]['distance minimale rouleau'])
            bibos[j]['points'].append(bibos[j]['point min rouleau'])
            bibos[j]['autres grads'].append( bibos[j]['gradient min rouleau'])
            bibos[j]['deuxieme distance minimale rouleau'] = bibos[j]['distance minimale rouleau']

          # On met à jour les nouvelles valeurs de min
            bibos[j]['distance minimale rouleau'] = dist
            bibos[j]['point min rouleau'] = indicatrice_grand_petit[k][1] # 'g' ou 'p'
            bibos[j]['gradient min rouleau'] = -directions_gradient[k]

        elif bibos[j]['distance minimale rouleau'] == float('inf') : 

          # On met à jour les nouvelles valeurs de min
            bibos[j]['distance minimale rouleau'] = dist
            bibos[j]['point min rouleau'] = indicatrice_grand_petit[k][1]
            bibos[j]['gradient min rouleau'] = -directions_gradient[k]

        else:

          # On met à jour le stockage               
            bibos[j]['distances'].append(dist)
            bibos[j]['points'].append(indicatrice_grand_petit[k][1])
            bibos[j]['autres grads'].append( -directions_gradient[k])

          # Si c'est une meilleure deuxième distance minimale, on met cette derniere à jour
            if dist < bibos[j]['deuxieme distance minimale rouleau'] and dist > bibos[j]['distance minimale rouleau']  :
                bibos[j]['deuxieme distance minimale rouleau'] = dist

def dist_grad_bibo_bord(i, arrangement, bibos, reglages= None):
    dims = projeter_sur_plan(arrangement['dimensions du carton'], arrangement['axe'])
    distances = transformer_distances_bibo_bord(i,dims,arrangement)
    directions_gradient = transformer_distances_bibo_bord.directions_gradient
    indicatrice_points = ['g']*4 + ['p']*4
    for k, dist in enumerate(distances) :

     # On traite distances/gradients relatifs aux rouleaux
      if indicatrice_points[k] == 'g' :
        if dist <= bibos[i]['distance minimale rouleau'] and bibos[i]['distance minimale rouleau'] < float('inf') :

          # On met les anciennes valeurs de min dans le stockage               
            bibos[i]['distances'].append(bibos[i]['distance minimale rouleau'])
            bibos[i]['points'].append(bibos[i]['point min rouleau'])
            bibos[i]['autres grads'].append( bibos[i]['gradient min rouleau'])
            bibos[i]['deuxieme distance minimale rouleau'] = bibos[i]['distance minimale rouleau']

          # On met à jour les nouvelles valeurs de min
            bibos[i]['distance minimale rouleau'] = dist
            bibos[i]['point min rouleau'] = indicatrice_points[k] # 'g' ou 'p'
            bibos[i]['gradient min rouleau'] = directions_gradient[k]

        elif bibos[i]['distance minimale rouleau'] == float('inf') : 

          # On met à jour les nouvelles valeurs de min
            bibos[i]['distance minimale rouleau'] = dist
            bibos[i]['point min rouleau'] = indicatrice_points[k]
            bibos[i]['gradient min rouleau'] = directions_gradient[k]

        else:

          # On met à jour le stockage               
            bibos[i]['distances'].append(dist)
            bibos[i]['points'].append(indicatrice_points[k])
            bibos[i]['autres grads'].append(directions_gradient[k])

          # Si c'est une meilleure deuxième distance minimale, on met cette derniere à jour
            if dist < bibos[i]['deuxieme distance minimale rouleau'] :
                bibos[i]['deuxieme distance minimale rouleau'] = dist

     # On traite distances/gradients relatifs aux mandrins
      if indicatrice_points[k] == 'p' :
        if dist <= bibos[i]['distance minimale mandrin'] and bibos[i]['distance minimale mandrin'] < float('inf') :

          # On met les anciennes valeurs de min dans le stockage               
            bibos[i]['distances'].append(bibos[i]['distance minimale mandrin'])
            bibos[i]['points'].append(bibos[i]['point min mandrin'])
            bibos[i]['autres grads'].append( bibos[i]['gradient min mandrin'])
            bibos[i]['deuxieme distance minimale mandrin'] = bibos[i]['distance minimale mandrin']

          # On met à jour les nouvelles valeurs de min
            bibos[i]['distance minimale mandrin'] = dist
            bibos[i]['point min mandrin'] = indicatrice_points[k] # 'g' ou 'p'
            bibos[i]['gradient min mandrin'] = directions_gradient[k]

        elif bibos[i]['distance minimale mandrin'] == float('inf') : 

          # On met à jour les nouvelles valeurs de min
            bibos[i]['distance minimale mandrin'] = dist
            bibos[i]['point min mandrin'] = indicatrice_points[k]
            bibos[i]['gradient min mandrin'] = directions_gradient[k]

        else:

          # On met à jour le stockage               
            bibos[i]['distances'].append(dist)
            bibos[i]['points'].append(indicatrice_points[k])
            bibos[i]['autres grads'].append(directions_gradient[k])

          # Si c'est une meilleure deuxième distance minimale, on met cette derniere à jour
            if dist < bibos[i]['deuxieme distance minimale mandrin'] :
                bibos[i]['deuxieme distance minimale mandrin'] = dist

def calcul_poids(bibos,rayon_mandrin, reglages= None ):

  # Gestion des réglages 
    reglages_par_defaut = {
        'fonction perte': quad,
        'portee clip' : float("inf"),
        'portee demi valeur' : 2 }
    
    if reglages is None :
        reglages = reglages_par_defaut
    else :
        reglages = reglages_par_defaut | reglages

  #

    fonction = reglages['fonction perte']
    portee_clip = reglages['portee clip']
    portee_demi_valeur = reglages['portee demi valeur']
  
    for bibo_i in bibos:
        poids = []
        dist_min = min(bibo_i['distance minimale rouleau'], bibo_i['distance minimale mandrin'] )
        dist_0 = max(dist_min, rayon_mandrin)
        for j, dist in enumerate(bibo_i['distances']):
            dist_ajustee = (dist - dist_min)/ (abs(dist_0)+ 1e-8)
            poids.append(fonction(dist_ajustee, portee_demi_valeur))
        poids = np.array(poids)
        poids = poids/(np.sum(poids)+1e-8)
        bibo_i['poids'] = poids
        
<<<<<<< HEAD
def maj_gradient(gradient, bibos, e_thetas, reglages):
=======
"""def maj_gradient(gradient, bibos, e_thetas, reglages):
>>>>>>> e1d9b9f (Premier commit : ajout des fichiers Python)
    n = len(gradient)//3
    d_min_rouleau = float('inf')
    d_min_mandrin = float('inf')
    for i, bibo_i in enumerate(bibos):

      # Calcul du coeff qui ponderera ce morceau du gradient (pour le rouleau)
        coeff_rouleau = np.abs(bibo_i['distance minimale rouleau'] - bibo_i['deuxieme distance minimale rouleau'])
        d_min_rouleau = min(bibo_i['distance minimale rouleau'], d_min_rouleau)

      # Calcul du coeff qui ponderera ce morceau du gradient (pour le mandrin)
        coeff_mandrin = np.abs(bibo_i['distance minimale mandrin'] - bibo_i['deuxieme distance minimale mandrin'])
        d_min_mandrin = min(bibo_i['distance minimale mandrin'], d_min_mandrin)
        if coeff_mandrin == float('inf'):
            coeff_mandrin = 0

      # Calcul de la contribution des directions secondaires
        poids = bibo_i['poids']
        for indice, point in enumerate(bibo_i['points']):
            vecteur = bibo_i['autres grads'][indice]
            if point == 'g':
                gradient[2*i] += poids[indice]*vecteur[0]*0.5*coeff_rouleau
                gradient[2*i+1] += poids[indice]*vecteur[1]*0.5*coeff_rouleau
            else:
                e_theta = e_thetas[i]
                #print( 'gradient : ', gradient)
                # Debug
                if poids[indice] * np.dot(vecteur, e_theta)*0.5*coeff_mandrin == float('inf'):
                  print('erreur inf')
                  print('poids : ',  poids[indice])
                  print('vecteur : ', vecteur)
                  print('e_theta : ', e_theta )
                  print('coeff mandrin : ', coeff_mandrin)
                  print('coeff rouleau : ', coeff_rouleau)
                  print('indice bibo : ', i)
                  print('indice point : ', indice )
                gradient[2*n+i] += poids[indice] * np.dot(vecteur, e_theta)*0.5*coeff_mandrin

      # Calcul de la contribution de la composante principale (rouleau)
        direction_pple_rouleau = bibo_i['gradient min rouleau']
        direction_pple_rouleau = direction_pple_rouleau /(np.linalg.norm(direction_pple_rouleau)+1e-10)
        gradient[2*i] += direction_pple_rouleau[0]*0.5*coeff_rouleau
        gradient[2*i+1] += direction_pple_rouleau[1]*0.5*coeff_rouleau
      
      # Calcul de la contribution de la composante principale (mandrin)
        direction_pple_mandrin = bibo_i['gradient min mandrin']
        direction_pple_mandrin = direction_pple_mandrin /(np.linalg.norm(direction_pple_mandrin)+1e-10)
        e_theta = e_thetas[i]
        gradient[2*n+i] += np.dot(direction_pple_mandrin, e_theta)*0.5*coeff_mandrin

      # Cas spéciaux : pour libérer le mandrin, on doit faire bouger le rouleau

       # Cas special 1 : Le mandrin est coincé dans un coin
        if bibo_i['distance minimale mandrin'] < bibo_i['distance minimale rouleau'] and bibo_i['deuxieme distance minimale mandrin'] < bibo_i['distance minimale rouleau']:
            bibo_i['cas special'] = 'cas special 1 : le mandrin est coincé dans un coin'
            pond = 1/ (1+np.abs(bibo_i['distance minimale mandrin'] - bibo_i['deuxieme distance minimale mandrin']))
            a = 1/ (1+ pond)
            direction_pple_mandrin_1 = direction_pple_mandrin /(np.linalg.norm(direction_pple_mandrin)+1e-10)
            coeff_rouleau_prime =  bibo_i['distance minimale rouleau'] - bibo_i['distance minimale mandrin']
            i_min = np.argmin(bibo_i['distances'])
            if bibo_i['distances'][i_min] != bibo_i['deuxieme distance minimale mandrin']:
               print('bleubleubleu')
            deuxieme_direction_pple_mandrin = bibo_i['autres grads'][i_min]
            direction_pple_mandrin_2 = deuxieme_direction_pple_mandrin /(np.linalg.norm(deuxieme_direction_pple_mandrin)+1e-10)
            direction_rouleau = a * (direction_pple_mandrin_1 + pond *direction_pple_mandrin_2 )
            gradient[2*i] = direction_rouleau[0]*0.5*coeff_rouleau_prime
            gradient[2*i+1] = direction_rouleau[1]*0.5*coeff_rouleau_prime

       # Cas spécial 2 : Le mandrin est coincé contre un bord
        elif bibo_i['distance minimale mandrin'] < bibo_i['distance minimale rouleau']:
            bibo_i['cas special'] = 'cas special 1 : le mandrin est coincé dans un coin'
            pond = 1/ (1+np.abs(bibo_i['distance minimale mandrin'] - bibo_i['distance minimale rouleau']))
            a = 1/ (1+ pond)
            direction_pple_rouleau = direction_pple_rouleau /(np.linalg.norm(direction_pple_rouleau)+1e-10)
            direction_pple_mandrin = direction_pple_mandrin /(np.linalg.norm(direction_pple_mandrin)+1e-10)
            direction_rouleau = a * (direction_pple_mandrin + pond *direction_pple_rouleau )
            gradient[2*i] = direction_rouleau[0]*0.5*coeff_rouleau
            gradient[2*i+1] = direction_rouleau[1]*0.5*coeff_rouleau

    maj_gradient.d_min_rouleau = d_min_rouleau
    maj_gradient.d_min_mandrin = d_min_mandrin
<<<<<<< HEAD
    return(gradient)
=======
    return(gradient)"""
>>>>>>> e1d9b9f (Premier commit : ajout des fichiers Python)

class ajustement_efficacite :
  def __init__(self, n, cible_efficacite, augmentation_pas, diminution_pas, activation_haute = None, activation_basse = None):
    self.efficacites_deplacements = np.zeros(3*n)
    self.ponderations_deplacements = np.ones(3*n)
    self.deplacements_effectifs = np.zeros(3*n)
    self.distances_parcourues = np.zeros(3*n)
    self.n = n
    self.cible_efficacite = cible_efficacite
    self.coeff_descente_log_haut = log(augmentation_pas)/(cible_efficacite - 1)
    self.coeff_descente_log_bas = log(diminution_pas)/(cible_efficacite)
    if activation_haute == None:
      activation_haute = 0.5*(1 + cible_efficacite)
    if activation_basse is None :
      activation_basse = 0.5 * cible_efficacite
    self.activation_haute = activation_haute
    self.activation_basse = activation_basse

  def maj_efficacite(self,histo):
    deb = histo[0]
    fin = histo[-1]
    n= self.n

   # On calcule le vrai déplacement depuis quelques itérations
    deplacements = fin-deb
    for i in range(len(deplacements)//3):
      deplacements[2*n+i]+=(pi)
      deplacements[2*n+i]%=(2*pi)
      deplacements[2*n+i]-=(pi)
    self.deplacements_effectifs = deplacements

   # On calcule la distance parcourue totale
    distances_brutes = np.abs(np.diff(np.array(histo), axis = 0) )
    for j in range(len(distances_brutes)):
      for i in range(n):
        distances_brutes[j][2*n+i]%=(2*pi)
    distances_parcourues= np.sum( distances_brutes , axis = 0)+1e-8

   # On peut donc calculer maintenant l'efficacité des déplacements
    self.efficacites_deplacements = np.abs(deplacements)/ distances_parcourues

  def maj_ponderations(self):
    n= self.n
    for i in range(3*n):
      if self.efficacites_deplacements[i] > self.activation_haute:
        self.ponderations_deplacements[i] *= exp(self.coeff_descente_log_haut * (-self.efficacites_deplacements[i] + self.cible_efficacite))
      elif self.efficacites_deplacements[i] < self.activation_basse:
        self.ponderations_deplacements[i] *= exp(self.coeff_descente_log_bas * (-self.efficacites_deplacements[i] + self.cible_efficacite))

def descente_gradient(fonction, x_0, reglages_descente = None): 

  # gestion des reglages
    reglages = reglages_descente
    reglages_par_defaut = {'eps' : 1e-4, 
                      'pas_initial':1e-2, 'facteur_reduction':0.75, 'facteur_augmentation':1.2,
                      'seuil_stabilite':10, 'pas_max':0.2, 'max_iter':500,
                      'pas_minimal':0.2*1e-1, 'nb_stagnation_max':1, 'stag_moyenne':10,'seuil_stab_moyennes': 0.25, 'inertie' : 0, 'max_deplacement' : 0.05 }
    if reglages is None : 
        reglages = reglages_par_defaut
    else:
        reglages = reglages_par_defaut | reglages

    eps = reglages['eps']
    pas_initial = reglages['pas_initial']
    facteur_reduction = reglages['facteur_reduction']
    facteur_augmentation = reglages['facteur_augmentation']
    seuil_stabilite = reglages['seuil_stabilite']
    pas_max = reglages['pas_max']
    pas_minimal = reglages['pas_minimal']
    max_iter = reglages['max_iter']
    nb_stagnation_max = reglages['nb_stagnation_max']
    stag_moyenne = reglages['stag_moyenne']
    seuil_stab_moyennes = reglages['seuil_stab_moyennes']
    inertie = reglages['inertie']
    max_deplacement = reglages['max_deplacement']

  # Initialisation des variables 
    n= len(x_0)//3
    pas_max *= 2*n
    pas_minimal *= 2*n
    delta_grad = 0
    m = np.zeros(3*n)
    m_prec = np.zeros(3*n)
    x = x_0.copy()
    pas = pas_initial
    grad_prec = None
    stagnation = 0
    deplacement_precedent = np.zeros(3*n)
    perte_precedente = float('inf')
    histo_x = []
    cpt_moy = 0
    gestion_efficacite = ajustement_efficacite(n, seuil_stab_moyennes, facteur_augmentation, facteur_reduction)
    cpt_petit_pas = 0
    deplacement = np.zeros(3*n)

  # Definition des fonctions support
    def maj_historique():
        if hasattr(sauvegarde,'historique'):

            sauvegarde.historique.append({
            'tour': it,
            'centres_orientations': ancien_x,
            'perte': perte_precedente,
            'delta grad' : delta_grad,
            'deplacement' : deplacement,
            "pas" : np.linalg.norm(deplacement),
            'ponderations' : copy.deepcopy(gestion_efficacite.ponderations_deplacements),
            'pas max' : pas_max,
            'pas min' : pas_minimal,
            'moyenne positions' : m,
            'max deplacement autorise': max_deplacement,
            'plus grande efficacite de deplacement' : np.max(gestion_efficacite.efficacites_deplacements),
            'efficacite des deplacements' : copy.deepcopy(gestion_efficacite.efficacites_deplacements),
            'max deplacement' : np.max(np.abs(deplacement[0:2*n]))   
            
        })

    def clip_deplacement(deplacement, stagnation) :
        stag = True
        deplacement_clipe = np.zeros(3*n)
        for i in range(n):

            dep_rouleau = deplacement[2*i:2*i+2]
            norme_dep_rouleau = np.linalg.norm(dep_rouleau)
            if norme_dep_rouleau < pas_minimal/(2*n):
                deplacement_clipe[2*i:2*i+2] = dep_rouleau * pas_minimal/ (norme_dep_rouleau+1e-8)
            else :
                deplacement_clipe[2*i:2*i+2] = dep_rouleau
                stag = False

            dep_mandrin = deplacement[2*n+i]
            norme_dep_mandrin = np.linalg.norm(dep_mandrin)
            if norme_dep_mandrin < pas_minimal/(2*n):
                deplacement_clipe[2*n+i] = 2*pi*dep_mandrin * pas_minimal / (norme_dep_mandrin+1e-8) 
            else :
                stag = False
                deplacement_clipe[2*n+i] = 2*pi*dep_mandrin

        if stag :
            stagnation +=1
        else :
            stagnation = 0 

        return deplacement_clipe, stagnation

    def moyennes_recentes(histo_x, x) :
        if len(histo_x) > stag_moyenne :
            histo_x.pop(0)
        histo_x.append(x)
        if len(histo_x) >= stag_moyenne:
            return np.mean(histo_x, axis = 0)
        else :
            return np.zeros(3*n)

  # Boucle Principale
    for it in range(max_iter):
       # Calcul de pert, gradient
<<<<<<< HEAD
        perte, grad = fonction(x)
=======
        perte, grad = fonction(x, reglages)
>>>>>>> e1d9b9f (Premier commit : ajout des fichiers Python)
        norme = np.linalg.norm(grad)

       # Conditions d'arret
        u= perte_precedente - perte


        if norme < eps and  u < eps:
            break
        perte_precedente = perte

        direction = grad

       # Augmentation du pas si le gradient est stable
        if grad_prec is not None:
            delta_grad = np.linalg.norm(grad/(np.linalg.norm(grad)+1e-8) - grad_prec/(np.linalg.norm(grad_prec)+1e-8))

       # Clip des déplacements (pour pas dépasser pas_max ou pas_min)
        deplacement, stagnation  =  clip_deplacement(pas * direction, stagnation)
        deplacement /= 2*n

       # Inertie
        deplacement  = (1-inertie)*deplacement + inertie * deplacement_precedent
        deplacement_precedent = deplacement

       # Pondération des déplacements suivant leur efficacité (on vise une efficacité pas trop grande (ce qui voudrait dire qu'on pourrait aller plus vite))
        m = moyennes_recentes(histo_x, x)
        '''u = x - histo_x[0] 
        for i in range(len(u)//3):
          u[2*n+i]+=(pi)
          u[2*n+i]%=(2*pi)
          u[2*n+i]-=(pi) 
        deplacement_effectif_recent_coordonnees = u
        d = np.abs(np.diff(np.array(histo_x), axis = 0) )
        for j in range(len(d)):
          for i in range(n//3):
            d[j][2*n+i]%=(2*pi)
        distance_parcourue_recente_coordonnees = np.sum( d , axis = 0)
        efficacite_deplacement_recent_coordonnees = np.abs(deplacement_effectif_recent_coordonnees )/ distance_parcourue_recente_coordonnees
        deplacement_coordonnee_le_plus_efficace = np.max(np.abs(efficacite_deplacement_recent_coordonnees))

        if it >= stag_moyenne and deplacement_coordonnee_le_plus_efficace <= seuil_stab_moyennes :
            # Si en moyenne on a stagné par rapport à la taille du pas (ie on est souvent revenu sur nos pas) 
            cpt_moy += 1
        else :
            cpt_moy = 0
        m_prec = m

        if cpt_moy >= nb_stagnation_max:
            cpt_moy = 0
            pas_minimal *= facteur_reduction
            pas_max*= facteur_reduction 
            pas = max(pas * facteur_reduction, pas_minimal)
        else : 
            pas_minimal *= facteur_augmentation
            pas_max*= facteur_augmentation 
            pas = max(pas * facteur_augmentation, pas_minimal) '''

        if it >= stag_moyenne :
          gestion_efficacite.maj_efficacite(histo_x)
          gestion_efficacite.maj_ponderations()
          deplacement *= gestion_efficacite.ponderations_deplacements

       # Clip pour garder des valeurs raisonnables
        for i in range(2*n):
            deplacement[i]=np.clip((deplacement[i]), a_min = -max_deplacement, a_max = max_deplacement)

       # On adapte le pas maximal en fonction de l'observé actuel (sinon il prend un peu la confiance et risque de réexploser rapidement )
        if max_deplacement < np.max(np.abs(deplacement[0:2*n])) :
            max_deplacement *=1.1
        else : 
          max_deplacement = np.max(np.abs(deplacement[0:2*n]))*1.2
        
       # Descente (montée en toute rigueur vu que mon gradient est à l'envers)
        ancien_x = x.copy()
        x = x + deplacement
        for i in range(n):
            x[2*n+i]%= 2*pi
        grad_prec = grad

        maj_historique()
        
  # Retour
    descente_gradient.nombre_iterations = it
    return x

def frames_descente(hist= None):
    if not hasattr(sauvegarde, 'historique') or not hasattr(sauvegarde, 'infos_arrangement') and hist is None:
        print('Aucune sauvegarde disponible')
        return []
    elif hist is not None :
      historique = sauvegarde.historique
      infos_arrg = sauvegarde.infos_arrangement
    else :
      historique = sauvegarde.historique
      infos_arrg = sauvegarde.infos_arrangement

    frames = []
    axe = np.array([0, 0, 1])
    x, y = infos_arrg['dimensions'].copy()
    dims = np.array([x, y, 1])
    pr = copy.copy(infos_arrg['petit rayon'])
    
    for h in historique:
        co = copy.copy(h['centres_orientations'])
        c,o = t_2_d_bibos(injection_bibos(co, axe, 0.5))
        it = copy.copy(h['tour'])
                
        frame = {
            'arrangement' : {
            'grand rayon': -h['perte'],
            'petit rayon': pr,
            'orientations': o,
            'dimensions du carton': dims,
            'centres': c,
            'axe': axe },
            'iteration': it,
            'deplacement': h['deplacement']*10/(np.linalg.norm(h['deplacement'])+1e-8),
            'longueur pas': np.linalg.norm(h['deplacement']),
            'perte' : h['perte']
        }
        
        frames.append(frame)
    
    return frames
  
def film_descente_classique(nom):
    import numpy as np
    import matplotlib.pyplot as plt
    from matplotlib.patches import Circle, Rectangle
    from io import BytesIO
    from PIL import Image

    frames = frames_descente()
    images = []

    def projeter_sur_plan(point, axe):
        axes_du_plan = []
        for e in np.eye(3):
            if not np.array_equal(e, axe):
                axes_du_plan.append(e)
        projete = np.zeros(2)
        projete[0] = np.dot(point, axes_du_plan[0])
        projete[1] = np.dot(point, axes_du_plan[1])
        return projete

    def ajouter_carton(ax, dimensions, axe, couleur='green', alpha=0.3):
        dimensions_2d = projeter_sur_plan(dimensions, axe)
        rect = Rectangle((0, 0), dimensions_2d[0], dimensions_2d[1], color=couleur, alpha=alpha)
        ax.add_patch(rect)

    def ajouter_disques(ax, centres, rayon, axe, couleur, alpha):
        for centre in centres:
            centre_2d = projeter_sur_plan(centre, axe)
            circ = Circle(centre_2d, rayon, color=couleur, alpha=alpha)
            ax.add_patch(circ)

    for i, frame in enumerate(frames):
        arrangement = frame['arrangement']
        deplacement = frame['deplacement']
        perte = frame['perte']
        longueur_pas = frame['longueur pas']
        iteration = frame['iteration']

        dimensions_carton = arrangement['dimensions du carton']
        axe = arrangement['axe']
        centres_grands = np.array(arrangement['centres'])
        orientations = np.array(arrangement['orientations'])
        rayon_grand = arrangement['grand rayon']
        rayon_petit = arrangement['petit rayon']
        centres_petits = petits_centres(arrangement)
        n = len(centres_grands)

        centres_grands_2d = [projeter_sur_plan(c, axe) for c in centres_grands]
        centres_petits_2d = [projeter_sur_plan(c, axe) for c in centres_petits]

        fig, ax = plt.subplots(figsize=(6, 4), dpi=80)
        ax.set_aspect('equal')

        if axe[0] == 1:
            ax.set_xlim(0, dimensions_carton[1])
            ax.set_ylim(0, dimensions_carton[2])
        elif axe[1] == 1:
            ax.set_xlim(0, dimensions_carton[0])
            ax.set_ylim(0, dimensions_carton[2])
        else:
            ax.set_xlim(0, dimensions_carton[0])
            ax.set_ylim(0, dimensions_carton[1])

        ajouter_carton(ax, dimensions_carton, axe)
        ajouter_disques(ax, centres_grands, rayon_grand, axe, couleur='blue', alpha=0.6)
        ajouter_disques(ax, centres_petits, rayon_petit, axe, couleur='orange', alpha=0.6)

        for j in range(n):
            x, y = centres_grands_2d[j]
            dx = deplacement[2 * j] *0.2
            dy = deplacement[2 * j + 1]*0.2
            ax.arrow(x, y, dx, dy,
                     head_width=0.2 * rayon_grand, head_length=0.3 * rayon_grand,
                     fc='red', ec='red', length_includes_head=True)
            ax.text(x, y, str(j), fontsize=9, ha='center', va='center', color='white', weight='bold')

        for j in range(n):
            petit_centre_j = centres_petits[j]
            theta_j = orientations[j]
            e_theta = -np.sin(theta_j) * np.array([1, 0]) + np.cos(theta_j) * np.array([0, 1])
            base_2d = projeter_sur_plan(petit_centre_j, axe)
            dir_2d = e_theta
            intensite = deplacement[2 * n + j]*0.2
            if intensite != 0:
                dx = intensite * dir_2d[0]
                dy = intensite * dir_2d[1]
                ax.arrow(base_2d[0], base_2d[1], dx, dy,
                         head_width=0.2 * rayon_petit, head_length=0.3 * rayon_petit,
                         fc='purple', ec='purple', length_includes_head=True)
                
        ax.text(0.01, 0.99, f"Itération : {iteration}", ha='left', va='top',transform=ax.transAxes, fontsize=10)
        ax.text(0.01, 0.95, f"Longueur du pas : {longueur_pas}", ha='left', va='top',transform=ax.transAxes, fontsize=10)

        plt.title(f"{nom} — frame {i} — perte = {perte:.4f}")
        plt.tight_layout()

        # Sauvegarde en mémoire
        buf = BytesIO()
        plt.savefig(buf, format='png')
        buf.seek(0)
        images.append(Image.open(buf).convert('RGB'))
        plt.close(fig)

    # Création du GIF
    images[0].save(f"{nom}.gif",
                   save_all=True,
                   append_images=images[1:],
                   duration=200,  # ms par frame
                   loop=0)

def inch_allah_lisse_bibos(arrangement, reglages=None, histo_bibos = False):
    if histo_bibos:
      inch_allah_lisse_bibos.hist_bibos = []
    # Réglages par défaut
    reglages_par_defaut = {
        'fonction pertes inch': perte_inch_lisse_v2,
        'epsilon': 1,
        'reglages_descente': None
    }
    if reglages is not None:
        reglages = reglages_par_defaut | reglages
    else:
        reglages = reglages_par_defaut
    
    if hasattr(sauvegarde,'infos_arrangement'):
      sauvegarde.infos_arrangement = {'petit rayon' : arrangement['petit rayon'],
         'axe' : arrangement['axe'],
         'valeur' : 0.5,
         'dimensions' : projeter_sur_plan(arrangement['dimensions du carton'], arrangement['axe'])}

    perte_inch = reglages['fonction pertes inch']
    valeurs_intermediaires = []

    # Copie pour ne pas modifier l'arrangement d'origine
    arrangement_modifiable = copy.deepcopy(arrangement)
    centres_3d = np.array(arrangement_modifiable['centres'])
    orientations = np.array(arrangement_modifiable['orientations'], dtype=float)
    axe = np.array(arrangement_modifiable['axe'], dtype=float)
    rayon = arrangement_modifiable['grand rayon']
    petit_rayon = arrangement_modifiable['petit rayon']

    # Projection initiale en 2D
    point_sert_a_rien = np.dot(axe, centres_3d[0])
    x0 = projection_bibos(t_1_d_bibos(centres_3d, orientations), axe)

    # Fonction à minimiser
<<<<<<< HEAD
    def a_minimiser(x_2d):
        centres_projetes,orientations_projetees = t_2_d_bibos(injection_bibos(x_2d, axe, point_sert_a_rien))
        arrangement_modifiable['centres'] = centres_projetes
        arrangement_modifiable['orientations'] = orientations_projetees
        perte, gradient = perte_inch(arrangement_modifiable)
=======
    def a_minimiser(x_2d, reglages):
        centres_projetes,orientations_projetees = t_2_d_bibos(injection_bibos(x_2d, axe, point_sert_a_rien))
        arrangement_modifiable['centres'] = centres_projetes
        arrangement_modifiable['orientations'] = orientations_projetees
        perte, gradient = perte_inch(arrangement_modifiable, reglages)
>>>>>>> e1d9b9f (Premier commit : ajout des fichiers Python)
        return perte, gradient

    # Optimisation
    dimensions_proj = projeter_sur_plan(arrangement_modifiable['dimensions du carton'], axe)
    res = descente_gradient( a_minimiser,x0, reglages['reglages_descente'])

    # Reconstruction finale
    x_final = res
    centres_3d, orientations = t_2_d_bibos(injection_bibos(x_final, axe, point_sert_a_rien))
    arrangement_modifiable['centres'] = centres_3d
    arrangement_modifiable['orientations'] = orientations

    # Ajustement du rayon (si requis)
    arrangement_final = ajuster_rayon_bibos(arrangement_modifiable, reglages)
    valeurs_intermediaires.append({
        'centres': arrangement_final['centres'],
        'orientations': arrangement_final['orientations'],
        'rayon': arrangement_final['grand rayon'],
        'code couleur': 1,
        'texte': 'rayon ajusté',
        'perte': 0,
        'iteration du minimiseur': 0
    })

    inch_allah_bibos.valeurs_intermediaires = valeurs_intermediaires
    return arrangement_final

def perte_inch_lisse_v2(arrangement, reglages= None):

  # Initialisation des objets
    arrangement['petits centres'] = petits_centres(arrangement)
    n= len(arrangement['petits centres'])
    rayon_mandrin = arrangement['petit rayon']
    def init_bibo():
        bibo =  {
        
               'distance minimale rouleau': float('inf'),
               'deuxieme distance minimale rouleau': float('inf'),
               'point min rouleau' : None,
               'gradient min rouleau' : np.zeros(2),

               'distance minimale mandrin': float('inf'),
               'deuxieme distance minimale mandrin': float('inf'),
               'point min mandrin' : None,
               'gradient min mandrin' : np.zeros(2),

               'autres grads' : [],
               'distances' : [],
               'points' : []
               
               }
        return(bibo)
    
    bibos = [init_bibo() for i in range(n)]

  # Remplissage des infos sur les gradients et les distances
    for i in range(n):
        for j in range(i+1,n) : 
            dist_grad_2_bibos(i,j,arrangement,bibos, reglages)
        dist_grad_bibo_bord(i, arrangement, bibos, reglages)
    
  # Initialisation des objets 'gradient' et "e_thetas"
    gradient = np.zeros(n * 3)
    e_thetas = np.zeros((n,2))
    for i in range(n):
        e_thetas[i] = np.array([-np.sin(arrangement['orientations'][i]), np.cos(arrangement['orientations'][i])])

  # Calcul des poids 
    poids = calcul_poids_v2(bibos, rayon_mandrin, reglages)

  # Calcul du gradient
    gradient = maj_gradient_v2(gradient, bibos, e_thetas, reglages)

  # Retour
    perte_inch_lisse_v2.bibos = bibos
    if hasattr(inch_allah_lisse_bibos, 'hist_bibos'):
        inch_allah_lisse_bibos.hist_bibos.append(bibos)
<<<<<<< HEAD
    return -min(maj_gradient.d_min_rouleau,maj_gradient.d_min_mandrin) , gradient
=======
    return -min(maj_gradient_v2.d_min_rouleau,maj_gradient_v2.d_min_mandrin) , gradient
>>>>>>> e1d9b9f (Premier commit : ajout des fichiers Python)
  
def calcul_poids_v2(bibos,rayon_mandrin, reglages= None ):

  # Gestion des réglages 
    reglages_par_defaut = {
        'fonction perte': inv,
        'portee clip' : float("inf"),
        'portee demi valeur' : 2 }
    
    if reglages is None :
        reglages = reglages_par_defaut
    else :
        reglages = reglages_par_defaut | reglages

  #

    fonction = reglages['fonction perte']
    portee_clip = reglages['portee clip']
    portee_demi_valeur = reglages['portee demi valeur']
  
    for bibo_i in bibos:
        poids = []
        dist_min = min(bibo_i['distance minimale rouleau'], bibo_i['distance minimale mandrin'] )
        for j, dist in enumerate(bibo_i['distances']):
            poids.append(fonction(dist, dist_min))
        
        poids = np.array(poids)
        bibo_i['somme poids'] = np.sum(poids)+1e-8
        poids = poids/(np.sum(poids)+1e-8)
        bibo_i['poids'] = poids

def maj_gradient_v2(gradient, bibos, e_thetas, reglages):
    n = len(gradient)//3
    d_min_rouleau = float('inf')
    d_min_mandrin = float('inf')

    for i, bibo_i in enumerate(bibos):

      # Mise à jour de d_min_rouleau et d_min_mandrin
        if bibo_i['distance minimale rouleau'] < d_min_rouleau:
            d_min_rouleau = bibo_i['distance minimale rouleau']
        if bibo_i['distance minimale mandrin'] < d_min_mandrin:
            d_min_mandrin = bibo_i['distance minimale mandrin']

      # Calcul de la contribution des directions secondaires
        poids = bibo_i['poids']
        direction_secondaire_i_rouleau = np.zeros(2)
        direction_secondaire_i_mandrin = np.zeros(1)
        for indice, point in enumerate(bibo_i['points']):
            vecteur = bibo_i['autres grads'][indice]
            if point == 'g':
                direction_secondaire_i_rouleau += poids[indice]*vecteur
                
            else:
                e_theta = e_thetas[i]
                direction_secondaire_i_mandrin += poids[indice] * np.dot(vecteur, e_theta)

      # Calcul de la contribution de la composante principale (rouleau)
        direction_pple_rouleau = bibo_i['gradient min rouleau']
        direction_pple_rouleau = direction_pple_rouleau /(np.linalg.norm(direction_pple_rouleau)+1e-10)

      # Calcul de la contribution de la composante principale (mandrin)
        direction_pple_mandrin = bibo_i['gradient min mandrin']
        direction_pple_mandrin = direction_pple_mandrin /(np.linalg.norm(direction_pple_mandrin)+1e-10)

      # Pour les rouleaux :
       # Cas 1 : La contribution principale s'oppose aux autres contributions 
        if np.dot(direction_pple_rouleau, direction_secondaire_i_rouleau) < 0:
          # On ne donne pas plus de la moitié du poids à la direction secondaire
          diff_longueur =  -  (bibo_i['distance minimale rouleau'] - bibo_i['deuxieme distance minimale rouleau'])
          d_0 = 0.11* bibo_i['distance minimale rouleau']
          poids_principal = 1/(1-abs(diff_longueur)/d_0)
          poids_secondaire = 1/(1+abs(diff_longueur)/d_0)
          norm = poids_principal + poids_secondaire 
          direction =(1/norm)*(direction_pple_rouleau * poids_principal + direction_secondaire_i_rouleau * poids_secondaire)
          direction /= (np.linalg.norm(direction)+1e-8)

         # Pour calculer la longueur on s'interesse au bibo le plus proche tel qu'il pousse le bibo dans une direction opposee à celle vers laquelle on se dirige
          dist_min = float('inf')
          for j, dist in enumerate(bibo_i['distances']):
            if dist < dist_min and np.dot(direction_pple_rouleau, bibo_i['autres grads'][j]) < 0:
              dist_min = dist
          longueur = dist_min - bibo_i['distance minimale rouleau']
          gradient[2*i] =  direction[0]*longueur
          gradient[2*i+1] =  direction[1]*longueur

          
          bibo_i['cas normal'] = 'cas normal 1 : la contribution principale s\'oppose aux autres contributions'

        # Cas spécial 1bis : Le rouleau est coincé entre deux "murs" :
          if diff_longueur < 0.05*bibo_i['distance minimale rouleau'] and np.dot(direction_pple_rouleau, direction_secondaire_i_rouleau) < -0.9 :
           # On va regarder ce qu'il se passe dans les directions orthogonales
            vecteur_presque_orthogonal_pos = np.random.rand(len(direction_pple_rouleau))
            vecteur_presque_orthogonal_pos /= np.linalg.norm(vecteur_presque_orthogonal_pos)
            vecteur_presque_orthogonal_pos -= np.dot(vecteur_presque_orthogonal_pos, direction_pple_rouleau) * direction_pple_rouleau
            vecteur_presque_orthogonal_pos /= np.linalg.norm(vecteur_presque_orthogonal_pos)
            vecteur_presque_orthogonal_neg = vecteur_presque_orthogonal_pos * (-1)

            dist_min_pos = float('inf')
            for j, dist in enumerate(bibo_i['distances']):
              if dist < dist_min_pos and np.dot(vecteur_presque_orthogonal_pos, bibo_i['autres grads'][j]) < 0:
                dist_min_pos = dist

            dist_min_neg = float('inf')
            for j, dist in enumerate(bibo_i['distances']):
              if dist < dist_min_neg and np.dot(vecteur_presque_orthogonal_neg, bibo_i['autres grads'][j]) < 0:
                dist_min_neg = dist
            
            if dist_min_neg < dist_min_pos:
              direction = vecteur_presque_orthogonal_pos
              longueur = dist_min_pos - bibo_i['distance minimale rouleau']
              gradient[2*i] =  direction[0]*longueur
              gradient[2*i+1] =  direction[1]*longueur
            else:
              direction = vecteur_presque_orthogonal_neg
              longueur = dist_min_neg - bibo_i['distance minimale rouleau']
              gradient[2*i] =  direction[0]*longueur
              gradient[2*i+1] =  direction[1]*longueur

            bibo_i['cas special'] = 'cas special 1bis : les deux contributions principales s\'annulent quasi parfaitement'


       # Cas 2 : La contribution principale est dans le même sens que les autres contributions
        else:
          somme_poids = bibo_i['somme poids']
          nouvelle_somme_poids = somme_poids+1
          poids_ppl = 1/nouvelle_somme_poids
          poids_sec = (somme_poids/nouvelle_somme_poids)
         # On se dirige dans la direction "moyennée"
          direction = direction_pple_rouleau * poids_ppl + direction_secondaire_i_rouleau * poids_sec
          direction /= np.linalg.norm(direction)

         # Pour calculer la longueur on s'interesse au bibo le plus proche tel qu'il pousse le bibo dans une direction opposee à celle vers laquelle on se dirige
          dist_min = float('inf')
          for j, dist in enumerate(bibo_i['distances']):
            if dist < dist_min  and np.dot(direction_pple_rouleau, bibo_i['autres grads'][j]) < 0 :
              dist_min = dist

          bibo_i['cas normal'] = 'cas normal 2 : la contribution principale est dans le même sens que les autres contributions'
              
         # On peut enfin calculer la longueur et mettre à jour le gradient
          longueur = dist_min - bibo_i['distance minimale rouleau']
          direction *= longueur
          gradient[2*i] =  direction[0]
          gradient[2*i+1] =  direction[1]

      # Pour les mandrins  :
       # Toujours cas 1 : La contribution principale s'oppose aux autres contributions 
        # On se dirige dans le sens de la direction principale, d'autant plus qu'on a de la place
        longueur =   -(bibo_i['distance minimale mandrin'] - bibo_i['deuxieme distance minimale mandrin'])
        direction = direction_pple_mandrin * longueur
        gradient[2*n+i] =  np.dot(direction, e_thetas[i])

      # Cas spéciaux : pour libérer le mandrin, on doit faire bouger le rouleau

<<<<<<< HEAD
       # Cas special 1 : Le mandrin est coincé dans un "coin"
        if bibo_i['distance minimale mandrin'] < bibo_i['distance minimale rouleau'] and bibo_i['deuxieme distance minimale mandrin'] < bibo_i['distance minimale rouleau']:
=======
       # Cas special 1.5 : Pour libérer le mandrin, on doit faire bouger le rouleau, et on decide de faire pivoter le rouleaux par rapport à l'axe du mandrin. On decide d'être dans ce cas sila direction que le mandrin demande au rouleau est opposée à la direction que le rouleau veut emprunter
        if reglages is not None and reglages.get('pivoter rouleau', False) :
          
          direction_rouleau  = np.array([gradient[2*i], gradient[2*i+1]])
          direction_rouleau /= np.linalg.norm(direction_rouleau) + 1e-10
          direction_demandee_mandrin = direction_pple_mandrin /(np.linalg.norm(direction_pple_mandrin)+1e-10)
          if np.dot  (direction_rouleau, direction_demandee_mandrin) < 0 :
            dir_rouleau = np.dot(direction_rouleau, e_thetas[i]) * e_thetas[i] 
            dir_rouleau/= dir_rouleau + 1e-10

            dir_mandrin = np.dot(direction_demandee_mandrin, e_thetas[i]) * e_thetas[i]
            gradient[2*n+i] = dir_mandrin[0]
            bibo_i['cas special'] = 'cas special 1.5 : pour libérer le mandrin, fait pivoter le rouleau selon l\'axe de rotation du mandrin'

            # Pour calculer la longueur on s'interesse au bibo le plus proche tel qu'il pousse le bibo dans une direction opposee à celle vers laquelle on se dirige
            dist_min = float('inf')
            for j, dist in enumerate(bibo_i['distances']):
              if dist < dist_min and np.dot(dir_rouleau, bibo_i['autres grads'][j]) < 0:
                dist_min = dist
              longueur = dist_min - bibo_i['distance minimale rouleau']
              gradient[2*i] = dir_rouleau[0] * longueur
              gradient[2*i+1] = dir_rouleau[1] * longueur

       # Cas special 1 : Le mandrin est coincé dans un "coin"
        elif bibo_i['distance minimale mandrin'] < bibo_i['distance minimale rouleau'] and bibo_i['deuxieme distance minimale mandrin'] < bibo_i['distance minimale rouleau']:
>>>>>>> e1d9b9f (Premier commit : ajout des fichiers Python)
            bibo_i['cas special'] = 'cas special 1 : le mandrin est coincé dans un coin'
            pond = 1/ (1+np.abs(bibo_i['distance minimale mandrin'] - bibo_i['deuxieme distance minimale mandrin']))
            a = 1/ (1+ pond)
            direction_pple_mandrin_1 = direction_pple_mandrin /(np.linalg.norm(direction_pple_mandrin)+1e-10)
            coeff_rouleau_prime =  bibo_i['distance minimale rouleau'] - bibo_i['distance minimale mandrin']
            i_min = np.argmin(bibo_i['distances'])
            if bibo_i['distances'][i_min] != bibo_i['deuxieme distance minimale mandrin']:
               print('bleubleubleu')
            deuxieme_direction_pple_mandrin = bibo_i['autres grads'][i_min]
            direction_pple_mandrin_2 = deuxieme_direction_pple_mandrin /(np.linalg.norm(deuxieme_direction_pple_mandrin)+1e-10)
            direction_rouleau = a * (direction_pple_mandrin_1 + pond *direction_pple_mandrin_2 )
            gradient[2*i] = direction_rouleau[0]*coeff_rouleau_prime
            gradient[2*i+1] = direction_rouleau[1]*coeff_rouleau_prime

       # Cas spécial 2 : Le mandrin est coincé contre un "bord"
        elif bibo_i['distance minimale mandrin'] < bibo_i['distance minimale rouleau']:
            bibo_i['cas special'] = 'cas special 2 : le mandrin est coincé contre un bord'
            pond = 1/ (1+np.abs(bibo_i['distance minimale mandrin'] - bibo_i['distance minimale rouleau']))
            a = 1/ (1+ pond)
            coeff_rouleau_prime =  bibo_i['distance minimale rouleau'] - bibo_i['distance minimale mandrin']
            direction_pple_rouleau = direction_pple_rouleau /(np.linalg.norm(direction_pple_rouleau)+1e-10)
            direction_pple_mandrin = direction_pple_mandrin /(np.linalg.norm(direction_pple_mandrin)+1e-10)
            direction_rouleau = a * (direction_pple_mandrin + pond *direction_pple_rouleau )
            gradient[2*i] = direction_rouleau[0]*coeff_rouleau_prime
            gradient[2*i+1] = direction_rouleau[1]*coeff_rouleau_prime

<<<<<<< HEAD
    maj_gradient.d_min_rouleau = d_min_rouleau
    maj_gradient.d_min_mandrin = d_min_mandrin
    return(gradient)










=======

    maj_gradient_v2.d_min_rouleau = d_min_rouleau
    maj_gradient_v2.d_min_mandrin = d_min_mandrin
    return(gradient)


>>>>>>> e1d9b9f (Premier commit : ajout des fichiers Python)
