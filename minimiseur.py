
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
import string
import random

from matplotlib.patches import Rectangle, Circle
import matplotlib.animation as animation
matplotlib.rcParams['animation.embed_limit'] = 100 * 1024 * 1024  # 100 Mo
from matplotlib.animation import PillowWriter
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

def legitime_bien_aligne(arrangement, reglages):
    ### FONCTION DE VALIDATION : ON VÉRIFIE LA LÉGITIMITÉ ET L’ALIGNEMENT
    reglages_par_defaut = {
        'seuil': float('inf')
    }
    if reglages is not None:
        reglages = reglages_par_defaut | reglages
    else:
        reglages = reglages_par_defaut
    ## CONDITIONS
        # 1. L’arrangement doit respecter toutes les contraintes géométriques
        # 2. Il doit être suffisamment bien aligné (perte < seuil défini)

    return (
        est_legitime(arrangement, reglages)
        and perte_alignement(arrangement, reglages) <= reglages['seuil']
    )

def arranger (arrangement, reglages=None) :
  ### PREPARATION


    ## ON EXTRAIT LES REGLAGES
      #On gère les réglages par défaut
  reglages_par_defaut = {
      'solveur': 'BFGS',
      'options' : {'maxiter' : 200, 'gtol' : 1e-8, 'eps' : 1e-8},
      'pellicule' : 1.1, 
      }
  if reglages is None:
    reglages = reglages_par_defaut
  else :
    reglages = reglages_par_defaut | reglages    
      #on récupère ceux dont on a besoin
  methode = reglages['solveur']
  options = reglages['options']
  pellicule = reglages['pellicule']


    ##ON EXTRAIT LES DONNEES
  arrangement_interne = copy.deepcopy(arrangement)#on fait une copie pour éviter de modifier l'arrangement d'entrée de manière intempestive
  centres_2d = arrangement_interne['centres']
  rayon = arrangement_interne['rayon']
  longueur = arrangement_interne['longueur']
  axe = arrangement_interne['axe']
  origine_carton = arrangement_interne['origine du carton']
  dimensions_carton = arrangement_interne['dimensions du carton']


    ##ON PREPARE LES OBJETS
      # On calcul la coordonnée à modifier 
  direction_longueur = np.dot([0,1,2], axe) # But : savoir qulle est la coordonée qu'il va falloir forcer
  coord_commune = origine_carton[direction_longueur] + longueur / 2  
      #On convertit les coordonnées en vecteur 1d
  centres_1d = t_1_d(centres_2d)
      # On projete les infos qui nous interessent (on ne veut pas modifier les coordonnées selon l'axe du cylindre)
  centres_1d_projete = projection(centres_1d, axe)

    ##ON DEFINIT LA FONCTION À MINIMISER
  def a_minimiser(centres_cylindres_1d_projetes):
        return perte_totale_1d_2d(injection(centres_cylindres_1d_projetes, axe, coord_commune ), pellicule*rayon, longueur, dimensions_carton, axe, origine_carton, reglages)
      
    ## ON PRÉPARE LA COMMUNICATION AVEC L'ICC
      #On se donne une liste pour stocker les frames du futur film   
  valeurs_intermediaires = []
      #On se donne un compteur pour pouvoir, dans le film, afficher l'itération à laquelle on est
  compteur_iteration_minimiseur = 0
      #On se donne une fonction callback pour y stocker les essais visités
  def store_points(xk):
      nonlocal compteur_iteration_minimiseur, valeurs_intermediaires
      valeurs_intermediaires.append({'centres' : t_2_d(injection(np.copy(xk), axe, coord_commune)), 'rayon': rayon, 'code couleur' : 0, 'texte': 'En cours de rangement', 'perte' : perte_totale_1d_2d(injection(xk, axe, coord_commune ), rayon, longueur, dimensions_carton, axe, origine_carton, reglages), 'iteration du minimiseur' : compteur_iteration_minimiseur})
      compteur_iteration_minimiseur += 1

  ### MINIMISATION
    # on execute la minimisation
  sortie_brute = minimize(a_minimiser,centres_1d_projete, method=methode, callback=store_points, options=options)
    # On vérifie qu'elle a bien convergée
  if not sortie_brute.success:
    print("Un tour d'optimisation de plus, message :", sortie_brute.message)
    # on retransforme les résultats en 2d
  centres_proposes_1d_projetes = sortie_brute.x  
  centres_proposes_1d = injection(centres_proposes_1d_projetes, axe=axe, valeur=coord_commune)
  centres_proposes_2d = t_2_d(centres_proposes_1d)

  ### SORTIE

    ##REPACKAGING
  arrangement_sortie = copy.deepcopy(arrangement_interne)
  arrangement_sortie['centres'] = centres_proposes_2d

    ##GESTION DES ATTRIBUTS
  arranger.valeurs_intermediaires = valeurs_intermediaires
  arranger.message = sortie_brute.message
  arranger.stag_0_detectee = False

    ##RETOUR
  return arrangement_sortie

def arranger_v2(arrangement, reglages=None):
 ### PREPARATION

  ## ON DEFINIT ET EXTRAIT LES REGLAGES
    reglages_par_defaut = {
        'solveur': 'BFGS',
        'options' : {'maxiter' : 500, 'gtol' : 1e-8, 'eps' : 1e-8},
        'pellicule' : 1.15, 
        'iterations par tour' : 5, 
        'borne sup perte' : float(inf),
        'n_tours_admissibles' : 30,
        'stagnation acceptee': 7,
        'ecart stagnation': 0.05,
        'stagnation autour de 0': 15
        
    }
    if reglages is None:
        reglages = reglages_par_defaut
    else:
        reglages = reglages_par_defaut | reglages    

    methode = reglages['solveur']
    options = reglages['options']
    pellicule = reglages['pellicule']
    iter_par_tour = reglages['iterations par tour']
    borne_perte = reglages['borne sup perte']
    n_tours_admissibles = reglages['n_tours_admissibles']
    n_stag = reglages['stagnation acceptee']
    ecart_stag = reglages['ecart stagnation']
    borne_stag_nulle = reglages['borne stagnation nulle']
    borne_stag_0_acc = reglages['borne stagnation nulle acceleree']
    n_stag_0 = reglages['stagnation autour de 0']
    version_acceleree = reglages['version acceleree']

  ##ON EXTRAIT LES DONNEES
    arrangement_interne = copy.deepcopy(arrangement)
    centres_2d = arrangement_interne['centres']
    rayon = arrangement_interne['rayon']
    longueur = arrangement_interne['longueur']
    axe = arrangement_interne['axe']
    origine_carton = arrangement_interne['origine du carton']
    dimensions_carton = arrangement_interne['dimensions du carton']

  ##ON PREPARE LES OBJETS
    direction_longueur = np.dot([0,1,2], axe)
    coord_commune = origine_carton[direction_longueur] + longueur / 2  
    centres_1d = t_1_d(centres_2d)
    centres_1d_projete = projection(centres_1d, axe)
    stag_0 = None
    stag = None

  ##ON DEFINIT LA FONCTION À MINIMISER
    def a_minimiser(centres_cylindres_1d_projetes):
        return perte_totale_1d_2d(
            injection(centres_cylindres_1d_projetes, axe, coord_commune),
            pellicule * rayon, longueur, dimensions_carton, axe, origine_carton, reglages
        )

  ##COMMUNICATION AVEC ICC
    valeurs_intermediaires = []
    compteur_iteration_minimiseur = 0
    pertes_recentres = []

    def store_points(xk):
        nonlocal compteur_iteration_minimiseur, valeurs_intermediaires, pertes_recentres
        perte = perte_totale_1d_2d(
            injection(xk, axe, coord_commune), rayon, longueur, dimensions_carton, axe, origine_carton, reglages)
        valeurs_intermediaires.append({
            'centres': t_2_d(injection(np.copy(xk), axe, coord_commune)),
            'rayon': rayon,
            'code couleur': 0,
            'texte': 'En cours de rangement',
            'perte': perte,
            'iteration du minimiseur': compteur_iteration_minimiseur})
        pertes_recentres.append(perte)
        if len(pertes_recentres) > max(n_stag, n_stag_0):
            pertes_recentres.pop(0)
        compteur_iteration_minimiseur += 1

 ### MINIMISATION EN BOUCLE
    converged = False
    max_tours = options['maxiter'] // iter_par_tour
    centres_1d_projete_temporaire = centres_1d_projete

    for tour in range(max_tours):

  ## APPLICATION DE MINIMIZE
        sortie_brute = minimize(a_minimiser, centres_1d_projete_temporaire, method=methode, callback=store_points, options={**options, 'maxiter': iter_par_tour})
        x = sortie_brute.x
        centres_1d_projete_temporaire = x
        centres_2d_temporaires = t_2_d(injection(centres_1d_projete_temporaire, axe, coord_commune))
        arrangement_temporaire = copy.deepcopy(arrangement_interne)
        arrangement_temporaire['centres'] = centres_2d_temporaires

  ## CONDITIONS D'ARRÊT
        perte_actuelle = perte_total_2d(arrangement_temporaire, reglages)

            #Première condition d'arrêt : on a un arrangement valide
        if legitime_bien_aligne(arrangement_temporaire, reglages):
            converged = True
            print('est legitime')
            arrangement_sortie = copy.deepcopy(arrangement_temporaire)
            break

            #Deuxième condition d'arrêt : on n'arrive pas à descendre sous un seuil
        if perte_actuelle >= borne_perte and tour <= n_tours_admissibles:
            converged = True
            arrangement_sortie = copy.deepcopy(arrangement_temporaire)
            break


        if len(pertes_recentres) >= ( n_stag_0):
                #Troisième condition d'arrêt : on a stagné proche de zéro
            if len(pertes_recentres) >= n_stag_0:
                dernieres_pour_stag_0 = pertes_recentres[-n_stag_0:]
            if all(abs(p) <= borne_stag_nulle for p in dernieres_pour_stag_0) or version_acceleree and all(abs(p) <= borne_stag_0_acc for p in dernieres_pour_stag_0):
                print('Stagnation autour de 0 détectée')
                print('on a une perte de ', dernieres_pour_stag_0[-1])
                stag_0 = dernieres_pour_stag_0[-1]
                converged = True
                arrangement_sortie = copy.deepcopy(arrangement_temporaire)
                break

        if len(pertes_recentres) >= ( n_stag):
            dernieres_pour_stag = pertes_recentres[-n_stag:]
            moyenne = np.mean(dernieres_pour_stag)
                # Quatrième condition d'arrêt : on a stagné tout court
            if all(abs(p - moyenne) <= ecart_stag * moyenne for p in dernieres_pour_stag) and  moyenne >= borne_stag_nulle:
                print('Stagnation détectée')
                converged = True
                arrangement_sortie = copy.deepcopy(arrangement_temporaire)
                stag = moyenne
                break

 ### SORTIE
    if not converged:
        centres_proposes_1d = injection(x, axe=axe, valeur=coord_commune)
        centres_proposes_2d = t_2_d(centres_proposes_1d)
        arrangement_sortie = copy.deepcopy(arrangement_interne)
        arrangement_sortie['centres'] = centres_proposes_2d

    arranger_v2.valeurs_intermediaires = valeurs_intermediaires
    arranger_v2.message = sortie_brute.message
    arranger_v2.stag_0_detectee = stag_0
    arranger_v2.stag = stag
    return arrangement_sortie

def ajuster_rayon(arrangement, reglages=None):
    reglages_par_defaut = {'precision' : 1e-8}
    if reglages is None :
      reglages= reglages_par_defaut
    else :
      reglages = reglages_par_defaut | reglages
    # Copie pour ne pas modifier l'original
    arr_test = copy.deepcopy(arrangement)

    # Rayon minimal (toujours légitime)
    r_min = 0.0

    # Rayon maximal initial : on prend la plus petite dimension du carton / 2
    dims = arrangement['dimensions du carton']
    r_max = max(dims) / 2

    precision = reglages['precision']

    # Dichotomie
    while r_max - r_min > precision:
        r_mid = (r_min + r_max) / 2
        arr_test['rayon'] = r_mid
        if est_legitime(arr_test, reglages):
            r_min = r_mid  # C'est valide → on peut essayer plus grand
        else:
            r_max = r_mid  # Trop grand → on réduit

    arr_test['rayon'] = r_min
    return arr_test

def ranger(arrangement, reglages):
  reglages_par_defaut = {'fonction arrangement' : arranger_v2}
  if reglages is None:
    reglages = reglages_par_defaut
  else :
    reglages = reglages_par_defaut | reglages
  fonction_arranger = reglages['fonction arrangement']
  arrangement_sortie = copy.deepcopy(arrangement)
  arrangement_sortie = fonction_arranger(arrangement_sortie, reglages)
  ranger.valeurs_intermediaires = fonction_arranger.valeurs_intermediaires
  ranger.indicateurs = {'stag 0' : fonction_arranger.stag_0_detectee, 'stag':fonction_arranger.stag }
  return(arrangement_sortie)

def inch_allah(arrangement, reglages=None):
    print("incha'allah lancé")
    debut = time.time()

    # Réglages par défaut
    reglages_par_defaut = {
        'fonction pertes inch': perte_inch_v2,
        'epsilon' : 1
    }
    if reglages is not None:
        reglages = reglages_par_defaut | reglages
    else:
        reglages = reglages_par_defaut

    perte_inch = reglages['fonction pertes inch']

    # Copie pour ne pas modifier l'arrangement d'origine
    arrangement_modifiable = copy.deepcopy(arrangement)
    centres_3d = np.array(arrangement_modifiable['centres'])
    axe = np.array(arrangement_modifiable['axe'], dtype=float)
    rayon = arrangement_modifiable['rayon']
    origine = np.array(arrangement_modifiable['origine du carton'], dtype=float)
    point_sert_a_rien = np.dot(axe, centres_3d[0])

    # Projection initiale en 2D
    x0 = projection(t_1_d(centres_3d), axe)

    # Fonction à minimiser
    def a_minimiser(x_2d):
        centres_projectes = t_2_d(injection(x_2d, axe, point_sert_a_rien)) + origine
        arrangement_modifiable['centres'] = centres_projectes.tolist()
        perte, gradient = perte_inch(arrangement_modifiable)
        return perte,  gradient  # Le facteur 0.1 peut être ajusté si besoin

    # Optimisation
    res = resol_a_la_main(x0, a_minimiser, rayon, axe, point_sert_a_rien, projeter_sur_plan(arrangement_modifiable['dimensions du carton'], axe) )
    valeurs_intermediaires = resol_a_la_main.valeurs_intermediaires
    

    # Reconstruction finale
    x_final = res
    centres_optim_3d = t_2_d(injection(x_final, axe, point_sert_a_rien)) + origine
    arrangement_modifiable['centres'] = centres_optim_3d.tolist()

    # Ajustement du rayon (si requis)
    arrangement_final = ajuster_rayon(arrangement_modifiable, reglages)
    valeurs_intermediaires.append({
            'centres': arrangement_final['centres'],
            'rayon': arrangement_final['rayon'],
            'code couleur': 1,
            'texte': 'rayon ajusté',
            'perte': 0,
            'iteration du minimiseur': 0})

    print("incha'allah terminé en", round(time.time() - debut, 2), "s")
    inch_allah.valeurs_intermediaires = valeurs_intermediaires
    return arrangement_final

def perte_inch_v1(arrangement, reglages=None):
    def projeter_sur_plan(point, axe):
        axes_du_plan = []
        for e in np.eye(3):
            if not np.array_equal(e, axe):
                axes_du_plan.append(e)
        projete = np.zeros(2)
        projete[0] = np.dot(point, axes_du_plan[0])
        projete[1] = np.dot(point, axes_du_plan[1])
        return projete

    centres = np.array(arrangement['centres'])
    axe = np.array(arrangement['axe'], dtype=float)
    rayon = arrangement['rayon']
    dimensions = np.array(arrangement['dimensions du carton'], dtype=float)
    origine = np.array(arrangement['origine du carton'], dtype=float)
    n = len(centres)

    # Projection des centres
    centres_2d = []
    for centre in centres:
        centre_relatif = centre - origine
        proj = projeter_sur_plan(centre_relatif, axe)
        centres_2d.append(proj)
    centres_2d = np.array(centres_2d)

    # Projection des limites du carton
    coins = [origine, origine + dimensions]
    coins_proj = [projeter_sur_plan(c - origine, axe) for c in coins]
    min_proj = np.minimum(coins_proj[0], coins_proj[1])
    max_proj = np.maximum(coins_proj[0], coins_proj[1])

    perte = 0.0
    points_proches = []

    for i in range(n):
        x, y = centres_2d[i]
        distances = []
        correspondances = []

        # Distance aux autres centres
        for j in range(n):
            if i != j:
                d = np.linalg.norm(centres_2d[i] - centres_2d[j]) - 2 * rayon
                distances.append(d)
                correspondances.append(centres_2d[j])

        # Distance aux bords
        bords = [
            (np.array([min_proj[0] + rayon, y]), abs(x - min_proj[0] - rayon)),  # gauche
            (np.array([max_proj[0] - rayon, y]), abs(max_proj[0] - x - rayon)),  # droite
            (np.array([x, min_proj[1] + rayon]), abs(y - min_proj[1] - rayon)),  # bas
            (np.array([x, max_proj[1] - rayon]), abs(max_proj[1] - y - rayon))   # haut
        ]
        for pt, d in bords:
            distances.append(d)
            correspondances.append(pt)

        distances = np.array(distances)
        correspondances = np.array(correspondances)

        indices_trie = np.argsort(distances)
        perte += distances[indices_trie[2]]
        points_proches.append(correspondances[indices_trie[2]])

    ## Calcul du gradient
    centres_projetes = [projeter_sur_plan(centre, axe) for centre in centres]
    e_x = [1,0]
    e_y = [0,1]
    n = len(centres_projetes)
    gradient = np.zeros(2*n)

    for i in range(n):
      [dp_x, dp_y] = (points_proches[i]-centres_projetes[i])/np.linalg.norm(points_proches[i]-centres_projetes[i])
      gradient[2*i] = -dp_x
      gradient[2*i +1] = -dp_y

    return perte, gradient

def perte_inch_v2(arrangement, reglages=None):
    import numpy as np

    centres = np.array(arrangement['centres'])
    axe = np.array(arrangement['axe'], dtype=float)
    rayon = arrangement['rayon']
    dimensions = np.array(arrangement['dimensions du carton'], dtype=float)
    origine = np.array(arrangement['origine du carton'], dtype=float)
    n = len(centres)

    # Projection des centres dans le plan orthogonal à l’axe
    centres_2d = np.array([projeter_sur_plan(c - origine, axe) for c in centres])

    # Limites du carton projeté (avec marge liée au rayon)
    coins = [origine, origine + dimensions]
    coins_proj = [projeter_sur_plan(c - origine, axe) for c in coins]
    min_proj = np.minimum(coins_proj[0], coins_proj[1]) + +0.01
    max_proj = np.maximum(coins_proj[0], coins_proj[1]) - 0.01
    dimensions = projeter_sur_plan(dimensions, axe)
    origine = projeter_sur_plan(origine, axe)


    # Initialiser avec une valeur élevée
    min_d = float('inf')
    min_i = []
    min_pt = []

    # Rechercher la paire la plus proche (entre bibos et bords)
    for i in range(n):
        pi = centres_2d[i]

        # Paires avec les autres bibos
        for j in range(n):
            if j == i:
                continue
            pj = centres_2d[j]
            d = np.linalg.norm(pi - pj) -2*rayon
            if d < min_d:
                min_d = d
                min_i = [i]
                min_pt = [pj]
            elif d == min_d:
                min_i.append(i)
                min_pt.append(pj)

        # Bords du carton
        x, y = pi
        bords = [
            (np.array([origine[0], y]), abs(x - min_proj[0])-rayon),
            (np.array([origine[0]+dimensions[0], y]), abs(x - origine[0]-dimensions[0])-rayon),
            (np.array([x, origine[1]]), abs(y - origine[1])-rayon),
            (np.array([x, origine[1]+dimensions[1]]), abs(y -  origine[1]-dimensions[1])-rayon)
        ]
        for pt_bord, d in bords:
            if 2*d < min_d:
                min_d = 2*d
                min_i = [i]
                min_pt = [pt_bord]
            elif 2*d == min_d:
                min_i.append(i)
                min_pt.append(pt_bord)

    # Calcul de la perte et du gradient uniquement pour la plus petite distance
    perte = -min_d
    gradient = np.zeros(2 * n)
    for i in range(len(min_i )):
        pi = centres_2d[min_i[i]]
        direction = (pi - min_pt[i]) / (np.linalg.norm(pi - min_pt[i]) + 1e-8)
        gradient[2 * min_i[i]] = direction[0]
        gradient[2 * min_i[i] + 1] = direction[1]


        # Forcer les points à rester strictement à l’intérieur
    for i in range(n):
        centres_2d_proj = np.clip(centres_2d[i], min_proj + 1e-3, max_proj - 1e-3)
        if not np.allclose(centres_2d_proj,centres_2d[i]):
            gradient[2 * i] = -10*(centres_2d_proj[0] - pi[0]) / (np.linalg.norm(centres_2d_proj[0] - pi[0]) + 1e-8)
            gradient[2 * i+1] = 10*(centres_2d_proj[1] - pi[1]) / (np.linalg.norm(centres_2d_proj[1] - pi[1]) + 1e-8)

    
    perte_inch_v2.min_i = min_i
    perte_inch_v2.min_pt = min_pt
    perte_inch_v2.min_d = min_d
    return perte, gradient

def perte_inch_v3(arrangement, reglages=None):
    import numpy as np

    centres = np.array(arrangement['centres'])
    axe = np.array(arrangement['axe'], dtype=float)
    rayon = arrangement['rayon']
    dimensions = np.array(arrangement['dimensions du carton'], dtype=float)
    origine = np.array(arrangement['origine du carton'], dtype=float)
    n = len(centres)

    centres_2d = np.array([projeter_sur_plan(c - origine, axe) for c in centres])
    coins = [origine, origine + dimensions]
    coins_proj = [projeter_sur_plan(c - origine, axe) for c in coins]
    min_proj = np.minimum(coins_proj[0], coins_proj[1]) + 0.01
    max_proj = np.maximum(coins_proj[0], coins_proj[1]) - 0.01

    # Stocker toutes les distances
    distances = []
    infos = []

    for i in range(n):
        pi = centres_2d[i]

        for j in range(n):
            if j == i:
                continue
            pj = centres_2d[j]
            d = np.linalg.norm(pi - pj) - 2 * rayon
            distances.append(d)
            infos.append(('bibos', i, pj, d))

        x, y = pi
        bords = [
            (np.array([origine[0], y]), abs(x - min_proj[0]) - rayon),
            (np.array([origine[0] + dimensions[0], y]), abs(x - origine[0] - dimensions[0]) - rayon),
            (np.array([x, origine[1]]), abs(y - origine[1]) - rayon),
            (np.array([x, origine[1] + dimensions[1]]), abs(y - origine[1] - dimensions[1]) - rayon)
        ]
        for pt_bord, db in bords:
            distances.append(2 * db)
            infos.append(('bord', i, pt_bord, 2 * db))

    # Trouver les deux plus petites distances
    sorted_indices = np.argsort(distances)
    idx1, idx2 = sorted_indices[:2]
    d1 = distances[idx1]
    d2 = distances[idx2]
    facteur = max(d2 - d1, 0)

    # Récupérer les données associées au plus proche
    _, i, pt_cible, _ = infos[idx1]
    pi = centres_2d[i]
    direction = (pi - pt_cible) / (np.linalg.norm(pi - pt_cible) + 1e-8)
    gradient = np.zeros(2 * n)
    gradient[2 * i] = facteur * direction[0]
    gradient[2 * i + 1] = facteur * direction[1]

    # Contraindre dans le carton
    for i in range(n):
        centres_2d_proj = np.clip(centres_2d[i], min_proj + 1e-3, max_proj - 1e-3)
        if not np.allclose(centres_2d_proj, centres_2d[i]):
            gradient[2 * i] = -10 * (centres_2d_proj[0] - centres_2d[i][0]) / (np.linalg.norm(centres_2d_proj[0] - centres_2d[i][0]) + 1e-8)
            gradient[2 * i + 1] = 10 * (centres_2d_proj[1] - centres_2d[i][1]) / (np.linalg.norm(centres_2d_proj[1] - centres_2d[i][1]) + 1e-8)

    perte_inch_v2.min_i = [i]
    perte_inch_v2.min_pt = [pt_cible]
    perte_inch_v2.min_d = d1
    perte_inch_v2.deuxieme_d = d2
    perte_inch_v2.facteur = facteur

    return -d1, gradient

def resol_a_la_main_ancienne_version(centres_2d, fonction, n_tours, pas, rayon, axe, valeur):
    n = len(centres_2d)/2 
    alpha = exp(- log(10)*n/n_tours)
    valeurs_intermediaires = []
    for i in range(n_tours):
        perte, gradient = fonction(centres_2d)
        gradient = gradient / np.linalg.norm(gradient)
        centres_2d += pas * gradient
        if i % 150 == 0:           
          valeurs_intermediaires.append({
              'centres': t_2_d(injection(np.copy(centres_2d), axe, valeur)),
              'rayon': rayon,
              'code couleur': 0,
              'texte': 'En cours de rangement',
              'perte': perte,
              'iteration du minimiseur': i})
        if i % len(centres_2d // 2) == 0:
            pas = alpha * pas
    resol_a_la_main.valeurs_intermediaires = valeurs_intermediaires
    return centres_2d

def resol_a_la_main(centres_2d, fonction,rayon, axe, valeur, dims):
    valeurs_intermediaires = []
    n = len(centres_2d)//2 
    print('dimensions : ', dims)
    n_tours =  2 * n* 20
    pas = 0.05
    for i in range(n_tours):
        perte, gradient = fonction(centres_2d)
        gradient = gradient / np.linalg.norm(gradient)
        centres_2d = clip_tab(centres_2d, dims)
        centres_2d += pas * gradient
        if i % 20 == 0:           
          valeurs_intermediaires.append({
              'centres': t_2_d(injection(np.copy(centres_2d), axe, valeur)),
              'rayon': rayon,
              'code couleur': 0,
              'texte': 'En cours de rangement',
              'perte': perte,
              'iteration du minimiseur': i})
          
    n_tours = 2 * n* 30
    pas = 0.01
    for i in range(n_tours):
        perte, gradient = fonction(centres_2d)
        gradient = gradient / np.linalg.norm(gradient)
        centres_2d = clip_tab(centres_2d, dims)
        centres_2d += pas * gradient
        if i % 20 == 0:           
          valeurs_intermediaires.append({
              'centres': t_2_d(injection(np.copy(centres_2d), axe, valeur)),
              'rayon': rayon,
              'code couleur': 0,
              'texte': 'En cours de rangement',
              'perte': perte,
              'iteration du minimiseur': i})
        
    n_tours = 2 * n* 30
    pas = 0.01
    for i in range(n_tours):
        perte, gradient = fonction(centres_2d)
        gradient = gradient / np.linalg.norm(gradient)
        centres_2d = clip_tab(centres_2d, dims)
        centres_2d += pas * gradient
        if i % n == 0:           
          pas = 0.9 * pas
        if i % 20 == 0:           
          valeurs_intermediaires.append({
              'centres': t_2_d(injection(np.copy(centres_2d), axe, valeur)),
              'rayon': rayon,
              'code couleur': 0,
              'texte': 'En cours de rangement',
              'perte': perte,
              'iteration du minimiseur': i})

    resol_a_la_main.valeurs_intermediaires = valeurs_intermediaires
    return centres_2d

def perturbation_propre(arrangement, reglages = None) :
  reglages_par_defaut = {'perturbation': perturbation_trou_aleatoire}
  if reglages is None:
    reglages = reglages_par_defaut
  else :
    reglages = reglages_par_defaut | reglages
  pert = reglages['perturbation']
  
  centres_perturbes = pert(arrangement)
  arrangement_sortie = copy.deepcopy(arrangement)
  arrangement_sortie['centres'] = centres_perturbes
  return(arrangement_sortie)
  return 

def minimiseur(proposition_actuelle, reglages=None):
 ### PREPARATION

  ## GESTION DES REGLAGES

   # on définit des réglages par défaut
  reglages_par_defaut = {
      'n_tours_minimiseur': 8,
      'n_tours_stag': 6, 
      'version acceleree' : False, 
      'approximation version acceleree' :0.005,
      'borne stagnation nulle acceleree' : 20,
      'borne stagnation nulle': 1,
      'n_tours_stag_0' : 6,
      'n_tours_stag_0_acc':6}
  
  if reglages is not None:
      reglages = reglages_par_defaut | reglages
  else:
      reglages = reglages_par_defaut
      
  n_tours = reglages['n_tours_minimiseur']
  print('c\'est parti pour', n_tours, 'de minimiseur')


  ## CREATION DES VARIABLES
  solution_actuelle = proposition_actuelle
  valeurs_intermediaires = []
  indicateurs = initialiser_indicateurs()

 ### BOUCLE
  for i in range(n_tours) :
    print('on entre dans le tour  : ' , i+1)

    ## perturbation 
    print('on perturbe')
    proposition_perturbee = perturbation_propre(solution_actuelle, reglages)
    update_vi (valeurs_intermediaires, proposition_perturbee, 'perturbation') 
  
    ## rangement
    print('on range')
    proposition_perturbee_rangee = ranger(proposition_perturbee, reglages)
     #on met à jour les indicateurs
    valeurs_intermediaires+= ranger.valeurs_intermediaires
    indicateurs = indicateurs | ranger.indicateurs

    ## mise a jour de la solution et des indicateurs
    print('on met à jour la solution si elle est meilleure')
    solution_actuelle = mise_a_jour(solution_actuelle, proposition_perturbee_rangee, indicateurs, reglages)  
    
    ## vérification des condition
    print('on check les conditions d\'arrêt ')
    if conditions_arret(proposition_perturbee_rangee, indicateurs,  reglages):
      solution_actuelle = conditions_arret.solution
      break

  minimiseur.valeurs_intermediaires = valeurs_intermediaires
  minimiseur.rayon_actuel = solution_actuelle['rayon']
  minimiseur.indicateur_acceleration = indicateurs['acceleration']
  return solution_actuelle

def update_vi (valeurs_intermediaires, proposition, type):
  if type == 'perturbation':
    vi = {'centres' : proposition['centres'], 'rayon': proposition['rayon'], 'code couleur' : 1, 'texte': 'Après avoir perturbé', 'perte' :0,'itération du minimiseur' : 0}
  else :
    print('pas de mise à jour')
  return(vi)

def initialiser_indicateurs ():
  indicateurs = {
    'tours_stag' : 0,
    'acceleration' : False, 
    'historique_pertes' : [],
    'valeur_stag': float('inf'), 
    'tours_stag_0': 0,
    'stag 0': None,
    'tours_stag_acc' : 0

  }
  return(indicateurs)

def conditions_arret(proposition, indicateurs, reglages):
  n_tours_stag = reglages['n_tours_stag']
  version_acceleree = reglages['version acceleree']
  erreur_acc = reglages['approximation version acceleree']
  borne_stag_0 = reglages['borne stagnation nulle']
  borne_stag_0_acc = reglages['borne stagnation nulle acceleree']

  stag_0 = indicateurs['stag 0']
  stag = indicateurs['stag'] #par defaut, valent None. Sinon, c'est qu'il y a une stagnation
  solution_actuelle = proposition
  proposition_inchee = inch_allah(proposition, reglages)

  arret = False

 #Stagnation à 0 version accélérée
  if stag_0 is not None :
    if stag_0 >= borne_stag_0 and version_acceleree and stag_0 <= borne_stag_0_acc:
      print('on est dans une stagnation autour de 0, version accélérée')
      #on essaie de voir si on peut court-circuiter

      #on propose un solution inchallée
      proposition_court_circuit = proposition_inchee

      #si elle est suffisament proche, on la garde
      if abs(proposition_court_circuit['rayon'] - proposition['rayon']) <= erreur_acc:
        print('on a trouvé par une solution à peu près convenable, avec une erreur d\'au plus', erreur_acc, ', on court-circuite donc la fin de la dichotomie')
        indicateurs['acceleration'] = True
        solution_actuelle = proposition_court_circuit
        arret = True
      else : 
        print('On a pas réussi à conclure, rayon trop petit : (',proposition_court_circuit['rayon'], ')' )
        indicateurs['tours_stag_acc'] += 1
      if indicateurs['tours_stag_acc'] >= reglages['n_tours_stag_0_acc']:
        print('on a stagné proche de 0 (mais pas trop) trop longtemps')
        arret = True

 #Stagnation à 0
  if stag_0 is not None :
    if stag_0 <= borne_stag_0 :
      print('on est dans une stagnation autour de 0, version classique. On regarde si il manquait juste un coup de pouce pour tout recaler bien')

      #on essaie de bien recaler les rouleaux
      proposition_coup_pouce = proposition_inchee
      if abs(proposition_coup_pouce['rayon']) >= proposition['rayon'] - max(reglages['precision'], 0.5*reglages['epsilon']):
        print('on a trouvé une solution en recalant bien les rouleaux')
        arret = True
      else :
        print('apparament, il manquait plus qu\'un coup de pouce')
        indicateurs['tours_stag_0'] += 1
      if indicateurs['tours_stag_0'] >= reglages['n_tours_stag_0']:
        print('on a stagné proche de 0 trop longtemps')
        arret = True


 #Stagnation
  if stag is not None :
    ecart_relatif = abs(stag - indicateurs['valeur_stag'])/stag 
    if ecart_relatif <= 0.05 :
      indicateurs['tours_stag'] +=1
      print('On a stagné pour le ' , indicateurs['tours_stag'], 'ème tour à la valeur ', indicateurs['valeur_stag'])
    elif stag <= indicateurs['valeur_stag']*0.95:
      indicateurs['tours_stag'] = 1
      indicateurs['valeur_stag'] = stag
      print('On a atteint une nouvelle valeur de stagnation : ', indicateurs['valeur_stag'])
    if indicateurs['tours_stag'] >= n_tours_stag :
      print('On a trop stagné, pendant ', indicateurs['tours_stag'], ' tours consécutifs au dessus de la valeur : ', indicateurs['valeur_stag']*0.95)
      solution_actuelle = proposition
      arret = True

 #Convergence
  if legitime_bien_aligne(proposition, reglages) :
    print('On a trouvé une solution légitime ')
    solution_actuelle = proposition
    arret = True

 #retour
  conditions_arret.solution = solution_actuelle
  return arret

def mise_a_jour(solution_actuelle, proposition_perturbee_rangee, indicateurs, reglages) :
  inchee = inch_allah(proposition_perturbee_rangee, reglages)
  perte_actuelle = perte_total_2d(solution_actuelle, reglages)
  perte_perturbee = perte_total_2d(proposition_perturbee_rangee, reglages)
  perte_inchee = perte_total_2d(inchee, reglages)
  solution_nouvelle = solution_actuelle
  perte_nouvelle = perte_actuelle
  if legitime_bien_aligne(solution_actuelle, reglages):
    indicateurs['historique_pertes'].append(perte_nouvelle)
    print('l\'ancienne solution était satisfaisante et légitime')
    return(solution_nouvelle)

  if perte_perturbee < perte_actuelle or legitime_bien_aligne(proposition_perturbee_rangee, reglages) :
    print('la solution perturbee/rearrangee fonctionne !')
    solution_nouvelle = proposition_perturbee_rangee
    perte_nouvelle = perte_perturbee
  
  if inchee['rayon'] >= solution_actuelle['rayon'] :
    print('incha\'allah nous permet une masterclass ' )
    inchee['rayon'] = solution_actuelle['rayon']
    solution_nouvelle = inchee
    perte_nouvelle = perte_inchee

  indicateurs['historique_pertes'].append(perte_nouvelle)
  return(solution_nouvelle)

def generer_chaine_aleatoire(longueur=5):
    caracteres = string.ascii_letters + string.digits
    return ''.join(random.choice(caracteres) for _ in range(longueur))

def generer_nom(arrangement, suffixe=None):
    n_bibos = len(arrangement['centres'])
    dims = arrangement['dimensions du carton']
    axe = arrangement['axe']
    if suffixe is None:
        suffixe = generer_chaine_aleatoire(4)
    dim_str = '_'.join([f"{d:.2f}" for d in dims])
    axe_str = ''.join(map(str, axe))
    return f"n{n_bibos}_dim{dim_str}_axe{axe_str}_{suffixe}"

def minimiseur_foireux(arrangement, reglages=None):
    reglages_par_defaut = {'n_iterations': 10}
    if reglages is None:
        reglages = reglages_par_defaut
    else:
        reglages = reglages_par_defaut | reglages

    # Normalisation
    dimensions_carton = np.array(arrangement['dimensions du carton'])
    axe = np.array(arrangement['axe'])
    plan = np.array([1, 1, 1]) - axe
    norm = 0.5 * np.dot(plan, dimensions_carton)
    dimensions_carton_normalisees = dimensions_carton / norm
    arrg =dilater_arrangement(arrangement, dimensions_carton_normalisees, [0, 0, 0], interaction=False)
    print('nouvelles dimensions pour les calculs :', dimensions_carton_normalisees)

    meilleur_arrangement = ajuster_rayon(arrg, reglages)
    meilleur_rayon = meilleur_arrangement['rayon']

    arrangements = [meilleur_arrangement]
    rayons = [meilleur_rayon]
    valeurs_intermediaires = []

    for i in range(reglages['n_iterations']):
        print('itération', i)

        index_base = tirage_softmax_milieu(rayons)
        base = arrangements[index_base]

        proposition = perturbation_propre(base, reglages)
        update_vi(valeurs_intermediaires, proposition, 'perturbation')
        

        proposition = inch_allah(proposition, {'epsilon': 1})
        valeurs_intermediaires += inch_allah.valeurs_intermediaires

        rayon_propose = proposition['rayon']
        print('rayon cette iteration :', rayon_propose)

        arrangements.append(proposition)
        rayons.append(rayon_propose)

        if rayon_propose > meilleur_rayon:
            meilleur_arrangement = proposition
            meilleur_rayon = rayon_propose

    # Dénormalisation finale
    arrangement_final = meilleur_arrangement.copy()
    arrangement_final = dilater_arrangement(arrangement_final, dimensions_carton, [0, 0, 0], interaction=False)
    arrangement_final['nom'] = generer_nom(arrangement_final)

    # Historique
    minimiseur_foireux.valeurs_intermediaires = valeurs_intermediaires
    minimiseur_foireux.historique_rayons = rayons
    minimiseur_foireux.historique_arrangements = arrangements

    return arrangement_final

def tirage_softmax_milieu(rayons, precision=0.01, cible=0.5):

    rayons = np.array(rayons)
    centre = rayons - np.mean(rayons)

    def proba_max_pour_sigma(sigma):
        scores = centre / sigma
        exp_scores = np.exp(scores - np.max(scores))  # stabilité numérique
        probs = exp_scores / np.sum(exp_scores)
        return np.max(probs)

    # Dichotomie sur sigma
    sigma_min, sigma_max = 1e-6, 1e6
    for _ in range(100):
        sigma = (sigma_min + sigma_max) / 2
        p = proba_max_pour_sigma(sigma)
        if abs(p - cible) < precision:
            break
        if p > cible:
            sigma_min = sigma
        else:
            sigma_max = sigma

    # Tirage
    scores = centre / sigma
    probs = np.exp(scores - np.max(scores))
    probs /= np.sum(probs)
    index = np.random.choice(len(rayons), p=probs)
    return index

def clip_tab(tableau_xy, dimensions):
  
    tableau = np.array(tableau_xy)
    x_max, y_max = dimensions
    epsilon = 1e-8
    for i in range(0, len(tableau), 2):
      if tableau[i] < epsilon or tableau[i] > x_max - epsilon :
        tableau[i] =  x_max /2 + random.uniform(-0.1, 0.1)

      if tableau[i+1] < epsilon or tableau[i+1] > y_max - epsilon : 
        tableau[i+1] = y_max /2 + random.uniform(-0.1, 0.1)
        
    return tableau
