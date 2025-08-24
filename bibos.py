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


def generation_grille_bibos(N, dimensions_carton, axe=[0,0,1], rayon_grand=0, rayon_petit=0.05, longueur=1.0, reglages=None):

    # Réglages par défaut
    reglages_par_defaut = {}
    if reglages is not None:
        reglages = reglages_par_defaut|reglages
    else:
        reglages = reglages_par_defaut

    direction_index = int(np.dot([0, 1, 2], axe))  # 0, 1 ou 2 selon l’axe donné
    ex, ey = None, None
    for e in np.eye(3):
        if np.dot(e, axe) == 0:
            if ex is None:
                ex = e
            else:
                ey = e
    d1 = int(np.dot([0, 1, 2], ex))
    d2 = int(np.dot([0, 1, 2], ey))

    L1 = dimensions_carton[d1]
    L2 = dimensions_carton[d2]

    def meilleur_decoupage(N, ratio): #ratio doit être >1
        meilleur_score = float('inf')
        p_min, q_min = None, None
        for q in range(1,N+1):
            p_1  = np.clip(math.ceil(N/q), 1,N)
            p_2 = np.clip(int(N/(q-1+1e-10)),1,N)
            
            if p_1<= p_2:
                for p in range(p_1, p_2+1):
                    d = abs(log(p / q / ratio))
                    if d < meilleur_score and (p-1)*q < N and p*(q-1)<N:
                        meilleur_score = d
                        p_min = p
                        q_min = q

            else :
                for p in range(p_2, p_1+1):
                    d = abs(log(p / q / ratio))
                    if d < meilleur_score and (p-1)*q < N and p*(q-1)<N:
                        meilleur_score = d
                        p_min = p
                        q_min = q


        return p_min, q_min

    if L2 > L1:
        ratio_carton = L2 / L1
        n_lignes, n_colonnes = meilleur_decoupage(N, ratio_carton)
    else:
        ratio_carton = L1 / L2
        n_colonnes, n_lignes = meilleur_decoupage(N, ratio_carton)
        
    # Coordonnées régulièrement espacées dans chaque direction
    x_coords = np.linspace(L1 / (2 * n_colonnes), L1 * (1 - 1 / (2 * n_colonnes)), n_colonnes)
    y_coords = np.linspace(L2 / (2 * n_lignes), L2 * (1 - 1 / (2 * n_lignes)), n_lignes)

    centres_grands = []
    orientations = []

    compteur_centres = 0
    for i in range(n_lignes-1):
        for j in range(n_colonnes):
            if compteur_centres < N:
                centre = np.zeros(3)
                centre[direction_index] = dimensions_carton[direction_index] / 2  # centré sur l’axe
                centre[d1] = x_coords[j]
                centre[d2] = y_coords[i]
                centres_grands.append(centre)
                orientations.append(np.pi / 4)
                compteur_centres += 1
            else:
                break

    # On complète la dernière ligne
    n_colonnes = N - compteur_centres
    if n_colonnes != 0 : 
        x_coords = np.linspace(L1 / (2 * n_colonnes), L1 * (1 - 1 / (2 * n_colonnes)), n_colonnes)
        for j in range(n_colonnes):
            if compteur_centres < N:
                    centre = np.zeros(3)
                    centre[direction_index] = dimensions_carton[direction_index] / 2  # centré sur l’axe
                    centre[d1] = x_coords[j]
                    centre[d2] = y_coords[n_lignes-1]
                    centres_grands.append(centre)
                    orientations.append(np.pi / 4)
                    compteur_centres += 1

    arrangement = {
        'nom': 'Arrangement en grille',
        'axe': axe,
        'nombre': N,
        'centres': np.array(centres_grands),
        'grand rayon': rayon_grand,
        'petit rayon': rayon_petit,
        'longueur': longueur,
        'orientations': np.array(orientations),
        'dimensions du carton': dimensions_carton,
        'origine du carton': [0, 0, 0],
        'arrange en grille' : True
    }

    generation_grille_bibos.valeurs_intermediaires = []
    return arrangement

def petits_centres(arrangement):
    axe  =arrangement['axe']
    ex = None
    ey = None
    for e in np.eye(3):
        if np.dot(e, axe) == 0:
            if ex is None:
                ex = e
            else:
                ey = e
    if ex is None or ey is None:
        print('axe : ', axe)
        raise ValueError("Axe invalide")
    centres_grands = np.array(arrangement['centres'])
    orientations = np.array(arrangement['orientations'])
    rayon_grand = arrangement['grand rayon']
    rayon_petit = arrangement['petit rayon']
    eloignement = rayon_grand + rayon_petit
    centres_petits = centres_grands + eloignement*np.array([ np.cos(orientation)*ex for orientation in orientations ] ) +eloignement*np.array([ np.sin(orientation)*ey for orientation in orientations ] )
    
    return centres_petits

def t_1_d_bibos(centres, orientations):
    n = centres.shape[0]
    t = np.zeros(4 * n)

    ind = 0
    for i in range(n):
        t[ind] = centres[i][0]
        t[ind + 1] = centres[i][1]
        t[ind + 2] = centres[i][2]
        ind = ind + 3
        t[3*n + i] = orientations[i]

    return t

def t_2_d_bibos(t):
    n = int(len(t) / 4)
    centres = np.zeros((n, 3))
    orientations = np.zeros(n)

    for i in range(n):
        centres[i][0] = t[3 * i]
        centres[i][1] = t[3 * i + 1]
        centres[i][2] = t[3 * i + 2 ]
        orientations[i] = t[3*n+i]
    return centres, orientations

def projection_bibos(t, axe):
    axe = np.array(axe)
    mod = np.dot([0, 1, 2], axe)
    n = int(len(t) / 4)
    t_proj = np.zeros(3 * n)

    ind = 0
    for i in range(n):
        for j in range(3):
            if j != mod:
                t_proj[ind] = t[3 * i + j]
                ind += 1
    for i in range(n):
      t_proj[ind]+= t[3*n+i]
      ind+=1

    return t_proj

def injection_bibos(t_proj, axe, valeur):
    axe = np.array(axe)
    mod = np.dot([0, 1, 2], axe)
    n = int(len(t_proj) / 3)
    t_full = np.zeros(4 * n)

    ind = 0
    for i in range(n):
        for j in range(3):
            if j == mod:
                t_full[3 * i + j] = valeur
            else:
                t_full[3 * i + j] = t_proj[ind]
                ind += 1
    
    for i in range(n):
      t_full[3*n+ i ] =t_proj[ind]
      ind += 1

    return t_full

def distance_entre_disques(centre_1, rayon_1, centre_2, rayon_2):
    """
    Calcule la distance entre deux disques dans le plan 2D.
    La distance est négative en cas de recouvrement.
    """
    distance_centres = np.linalg.norm(np.array(centre_1) - np.array(centre_2))
    return distance_centres - rayon_1 - rayon_2

def projet_orthogonal_au_plan(vecteur, axe_orthogonal_au_plan):
    """
    Projette un vecteur 3D dans le plan orthogonal à l'axe donné.
    """
    axe = np.array(axe_orthogonal_au_plan)
    return np.array(vecteur) - np.dot(vecteur, axe) * axe

def distance_deux_bibos(indice_bibo_1, indice_bibo_2, arrangement, reglages=None):
    """
    Calcule la distance minimale entre les deux bibos d'indices donnés.
    Chaque bibo est composé d'un grand cylindre et d'un petit cylindre accolé.
    Le calcul se fait dans le plan orthogonal à l'axe de rotation commun à tous les cylindres.
    """
    axe_commun = np.array(arrangement['axe'])
    rayon_grand_cylindre = arrangement['grand rayon']
    rayon_petit_cylindre = arrangement['petit rayon']
    ptits_centres = np.array(petits_centres(arrangement))

    centre_grand_cylindre_1 = np.array(arrangement['centres'][indice_bibo_1])
    centre_petit_cylindre_1 = ptits_centres[indice_bibo_1]

    centre_grand_cylindre_2 = np.array(arrangement['centres'][indice_bibo_2])
    centre_petit_cylindre_2 = ptits_centres[indice_bibo_2]

    # Projeter tous les centres dans le plan orthogonal à l'axe
    centre_grand_1_2d = projet_orthogonal_au_plan(centre_grand_cylindre_1, axe_commun)
    centre_petit_1_2d = projet_orthogonal_au_plan(centre_petit_cylindre_1, axe_commun)
    centre_grand_2_2d = projet_orthogonal_au_plan(centre_grand_cylindre_2, axe_commun)
    centre_petit_2_2d = projet_orthogonal_au_plan(centre_petit_cylindre_2, axe_commun)

    # Calcul des 4 distances entre les disques projetés
    distances = np.array([
        distance_entre_disques(centre_grand_1_2d, rayon_grand_cylindre, centre_grand_2_2d, rayon_grand_cylindre),
        distance_entre_disques(centre_grand_1_2d, rayon_grand_cylindre, centre_petit_2_2d, rayon_petit_cylindre),
        distance_entre_disques(centre_petit_1_2d, rayon_petit_cylindre, centre_grand_2_2d, rayon_grand_cylindre),
        distance_entre_disques(centre_petit_1_2d, rayon_petit_cylindre, centre_petit_2_2d, rayon_petit_cylindre)
    ])
    i = np.argmin(distances)
    feur_petits_centres = np.array([[False, False], [False, True], [True, False], [True, True]])#sert à savoir si la distance est entre les grand centres ou les petits. 
    indic_petits_centres = feur_petits_centres[i][True, False] #signifie que la distance minimale est entre le petit centre et le grand de j
    distance_minimale = min(distances)
    distance_deux_bibos.distances = distances
    return distance_minimale

def perte_chevauchement_2d_bibos(arrangement_bibo, reglages=None):
    reglages_par_defaut = {
        'penalisation chevauchement': puiss_8_tronque,
        'coefficient chevauchement': 100,
        'derivee au bord chevauchement': 1,
        'scale chevauchement': None,
        'portee penalite chevauchement': 0
    }

    if reglages is None:
        reglages = reglages_par_defaut
    else:
        reglages = reglages_par_defaut | reglages

    n_bibos = arrangement_bibo['centres'].shape[0]
    grand_rayon = arrangement_bibo['grand rayon']
    petit_rayon = arrangement_bibo['petit rayon']
    longueur = arrangement_bibo['longueur']
    axe = arrangement_bibo['axe']

    if reglages['scale chevauchement'] is None:
        reglages['scale chevauchement'] = grand_rayon

    reglages_fonction_de_penalisation = {
        'scale': reglages['scale chevauchement'],
        'derivee au bord': reglages['derivee au bord chevauchement'],
        'portee penalite': reglages['portee penalite chevauchement']
    }

    # Matrice des distances négatives (0 sinon)
    distances = np.zeros((n_bibos, n_bibos))
    somme_penalites = 0

    for i in range(n_bibos):
        for j in range(i + 1, n_bibos):
            distance = distance_deux_bibos(i, j, arrangement_bibo)
            if distance < 0:
                distances[i, j] = -distance
                somme_penalites += reglages['penalisation chevauchement'](
                    -distance, reglages_fonction_de_penalisation
                )

    perte_chevauchement_2d_bibos.distances = distances

    return reglages['coefficient chevauchement'] * somme_penalites / n_bibos

def perte_depassement_2d_bibos(arrangement, reglages=None):
    # Création de l'arrangement des petits cylindres
    arrangement_petits = {
        'centres': petits_centres(arrangement),
        'rayon': arrangement['petit rayon'],
        'longueur': arrangement['longueur'],
        'axe': arrangement['axe'],
        'dimensions du carton': arrangement['dimensions du carton'], 
        'origine du carton' : [0,0,0]
    }

    # Création de l'arrangement des grands cylindres
    arrangement_grands = {
        'centres': arrangement['centres'],
        'rayon': arrangement['grand rayon'],
        'longueur': arrangement['longueur'],
        'axe': arrangement['axe'],
        'dimensions du carton': arrangement['dimensions du carton'], 
        'origine du carton' : [0,0,0]
    }

    # Évaluation des pénalités de dépassement pour chaque arrangement
    perte_grands = perte_depassement_2d(arrangement_grands, reglages)
    perte_petits = perte_depassement_2d(arrangement_petits, reglages)

    # Somme des deux
    return perte_grands + perte_petits

def perte_densite_2d_bibos(arrangement, reglages = None):
  reglages_par_defaut = {'penalisation densite' : id, 'coefficient densite' : 0}
  if reglages is None:
    reglages = reglages_par_defaut
  else:
    reglages = reglages_par_defaut | reglages
  centres = np.array(arrangement['centres'])
  dimensions = arrangement['dimensions du carton']
  n = np.shape(centres)[0]
  rayon = arrangement['grand rayon']
  d = 0
  for i in range(n):
    for j in range(i+1, n):
      d += np.linalg.norm(centres[i] - centres[j]) - 2 * rayon
  distance_moyenne = d / (n * (n - 1 +0.01) / 2)
  distance_moyenne /= np.mean(dimensions)
  return reglages['coefficient densite'] * reglages['penalisation densite'](distance_moyenne)

def perte_alignement_bibos(arrangement, reglages = None):
  reglages_par_defaut = {
    'Ponderation' : exp_neg,
    'penalisation alignement' : mesure_dispersion,
    'portee alignement' : 1,
    'coefficient alignement' : 0
  }
  if reglages is None:
    reglages = reglages_par_defaut
  else:
    reglages = reglages_par_defaut | reglages

  centres = arrangement['centres']
  axe = arrangement['axe']
  rayon = arrangement['grand rayon']
  ponderation = reglages['Ponderation']
  penalisation_alignement = reglages['penalisation alignement']
  coeff_align = reglages['coefficient alignement']

  n_centres = len(centres)
  angles_ponderations = []  # ((angle, ponderation) for angle in angles)
  reglages_int = {'scale' : reglages['portee alignement'] * rayon}

  for i in range(n_centres):
    for j in range(i + 1, n_centres):
      distance, angle = calculs_spatiaux(centres[i], centres[j], axe)  # angle entre 0 et pi/2
      pond = ponderation(distance / (rayon+1e-8), reglages = reglages_int)
      angles_ponderations = ajout_angle(angles_ponderations, angle, pond)

  perte = penalisation_alignement(angles_ponderations, reglages)
  return coeff_align * perte

def perte_total_2d_bibos(arrangement, reglages = None):
  reglages_par_defaut = {}
  if reglages is not None:
    reglages = reglages_par_defaut | reglages
  else:
    reglages = reglages_par_defaut

  s = 0
  a = perte_chevauchement_2d_bibos(arrangement, reglages)
  b = perte_depassement_2d_bibos(arrangement, reglages)
  c = perte_densite_2d_bibos(arrangement, reglages)
  d = perte_alignement_bibos(arrangement, reglages)
  s = a + b + c + d

  perte_total_2d_bibos.perte_de_chevauchement = a
  perte_total_2d_bibos.perte_de_depassement = b
  perte_total_2d_bibos.perte_de_densite = c
  perte_total_2d_bibos.perte_de_alignement = d

  return s

def perte_totale_1d_2d_bibos(centres_orientations_1d, grand_rayon, petit_rayon, longueur, dimensions_carton, axe , reglages = None):
  '''
  Prend un tableau 1d de centres de bibos et renvoie une perte totale.
  '''
  reglages_par_defaut = {}
  if reglages is not None:
    reglages = reglages_par_defaut | reglages
  else:
    reglages = reglages_par_defaut

  centres_cylindres, orientations = t_2_d_bibos(centres_orientations_1d)
  arrangement = {
    'centres': centres_cylindres,
    'orientations': orientations,
    'grand rayon': grand_rayon,
    'petit rayon': petit_rayon,
    'longueur': longueur,
    'axe': axe,
    'dimensions du carton': dimensions_carton  }

  perte = perte_total_2d_bibos(arrangement, reglages)

  perte_totale_1d_2d_bibos.perte_de_chevauchement = perte_total_2d_bibos.perte_de_chevauchement
  perte_totale_1d_2d_bibos.perte_de_depassement = perte_total_2d_bibos.perte_de_depassement
  perte_totale_1d_2d_bibos.perte_de_densite = perte_total_2d_bibos.perte_de_densite
  perte_totale_1d_2d_bibos.perte_de_alignement = perte_total_2d_bibos.perte_de_alignement

  return perte

def legitime_chevauchements_bibos(arrangement, reglages=None):
    """
    Détermine s'il y a des chevauchements entre bibos dans un arrangement donné.

    - arrangement : dictionnaire décrivant les bibos (centres, rayons, longueurs, etc.)
    - reglages : dictionnaire d'options (notamment la distance utilisée, par défaut 'distance_deux_bibos')

    Retourne :
    - est_legitime_chevauchement : booléen indiquant si aucun chevauchement n'est détecté.
    """
    reglages_par_defaut = {'distance entre bibos': distance_deux_bibos}
    if reglages is None:
        reglages = reglages_par_defaut
    else:
        reglages = reglages_par_defaut | reglages

    distance = reglages['distance entre bibos']
    n_bibos = len(arrangement['centres'])
    est_legitime_chevauchement = True
    for ind_1 in range(n_bibos - 1):
        for ind_2 in range(ind_1 + 1, n_bibos):
            dist = distance(ind_1, ind_2, arrangement, reglages)
            if dist <= 0:
                est_legitime_chevauchement = False
    return est_legitime_chevauchement

def legitime_depassement_bibos(arrangement, reglages=None):
    """
    Vérifie si les bibos dépassent du carton, en testant séparément les grands et les petits cylindres.

    - arrangement : dictionnaire contenant les centres, rayons, longueurs, axe, origine et dimensions du carton.
    - reglages : options passées à `legitime_depassement`

    Retourne :
    - booléen : True si aucun dépassement (grands ni petits cylindres)
    """
    # Extraction des éléments communs
    dimensions = arrangement['dimensions du carton']
    axe = arrangement['axe']
    grands_centres = arrangement['centres']
    ptits_centres = petits_centres(arrangement)
    longueur = arrangement['longueur']
    g_rayon = arrangement['grand rayon']
    p_rayon = arrangement['petit rayon']
    dimensions_carton = arrangement['dimensions du carton']
    origine_carton =[0,0,0]

    # Test sur les deux parties
    leg_grand = legitime_depassement(grands_centres,  longueur, g_rayon, dimensions_carton , axe, origine_carton , reglages )
    leg_petit = legitime_depassement(ptits_centres,  longueur, p_rayon, dimensions_carton , axe, origine_carton , reglages )

    # Transfert des attributs utiles pour inspection (comme dans ta version précédente)
    legitime_depassement_bibos.legitime_grand = leg_grand
    legitime_depassement_bibos.legitime_petit = leg_petit

    legitime_depassement_bibos.points_extremes = []
    return leg_grand and leg_petit

def est_legitime_bibos(arrangement, reglages=None):
    """
    Vérifie la légitimité complète d’un arrangement de bibos (pas de chevauchement ni de dépassement).

    - arrangement : dictionnaire décrivant les bibos
    - reglages : dictionnaire d'options

    Retourne :
    - booléen indiquant si l’arrangement est valide
    """
    rayon = arrangement['grand rayon']
    longueur = arrangement['longueur']
    reglages_defaut = {
        'fit_carton': False,
        'erreur acceptee depassement': 0.001 * rayon,
        'distance entre bibos': distance_deux_bibos
    }
    if reglages is None:
        reglages = reglages_defaut
    else:
        reglages = reglages_defaut | reglages

    leg_dep = legitime_depassement_bibos(arrangement, reglages)
    leg_chev = legitime_chevauchements_bibos(arrangement, reglages)

    return leg_dep and leg_chev

def generation_aleatoire_bibos(N, dimensions_carton, axe, rayon_grand=0, rayon_petit = 0.05, longueur=1.0, reglages=None):
    """
    Génère un arrangement aléatoire de N bibos dans un carton de dimensions données,
    avec un axe donné (vecteur directionnel).
    
    Chaque bibo est composé de deux cylindres accollés :
    - grand cylindre de rayon 'rayon_grand'
    - petit cylindre de rayon 'rayon_grand * facteur_petit'
    - longueur commune 'longueur'
    """
    
    # Réglages par défaut
    reglages_par_defaut = {}
    if reglages is not None:
        reglages = {**reglages_par_defaut, **reglages}
    else:
        reglages = reglages_par_defaut
    

    grand_rayon = rayon_grand
    petit_rayon = rayon_petit
    centres_grands = []
    orientations = []
    direction_index = np.dot( [0,1,2], axe)
    ex= None
    ey = None
    for e in np.eye(3):
        if np.dot(e, axe) == 0:
            if ex is None:
                ex = e
            else:
                ey = e


    
    for _ in range(N):
        centre_grand = np.zeros(3)
        for i in range(3):
            if i == direction_index:
                centre_grand[i] = dimensions_carton[i] / 2  # centré dans la direction de l'axe
            else:
                centre_grand[i] = np.random.uniform(dimensions_carton[i]/4, 3*dimensions_carton[i]/4)
        
        
        angle = np.random.uniform(0, 2 * np.pi)
        
        centres_grands.append(centre_grand)
        orientations.append(angle)

    arrangement = {
        'nom': 'Arrangement aléatoire de bibos',
        'axe': axe,
        'nombre': N,
        'centres': np.array(centres_grands),
        'grand rayon': rayon_grand,
        'petit rayon': petit_rayon,
        'longueur': longueur,
        'orientations': np.array(orientations),
        'dimensions du carton': dimensions_carton,
        'origine du carton': [0, 0, 0],
    }

    generation_aleatoire_bibos.valeurs_intermediaires = []
    return arrangement

def legitime_bien_aligne_bibos(arrangement, reglages= None):
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
        est_legitime_bibos(arrangement, reglages)
        and perte_alignement_bibos(arrangement, reglages) <= reglages['seuil']
    )

def arranger_v2_bibos(arrangement, reglages=None):

 ### PREPARATION

  ## définition et extraction des réglages

    reglages_par_defaut = {
        'solveur': 'BFGS',
        'options': {'maxiter': 200, 'gtol': 1e-8, 'eps': 1e-8},
        'pellicule': 1.15,
        'iterations par tour': 5,
        'borne sup perte': float('inf'),
        'n_tours_admissibles': 30,
        'stagnation acceptee': 10,
        'ecart stagnation': 0.05,
        'stagnation autour de 0': 30,
        'borne stagnation nulle': 1,
        'borne stagnation nulle acceleree':20,
        'version acceleree': False
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
    n_stag_0 = reglages['stagnation autour de 0']
    ecart_stag = reglages['ecart stagnation']
    borne_stag_nulle = reglages['borne stagnation nulle']
    borne_stag_0_acc = reglages['borne stagnation nulle acceleree']
    version_acceleree = reglages['version acceleree']

  ## extraction des données

    arrangement_interne = copy.deepcopy(arrangement)
    centres_2d = arrangement_interne['centres']
    orientations = arrangement_interne['orientations']
    grand_rayon = arrangement_interne['grand rayon']
    petit_rayon = arrangement_interne['petit rayon']
    longueur = arrangement_interne['longueur']
    axe = arrangement_interne['axe']
    dimensions_carton = arrangement_interne['dimensions du carton']

  ## préparation des objets

    direction_longueur = np.dot([0, 1, 2], axe)
    coord_commune = longueur / 2
    centres_orientations_1d = t_1_d_bibos(centres_2d, orientations)
    centres_orientations_1d_projete = projection_bibos(centres_orientations_1d, axe)
    stag_0 = None
    stag = None

  ## définition de la fonction à minimiser

    def a_minimiser(centres_orientations_1d_projete):
        centres_orientations_reinjectes = injection_bibos(centres_orientations_1d_projete, axe, coord_commune)
        return perte_totale_1d_2d_bibos(centres_orientations_reinjectes, pellicule * grand_rayon, pellicule*petit_rayon, longueur, dimensions_carton, axe, reglages)

  ## communication avec ICC

    valeurs_intermediaires = []
    compteur_iteration_minimiseur = 0
    pertes_recentres = []

    def store_points(xk):
        nonlocal compteur_iteration_minimiseur, valeurs_intermediaires, pertes_recentres
        centres_orientations = injection_bibos(xk, axe, coord_commune)
        centres, orientations = t_2_d_bibos(centres_orientations)
        perte = perte_total_2d_bibos({
            'centres': centres,
            'orientations': orientations,
            'grand rayon': grand_rayon,
            'petit rayon': petit_rayon,
            'longueur': longueur,
            'axe': axe,
            'dimensions du carton': dimensions_carton
        }, reglages)
        valeurs_intermediaires.append({
            'centres': np.copy(centres),
            'orientations': np.copy(orientations),
            'rayon': grand_rayon,
            'code couleur': 0,
            'texte': 'En cours de rangement',
            'perte': perte,
            'iteration du minimiseur': compteur_iteration_minimiseur
        })
        pertes_recentres.append(perte)
        if len(pertes_recentres) > max(n_stag, n_stag_0):
            pertes_recentres.pop(0)
        compteur_iteration_minimiseur += 1

 ### MINIMISATION EN BOUCLE

    converged = False
    max_tours = options['maxiter'] // iter_par_tour
    centres_orientations_1d_projete_temporaire = centres_orientations_1d_projete

    for tour in range(max_tours):

        sortie_brute = minimize(a_minimiser, centres_orientations_1d_projete_temporaire,
                                method=methode, callback=store_points,
                                options={**options, 'maxiter': iter_par_tour})
        x = sortie_brute.x
        centres_orientations_1d_projete_temporaire = x
        centres_2d_temporaires, orientations_temporaires  = t_2_d_bibos(injection_bibos(x, axe, coord_commune))
        arrangement_temporaire = copy.deepcopy(arrangement_interne)
        arrangement_temporaire['centres'] = centres_2d_temporaires
        arrangement_temporaire['orientations'] = orientations_temporaires

        perte_actuelle = perte_total_2d_bibos(arrangement_temporaire, reglages)

        if legitime_bien_aligne_bibos(arrangement_temporaire, reglages):
            converged = True
            print('est legitime')
            arrangement_sortie = copy.deepcopy(arrangement_temporaire)
            break

        if perte_actuelle >= borne_perte and tour <= n_tours_admissibles:
            converged = True
            arrangement_sortie = copy.deepcopy(arrangement_temporaire)
            break

        if len(pertes_recentres) >= max(n_stag, n_stag_0):

            if len(pertes_recentres) >= n_stag_0:
                dernieres_pour_stag_0 = pertes_recentres[-n_stag_0:]
                if all(abs(p) <= borne_stag_nulle for p in dernieres_pour_stag_0) or (version_acceleree and all(abs(p) <= borne_stag_0_acc for p in dernieres_pour_stag_0)):
                    print('Stagnation autour de 0 détectée')
                    stag_0 = dernieres_pour_stag_0[-1]
                    converged = True
                    arrangement_sortie = copy.deepcopy(arrangement_temporaire)
                    break

            dernieres_pour_stag = pertes_recentres[-n_stag:]
            moyenne = np.mean(dernieres_pour_stag)
            if all(abs(p - moyenne) <= ecart_stag * moyenne for p in dernieres_pour_stag):
                print('Stagnation détectée')
                converged = True
                arrangement_sortie = copy.deepcopy(arrangement_temporaire)
                stag = moyenne
                break

 ### SORTIE

    if not converged:
        centres_orientations_proposes_1d = injection_bibos(x, axe=axe, valeur=coord_commune)
        centres_proposes_2d, orientations_proposees = t_2_d_bibos(centres_orientations_proposes_1d)
        arrangement_sortie = copy.deepcopy(arrangement_interne)
        arrangement_sortie['centres'] = centres_proposes_2d
        arrangement_sortie['orientations'] = orientations_proposees

    arranger_v2_bibos.valeurs_intermediaires = valeurs_intermediaires
    arranger_v2_bibos.message = sortie_brute.message
    arranger_v2_bibos.stag_0_detectee = stag_0
    arranger_v2_bibos.stag = stag

    return arrangement_sortie

def film_bibos(valeurs_intermediaires, arrangement, nom_fichier="animation_compressed.gif"):
    def projeter_sur_plan(point, axe):
        axes_du_plan = []
        iterateur = np.eye(3)
        for e in iterateur:
            if not np.array_equal(e, axe):
                axes_du_plan.append(e)
        projete = np.zeros(2)
        projete[0] = np.dot(point, axes_du_plan[0])
        projete[1] = np.dot(point, axes_du_plan[1])
        return projete

    def ajouter_carton(ax, dimensions, axe, couleur='green', alpha=0.3):
        dimensions_2d = projeter_sur_plan(dimensions, axe)
        largeur = dimensions_2d[0]
        hauteur = dimensions_2d[1]
        coin = [0, 0]
        rect = Rectangle(coin, largeur, hauteur, color=couleur, alpha=alpha)
        ax.add_patch(rect)
        return rect

    def ajouter_disques(ax, centres, rayon, axe, couleur='blue', alpha=0.6):
        plots = []
        for centre in centres:
            centre_2d = projeter_sur_plan(centre, axe)
            circ = Circle(centre_2d, rayon, color=couleur, alpha=alpha)
            ax.add_patch(circ)
            plots.append(circ)
        return plots

    # === Extraction des paramètres ===
    dimensions_carton = np.array(arrangement['dimensions du carton'])
    axe = arrangement['axe']
    ex = None
    ey = None
    for e in np.eye(3):
        if np.dot(e, axe) == 0:
            if ex is None:
                ex = e
            else:
                ey = e
    norm = 0.5 * np.dot(dimensions_carton, np.array([1, 1, 1]) - np.array(axe))
    frames = valeurs_intermediaires

    # === Création de la figure compressée ===
    fig, ax = plt.subplots(figsize=(6, 4), dpi=80)
    ax.set_aspect('equal')

    # Détermination des limites automatiquement à partir du carton
    if axe[0] == 1:
        ax.set_xlim(0, dimensions_carton[1] / norm)
        ax.set_ylim(0, dimensions_carton[2] / norm)
    elif axe[1] == 1:
        ax.set_xlim(0, dimensions_carton[0] / norm)
        ax.set_ylim(0, dimensions_carton[2] / norm)
    else:
        ax.set_xlim(0, dimensions_carton[0] / norm)
        ax.set_ylim(0, dimensions_carton[1] / norm)

    # Textes d'information
    info_text = ax.text(0.02, 0.98, '', transform=ax.transAxes, va='top', ha='left',
                        bbox=dict(facecolor='white', alpha=0.8, edgecolor='gray'))
    info_text_perte = ax.text(0.02, 0.88, '', transform=ax.transAxes, va='top', ha='left',
                              bbox=dict(facecolor='white', alpha=0.8, edgecolor='gray'))
    info_text_iter = ax.text(0.02, 0.78, '', transform=ax.transAxes, va='top', ha='left',
                             bbox=dict(facecolor='white', alpha=0.8, edgecolor='gray'))
    info_text_rayon = ax.text(0.02, 0.68, '', transform=ax.transAxes, va='top', ha='left',
                              bbox=dict(facecolor='white', alpha=0.8, edgecolor='gray'))

    patch_carton = ajouter_carton(ax, dimensions_carton / norm, axe)
    cercle_plots = []

    def init():
        nonlocal cercle_plots
        for patch in cercle_plots:
            patch.remove()
        cercle_plots = []
        return []

    def update(frame):
        nonlocal cercle_plots
        for patch in cercle_plots:
            patch.remove()
        cercle_plots = []

        grand_rayon = frame['rayon']
        petit_rayon = arrangement['petit rayon']
        eloignement = petit_rayon + grand_rayon
        orientations = np.array(frame['orientations'])
        centres_grands = np.array(frame['centres'])
        arrg_factice = {'axe': axe, 'orientations': orientations, 'centres': centres_grands,  'grand rayon': grand_rayon, 'petit rayon': petit_rayon}
        centres_petits = petits_centres(arrg_factice)

        
        texte = frame.get('texte', '')
        perte = frame.get('perte', 0)
        couleur = frame.get('code couleur', 0)
        iteration = frame.get('iteration du minimiseur', 'N/A')

        cercle_plots += ajouter_disques(ax, centres_grands, grand_rayon, axe, couleur='blue')
        cercle_plots += ajouter_disques(ax, centres_petits, petit_rayon, axe, couleur='orange')

        if couleur == 1:
            patch_carton.set_color('red')
        elif couleur == 2:
            patch_carton.set_color('green')
        elif couleur == 5:
            patch_carton.set_color('magenta')
        else:
            patch_carton.set_color('blue')

        info_text.set_text(f"Étape : {texte}")
        info_text_perte.set_text(f"Perte : {perte:.2f}")
        info_text_iter.set_text(f"Itération : {iteration}")
        info_text_rayon.set_text(f"Rayons : {grand_rayon:.2f} / {petit_rayon:.2f}")

        return cercle_plots + [info_text, info_text_perte, info_text_iter, patch_carton]

    ani = animation.FuncAnimation(fig, update, frames=frames, init_func=init, blit=False)

    # === Sauvegarde en GIF compressé ===
    writer = PillowWriter(fps=4)
    ani.save(nom_fichier, writer=writer)

    print(f"✅ Animation GIF enregistrée sous '{nom_fichier}'")

def tracer_arrangement_bibos(arrangement):
    print('Nom : ', arrangement['nom'])
    print('Grand Rayon : ', arrangement['grand rayon'])
    print('Longueur : ', arrangement['longueur'])
    print('axe : ', arrangement['axe'])
    if est_legitime_bibos(arrangement):
        print('Cet arrangement de bibos est légitime')
    else :
        print('Cet arrangement de bibos n\'est pas légitime')
        print('Perte : ', perte_total_2d_bibos(arrangement))
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
        largeur = dimensions_2d[0]
        hauteur = dimensions_2d[1]
        rect = Rectangle((0, 0), largeur, hauteur, color=couleur, alpha=alpha)
        ax.add_patch(rect)

    def ajouter_disques(ax, centres, rayon, axe, couleur='blue', alpha=0.6):
        for centre in centres:
            centre_2d = projeter_sur_plan(centre, axe)
            circ = Circle(centre_2d, rayon, color=couleur, alpha=alpha)
            ax.add_patch(circ)

    # === Extraction des données ===
    dimensions_carton = arrangement['dimensions du carton']
    axe = arrangement['axe']

    centres_grands = np.array(arrangement['centres'])
    orientations = np.array(arrangement['orientations'])
    rayon_grand = arrangement['grand rayon']
    rayon_petit = arrangement['petit rayon']
    centres_petits = petits_centres(arrangement)
    

    # === Création de la figure ===
    fig, ax = plt.subplots(figsize=(6, 4), dpi=80)
    ax.set_aspect('equal')

    # Limites du graphique
    if axe[0] == 1:
        ax.set_xlim(0, dimensions_carton[1] )
        ax.set_ylim(0, dimensions_carton[2] )
    elif axe[1] == 1:
        ax.set_xlim(0, dimensions_carton[0] )
        ax.set_ylim(0, dimensions_carton[2] )
    else:
        ax.set_xlim(0, dimensions_carton[0])
        ax.set_ylim(0, dimensions_carton[1])

    ajouter_carton(ax, dimensions_carton , axe)
    ajouter_disques(ax, centres_grands, rayon_grand, axe, couleur='blue', alpha=0.6)
    ajouter_disques(ax, centres_petits, rayon_petit, axe, couleur='orange', alpha=0.6)

    plt.title(arrangement['nom'])
    plt.tight_layout()
    plt.show()

def ranger_bibos(arrangement, reglages=None):
    reglages_par_defaut = {'fonction arrangement' : arranger_v2_bibos}
    if reglages is None:
        reglages = reglages_par_defaut
    else:
        reglages = reglages_par_defaut | reglages
    fonction_arranger = reglages['fonction arrangement']
    arrangement_sortie = copy.deepcopy(arrangement)
    arrangement_sortie = fonction_arranger(arrangement_sortie, reglages)
    ranger_bibos.valeurs_intermediaires = fonction_arranger.valeurs_intermediaires
    ranger_bibos.stag_0_detectee = fonction_arranger.stag_0_detectee
    ranger_bibos.stag_detectee = fonction_arranger.stag
    return arrangement_sortie

def perte_inch_v1_bibos(arrangement, reglages=None):
    grands_centres = arrangement["centres"]
    axe = arrangement["axe"]
    ptits_centres = petits_centres(arrangement)
    orientations = arrangement["orientations"]
    n = len(grands_centres)

    min_dist = float('inf')
    centre_i_min = [] #liste des centres "que l'on est en train d'explorer" les plus petits
    est_petit_centre = [] #liste de booleens. Si true, alors le centre concerné est le petit centre
    centre_ext_min = [] #liste des centres le plus proche
    indices_i_min = [] #liste des indices des bibos concernés
    post_min_dist = float('inf')
    indicatrice_petits_centres = np.array([[False, False], [False, True], [True, False], [True, True]])
    eloignement  = 0.5* (arrangement["petit rayon"] + arrangement["grand rayon"] )

    for i in range(n):
        for j in range(n):
            distance_deux_bibos(i, j, arrangement)
            distances = distance_deux_bibos.distances
            for k,dist in enumerate(distances):
                if dist < min_dist: #on a trouvé une nouvelle plus petite distance. On remet tout à jour
                    post_min_dist = min_dist
                    min_dist = dist
                    est_petit_centre = [indicatrice_petits_centres[k][0]] #on enregistre si elle vient du petit centre
                    indices_i_min = [i]
                   # on met à jour les centres
                    if indicatrice_petits_centres[k][0] :
                        centre_i_min = [ptits_centres[i]]
                    else:
                        centre_i_min = [grands_centres[i]]

                    if indicatrice_petits_centres[k][1] :
                        centre_ext_min = [ptits_centres[j]]
                    else:
                        centre_ext_min = [grands_centres[j]]

                elif dist == min_dist: #on a trouvé une autre plus petite distance. On l'ajoute 
                    est_petit_centre.append(indicatrice_petits_centres[k][0]) #on enregistre si elle vient du petit centre
                    indices_i_min.append(i)
                   # on met à jour les centres
                    if indicatrice_petits_centres[k][0] :
                        centre_i_min.append(ptits_centres[i])
                    else:
                        centre_i_min.append(grands_centres[i])

                    if indicatrice_petits_centres[k][1] :
                        centre_ext_min.append(ptits_centres[j])
                    else:
                        centre_ext_min.append(grands_centres[j])

                elif dist < post_min_dist:
                    post_min_dist = dist

    #mintenant on va chercher à calculer le gradient. Sa norme sera proportionnelle à l'écart entre la distance minimale et la distance suivante
    gradient = np.zeros((3*n))
    k= len(indices_i_min)
    coef = abs(min_dist - post_min_dist)
    for l in range(k):
        ind_l = indices_i_min[l]
        centre_int_l = projeter_sur_plan(centre_i_min[l], axe)
        centre_ext_l = projeter_sur_plan(centre_ext_min[l], axe)
        est_petit_l = est_petit_centre[l]

        if est_petit_l:
            e_l = np.array(centre_ext_l)-np.array(centre_int_l)
            theta = orientations[ind_l]
            e_theta = -sin(theta)*np.array([1,0]) + cos(theta)*np.array([0,1 ])
            gradient[3*ind_l] += np.dot(e_theta,e_l)*eloignement *coef

        else:
            e_l = np.array(centre_int_l)-np.array(centre_ext_l)
            e_x = np.array([1,0])
            e_y = np.array([0,1])
            gradient[2*ind_l] += np.dot(e_x,e_l) * coef
            gradient[2*ind_l+1] += np.dot(e_y,e_l) *coef
    return( - min_dist, gradient)

def ajuster_rayon_bibos(arrangement, reglages=None):
    reglages_par_defaut = {
        'precision': 0.000001}
    if reglages is None :
        reglages = reglages_par_defaut
    else :
        reglages = reglages_par_defaut | reglages
    # Copie pour ne pas modifier l'original
    arr_test = copy.deepcopy(arrangement)

    # Rayon minimal (toujours légitime)
    r_min = 0.0

    # Rayon maximal initial : on prend la plus petite dimension du carton / 2
    dims = arrangement['dimensions du carton']
    r_max = max(dims) / 2

    reglages_par_defaut = {
       }
    if reglages is None:
        reglages = reglages_par_defaut
    else :
        reglages = reglages_par_defaut | reglages
    precision = reglages['precision']

    # Dichotomie
    while r_max - r_min > precision:
        r_mid = (r_min + r_max) / 2
        arr_test['grand rayon'] = r_mid
        if est_legitime_bibos(arr_test, reglages):
            r_min = r_mid  # C'est valide → on peut essayer plus grand
        else:
            r_max = r_mid  # Trop grand → on réduit

    arr_test['grand rayon'] = r_min
    return arr_test





