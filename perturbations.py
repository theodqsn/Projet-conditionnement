

import copy
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math
from math import *
import scipy

from arrangement_legitimite import *
from outils_voronoi import *
from outils_courants import *
from outils_dessin import *
from pertes_2d import *


def inv_sigmoide(x):
  return 1-1 / (1 + np.exp(-x))

def inv_gauss(x) :
  if x <-0.5 or x>0.5:
    return 40
  else : 
    u=scipy.stats.norm.pdf(x, scale = 1/6)
  return 1/u

def perturbation_aleatoire(arrangement, reglages = None):

  #ici, on gère les reglages
  reglages_defauts = {'portee_perturbation' : 0.25* np.max(arrangement['dimensions du carton'] )}
  if reglages is None:
     reglages_int = reglages_defauts
  else:
    reglages_int = reglages_defauts| reglages
  portee_perturbation = reglages_int['portee_perturbation']

  centres = arrangement['centres']
  n_cyl = np.shape(centres)[0]
  ind_centre = np.random.randint(0,n_cyl)
  centre = centres[ind_centre]

  r = np.random.uniform(0,1)*portee_perturbation #on choisit aleatoirement un vecteur (R, \theta, \phi pour des coordonnées sphériques)(si on choisit un vecteur de R^3 au hasard et qu'on le normalise, on perd l'uniformité)
  theta = np.random.uniform(0,2*np.pi)
  phi = np.random.uniform(0,np.pi)
  pas = np.array([r*cos(theta), r*sin(theta), r*cos(phi)])

  centres = centres
  centres[ind_centre] = centres[ind_centre] + pas
  perturbation_aleatoire.cylindre_perturbe = ind_centre
  return centres

def perturbation_tatonnement(arrangement, reglages=None ):
  
  #ici, on gère les reglages
  reglages_defauts = {'portee perturbation' : 0.25*np.max(arrangement['dimensions du carton'] ), 'fonction de perte' :  perte_total_2d, 'nombre essais tatonnement' : 10, 'fonction inversion' : inv_sigmoide }
  if reglages is None:
     reglages_int=reglages_defauts
  else:
    reglages = reglages_defauts | reglages
  portee_perturbation = reglages_int['portee perturbation']
  fonction_perte = reglages_int['fonction de perte']
  n_essais = reglages_int['nombre essais tatonnement']
  fonction_inv = reglages_int['fonction inversion']
  echelle =None
  translation = None
  
  arrg = arrangement.copy()
  centres = arrg['centres']
  n_cyl = np.shape(centres)[0]
  ind_centre = np.random.randint(0,n_cyl)
  tests= np.zeros((n_essais, n_cyl, 3))



    #pour chaque essai, on choisit aleatoirement un vecteur (R, \theta, \phi pour des coordonnées sphériques)(si on choisit un vecteur de R^3 au hasard et qu'on le normalise, on perd l'uniformité)
  for i in range(n_essais):
    r = np.random.uniform(0,1)*portee_perturbation
    theta = np.random.uniform(0,2*np.pi)
    phi = np.random.uniform(0,np.pi)
    pas = np.array([r*cos(theta), r*sin(theta), r*cos(phi)])
    tests[i] = centres
    tests[i][ind_centre] = centres[ind_centre]+pas

  #pour chacun des essais, on clacule la perte si l'on accepte cet essai
  pertes = np.zeros(n_essais)
  for i in range(n_essais):
    arrangement_test = arrg.copy()
    arrangement_test['centres'] = tests[i]
    pertes[i] = fonction_perte(arrangement_test, reglages)

  #on renormalise les valeurs des pertes afin qu'elles soient pertinentes avec la fonction qui va les transformer en probabilités
  if echelle == None:
    echelle = np.sqrt(np.var(pertes))
  if translation == None:
    translation = np.mean(pertes)
  pertes = (pertes - translation)/echelle
  probas = np.zeros(n_essais)

  # On transforme les valeurs de perte en probabilité grâce à la fonction qu'on a choisie (par défaut, l'opposée d'une sigmoïde)
  for i in range(n_essais):
    probas[i] = fonction_inv(pertes[i])
  probas = probas/np.sum(probas)

  # On tire aléatoirement l'essai que l'on va choisir de garder
  ind_aleat  = np.random.choice(n_essais, p = probas)

  centres_p = centres.copy()
  centres_p[ind_centre] = tests[ind_aleat][ind_centre]
  perturbation_tatonnement.cylindre_perturbe = ind_centre
  perturbation_tatonnement.test_choisi = ind_aleat
  perturbation_tatonnement.probas = probas
  perturbation_tatonnement.tests = tests
  return centres_p

def perturbation_voronoi(arrangement, reglages = None ):

  centres =np.array(arrangement['centres']).copy()
  n_cyl = np.shape(centres)[0]
  perte_tests= np.zeros(n_cyl)


  #ce morceaux sert à gérer les reglages de la fonction
  reglages_defauts = {'fonction_perte' : perte_total_2d,  'fonction_inv' : inv_sigmoide, 'echelle' : None, 'translation' : None, 'reglages_pertes' : None, 'coeff_exploration' : 1/3, 'coeff_trou' : 1}
  if reglages is None:
     reglages_int=reglages_defauts
  else:
    reglages_int = reglages_defauts | reglages
  fonction_perte = reglages_int['fonction_perte']
  fonction_inv = reglages_int['fonction_inv']
  echelle = reglages_int['echelle']
  translation = reglages_int['translation']
  reglages_pertes = reglages_int['reglages_pertes']
  coeff_exploration = reglages_int['coeff_exploration']
  coeff_trou = reglages_int['coeff_trou']



  #on identifie le plus gros trou de Voronoi
  barycentre = plus_gros_trou(arrangement, reglages)

  #on test les cylindres
  for i in range(n_cyl):
    centres_test = centres.copy()
    centres_test[i] = barycentre
    arrangement_test = arrangement.copy()
    arrangement_test['centres'] = centres_test
    perte_tests[i] = fonction_perte(arrangement_test, reglages_pertes)
    max= np.max(perte_tests)
    perte_tests = perte_tests/max
    for i in range(n_cyl):
      perte_tests[i] = -(1/perte_tests[i])



  #on renormalise les valeurs des pertes afin qu'elles soient pertinentes avec la fonction qui va les transformer en probabilités
  #utiliser les medianes va permettre de tenir compte des valeurs extrêmes
  if echelle == None:
    sup = np.max(perte_tests)
    sup_moins = np.max(perte_tests[perte_tests !=sup])
    med_sup = 0.5* (sup+sup_moins)

    inf = np.min(perte_tests)
    inf_plus = np.min(perte_tests[perte_tests !=inf])
    med_inf = 0.5* (inf+inf_plus)

    echelle = med_sup-med_inf
  if translation == None:
    translation = np.median(perte_tests)
  perte_tests = 1/coeff_exploration*(perte_tests - translation)/echelle #coeff_exploration sert à plus ou moins pénaliser assez fort la proba de deplacer un cylindre qui induit une faible perte (et à favoriser les fortes)

  #on calcule les probas et on renormalise
  probas = np.zeros(n_cyl)
  for i in range(n_cyl):
    probas[i] = fonction_inv(perte_tests[i])*vers_le_centre(centres[i], axe, dimensions_du_carton)
  probas = probas/np.sum(probas)

  # On tire aléatoirement l'essai que l'on va choisir de garder
  ind_aleat  = np.random.choice(n_cyl, p = probas)

  centres_p = centres.copy()
  centres_p[ind_aleat] = barycentre
  perturbation_voronoi.pertes_test = perte_tests
  perturbation_voronoi.cylindre_perturbe = ind_aleat
  perturbation_voronoi.probas = probas
  return centres_p

def perturbation_trou( arrangement, reglages = None):
  reglages_defaut =  { 'taille de la grille' : 10,  'fonction de perte' : perte_total_2d, 'reglages_pertes' : None}
  if reglages is None:
    reglages_int = reglages_defaut.copy()
  else:
    reglages_int = reglages_defaut | reglages

  n_grid = reglages_int['taille de la grille']
  fonction_perte = reglages_int['fonction de perte']
  reglages_pertes = reglages_int['reglages_pertes']

  n_cyl =len(arrangement['centres'])
  centres = arrangement['centres']
  dimensions_carton = arrangement['dimensions du carton']
  origine_carton = arrangement['origine du carton']

  grille = np.zeros((n_grid, n_grid, n_grid,3))
  voronoi_3d_calcul(arrangement, reglages)
  distances=voronoi_3d_calcul.distances
  trou = grille[0,0,0]
  x ,y, z = trou
  x,y,z = int(x), int(y), int(z)
  dist_trou = distances[x][y][z]
  for i in range(n_grid):
    for j in range(n_grid):
      for k in range(n_grid):
        if distances[i][j][k] > dist_trou:
          dist_trou = distances[i,j,k]
          trou =  [i,j,k]
  trou = [trou[0]/n_grid*dimensions_carton[0] + origine_carton[0],trou[1]/n_grid*dimensions_carton[1]+origine_carton[1],trou[2]/n_grid*dimensions_carton[2]+origine_carton[2]]
  
  #on test les cylindres
  pertes_tests = np.zeros(n_cyl)
  for i in range(n_cyl):
    centres_test = centres.copy()
    centres_test[i] = trou
    arrangement_test = arrangement.copy()
    arrangement_test['centres'] = centres_test
    pertes_tests[i] = fonction_perte(arrangement_test, reglages_pertes)

  
  #on garde celui dont le déplacement minimise la perte 
  ind = np.argmin(pertes_tests) 
  centres_p = centres.copy()
  centres_p[ind] = trou
  arrangement_sortie = arrangement.copy()
  arrangement_sortie['centres'] = centres_p
  perturbation_trou.pertes_test = pertes_tests
  perturbation_trou.cylindre_perturbe = ind
  return centres_p
  
def perturbation_trou_aleatoire(arrangement, reglages=None):

 ### PRÉPARATION
    ## GESTION DES REGLAGES
    reglages_defaut = {
        'taille de la grille' : 30,
        'fonction de perte' : perte_total_2d,
        'reglages_pertes' : None
    }
    if reglages is None:
        reglages_int = reglages_defaut.copy()
    else:
        reglages_int = reglages_defaut | reglages

    n_grid = reglages_int['taille de la grille']
    fonction_perte = reglages_int['fonction de perte']
    reglages_pertes = reglages_int['reglages_pertes']

    ## EXTRACTION DES DONNÉES
    dimensions_carton = arrangement['dimensions du carton']
    origine_carton = arrangement['origine du carton']
    centres = arrangement['centres']

    ## CALCUL DE LA DISTANCE À CHAQUE POINT DU CARTON
    voronoi_3d_calcul(arrangement, reglages_int)
    distances = voronoi_3d_calcul.distances

 ### CALCUL DES PERTURBATIONS
    ## On repère les 3 plus gros trous
    trous = trouver_trous(distances, dimensions_carton, origine_carton, n_grid, arrangement)

    ## On teste tous les cylindres dans chaque trou
    pertes = calculer_pertes_tests(arrangement, trous, fonction_perte, reglages_pertes)

    ## On choisit une perturbation avec proba softmax sur -perte réduite pondérée par une gaussienne clipée
    i, j = choisir_perturbation(pertes)
    nouveau_centre = trous[j]

 ### APPLICATION DE LA PERTURBATION
    centres_p = copy.deepcopy(centres)
    centres_p[i] = nouveau_centre
    arrangement_sortie = copy.deepcopy(arrangement)
    arrangement_sortie['centres'] = centres_p

 ### RETOUR
    perturbation_trou_aleatoire.pertes_test = pertes
    perturbation_trou_aleatoire.cylindre_perturbe = i
    perturbation_trou_aleatoire.indice_du_trou = j
    return centres_p

'sous fonctions pour perturbation_trou_aleatoire'
def calculer_coordonnees_reelles(indice_grille, dimensions_carton, origine_carton, n_grid):
    # On convertit les indices de la grille en coordonnées réelles dans le carton
    return [
        indice_grille[0] / n_grid * dimensions_carton[0] + origine_carton[0],
        indice_grille[1] / n_grid * dimensions_carton[1] + origine_carton[1],
        indice_grille[2] / n_grid * dimensions_carton[2] + origine_carton[2]
    ]

def trouver_trous(distances, dimensions_carton, origine_carton, n_grid, arrangement, n_trous=3):
    """
    Trouve les `n_trous` plus grandes distances dans la grille 2D (plan orthogonal à l'axe),
    en excluant les points trop proches du bord dans ce plan.
    """
    rayon = arrangement['rayon']
    axe = np.array(arrangement['axe'])

    # Déterminer les deux axes orthogonaux à l’axe principal
    directions = [np.array(e) for e in np.eye(3)]
    axes_ortho = [i for i, d in enumerate(directions) if not np.isclose(np.abs(np.dot(d, axe)), 1)]
    a1, a2 = axes_ortho
    a3 = 3 - a1 - a2  # l’axe aligné avec axe

    liste_indices = []
    liste_distances = []

    for i in range(n_grid):
        for j in range(n_grid):
            indice_grille = [0, 0, 0]
            indice_grille[a1] = i
            indice_grille[a2] = j
            indice_grille[a3] = 0  # unused mais requis par la fonction existante

            # Coordonnées réelles du point
            coord = calculer_coordonnees_reelles(indice_grille, dimensions_carton, origine_carton, n_grid)

            # Vérifie que ce point est à au moins `rayon` du bord du carton, uniquement dans le plan orthogonal
            bord_ok = True
            for dim in [a1, a2]:
                c = coord[dim]
                o = origine_carton[dim]
                L = dimensions_carton[dim]

            d = distances[i, j]
            if len(liste_distances) < n_trous:
                liste_distances.append(d)
                liste_indices.append(indice_grille)
            else:
                idx_min = np.argmin(liste_distances)
                if d > liste_distances[idx_min]:
                    liste_distances[idx_min] = d
                    liste_indices[idx_min] = indice_grille

    # Conversion finale
    return [calculer_coordonnees_reelles(ind, dimensions_carton, origine_carton, n_grid) for ind in liste_indices]

def calculer_pertes_tests(arrangement, trous, fonction_perte, reglages_pertes):
    # On teste chaque cylindre dans chaque trou et on mesure les pertes
    n_cyl = len(arrangement['centres'])
    pertes = np.zeros((n_cyl, len(trous)))
    axe = np.array(arrangement['axe'])
    axe_plan = []
    for e in np.eye(3):
      if np.dot(e, axe) == 0 :
        axe_plan.append( np.array(e))
    axe_plan = np.array(axe_plan)
  
    dim_carton_proj = np.array([np.dot(np.array(arrangement['dimensions du carton']), axe_plan[0]), np.dot(np.array(arrangement['dimensions du carton']), axe_plan[1])])

    for i in range(n_cyl):
        for j, trou in enumerate(trous):
            centres_test = copy.deepcopy(arrangement['centres'])
            centres_test[i] = trou
            arrangement_test = copy.deepcopy(arrangement)
            arrangement_test['centres'] = centres_test
            trou_projete = np.array([(np.dot(trou, axe_plan[0])) , (np.dot(trou, axe_plan[1]))])
            origine_projete = np.array([np.dot(np.array(arrangement['origine du carton']), axe_plan[0]), np.dot(np.array(arrangement['origine du carton']), axe_plan[1])])
            coeff = 1
            #coeff *= inv_gauss((trou_projete[0]- origine_projete[0])/dim_carton_proj[0] - 0.5)
            #coeff *= inv_gauss((trou_projete[1]- origine_projete[1])/dim_carton_proj[1] - 0.5)
            pertes[i, j] = fonction_perte(arrangement_test, reglages_pertes)*coeff**3

    return pertes

def choisir_perturbation(pertes):
    # On normalise les pertes et on en tire une avec une probabilité dépendante de la perte
    centrees = pertes - np.mean(pertes)
    reduites = centrees / (np.std(centrees) + 1e-8)
    exp = np.exp(-reduites)  # plus la perte est faible, plus la proba est grande
    proba = exp / np.sum(exp)

    n_cyl, n_trous = pertes.shape
    tirage = np.random.choice(n_cyl * n_trous, p=proba.flatten())
    i = tirage // n_trous
    j = tirage % n_trous

    return i, j

import copy
import numpy as np

def perturbation_centre(arrangement, reglages=None):

    ### PRÉPARATION
    ## GESTION DES RÉGLAGES
    reglages_defaut = {
        'fonction de perte': perte_total_2d,
        'reglages_pertes': None
    }
    if reglages is None:
        reglages_int = reglages_defaut.copy()
    else:
        reglages_int = reglages_defaut | reglages

    fonction_perte = reglages_int['fonction de perte']
    reglages_pertes = reglages_int['reglages_pertes']

    ## EXTRACTION DES DONNÉES
    dimensions_carton = arrangement['dimensions du carton']
    origine_carton = arrangement['origine du carton']
    centres = arrangement['centres']

    centre_du_carton = [
        origine_carton[0] + dimensions_carton[0] / 2,
        origine_carton[1] + dimensions_carton[1] / 2
    ]
    sigma = dimensions_carton[0] / 3

    ### CALCUL DES CONTRIBUTIONS AUX PERTES
    pertes_sans_centre = []
    for i in range(len(centres)):
        arrangement_sans_i = copy.deepcopy(arrangement)
        centres_sans_i = centres[:i] + centres[i+1:]
        arrangement_sans_i['centres'] = centres_sans_i
        perte = fonction_perte(arrangement_sans_i, reglages_pertes)
        pertes_sans_centre.append(perte)

    ## Softmax inversé (on veut plus de chance de perturber ceux qui causent de la perte)
    pertes_array = np.array(pertes_sans_centre)
    scores = pertes_array - np.max(pertes_array)  # stabilité numérique
    proba = np.exp(scores)
    proba /= np.sum(proba)

    i = np.random.choice(len(centres), p=proba)

    ### TIRAGE D'UN NOUVEAU CENTRE
    nouveau_centre = np.random.normal(loc=centre_du_carton, scale=sigma, size=3)

    ### APPLICATION DE LA PERTURBATION
    centres_p = copy.deepcopy(centres)
    centres_p[i] = nouveau_centre
    arrangement_sortie = copy.deepcopy(arrangement)
    arrangement_sortie['centres'] = centres_p

    ### RETOUR
    perturbation_centre.contributions = pertes_sans_centre
    perturbation_centre.cylindre_perturbe = i
    perturbation_centre.nouveau_centre = nouveau_centre
    return centres_p




