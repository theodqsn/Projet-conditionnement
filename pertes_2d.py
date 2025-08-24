import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math
from math import *
import scipy
import random
from arrangement_legitimite import legitime_depassement
from outils_courants import *
from outils_pertes import *
from penalisation_pertes import *
'fonctions outils'

def indic(x, portee):
  if x <= portee :
    return 1
  else :
    return 0

'fonctions de pénalisation'

def carre_tronque(x, reglages=None):
  reglages_par_defaut = {'scale' :1, 'derivee au bord': 1}
  if reglages is None :
    reglages = reglages_par_defaut
  else :
    reglages = reglages_par_defaut | reglages
  db = reglages['derivee au bord']
  if x <= 0 :
    return 0
  else :
    return 0.5*db*x**2

def puiss_4_tronque(x, reglages=None):
  return(carre_tronque(x, reglages)**2)

def puiss_8_tronque(x, reglages=None):
  return(carre_tronque(x/0.01, reglages)**4)

def exp_2x(x, reglages=None):
  return(np.exp(2*(x/0.1)))
'fonctions de perte'

def perte_chevauchement_2d(arrangement, reglages=None):

  reglages_par_defaut ={'penalisation chevauchement' : puiss_8_tronque, 'coefficient chevauchement'  :100, 'derivee au bord chevauchement' : 1 , 'scale chevauchement' : None, 'portee penalite chevauchement' : 0 }
  if reglages is None :
    reglages = reglages_par_defaut
  else :
    reglages = reglages_par_defaut | reglages

  centres  = arrangement['centres']
  rayon = arrangement['rayon']
  longueur = arrangement['longueur']
  axe = arrangement['axe']
  n_cyl = np.shape(centres)[0]
  if reglages['scale chevauchement'] is None:
    reglages['scale chevauchement'] = rayon
  reglages_fonction_de_penalisation = {'scale': reglages['scale chevauchement'], 'derivee au bord': reglages['derivee au bord chevauchement'], 'portee penalite': reglages['portee penalite chevauchement']}


  #on extrait les vecteurs qui portent le plan, ainsi que leurs indices
  axes = np.eye(3)
  axes_du_plan= np.zeros((3,3))
  ind= 0
  indices_direction_plan = np.zeros(2, np.int16)
  for i in range(3):
    if  not np.array_equal(axes[i], axe):
      axes_du_plan[ind]=axes[i]
      indices_direction_plan[ind]=i
      ind+=1

  distances = np.zeros((n_cyl, n_cyl)) #va servir à mesurer les "distances négatives"
  #on récupère les coordonnées selon ces axes*
  somme_penalites = 0
  for i in range(n_cyl):
    for j in range(i+1,n_cyl):
      u1,v1  = centres[i][indices_direction_plan[0]], centres[i][indices_direction_plan[1]]
      u2,v2  = centres[j][indices_direction_plan[0]], centres[j][indices_direction_plan[1]]

      distance_radiale_i_j = np.linalg.norm(np.array([u1-u2, v1-v2])) #on calcule la distance entre les deux centres
      if distance_radiale_i_j <= 4*rayon :
        distances[i][j] = 2*rayon - distance_radiale_i_j
        somme_penalites = somme_penalites + reglages['penalisation chevauchement'](distances[i][j], reglages_fonction_de_penalisation)

  perte_chevauchement_2d.distances = distances
  perte_chevauchement_2d.axes_du_plan = axes_du_plan

  return(reglages['coefficient chevauchement']*somme_penalites/(n_cyl))

def perte_depassement_2d(arrangement, reglages = None ):
  reglages_par_defaut =  {'penalisation depassement' : puiss_8_tronque , 'coefficient depassement' : 10000,  'derivee au bord depassement' : 1, 'scale depassement' : None, 'portee penalite depassement':0}
  if reglages is None :
    reglages = reglages_par_defaut
  else :
    reglages = reglages_par_defaut | reglages
  centres = arrangement['centres']
  rayon = arrangement['rayon']
  longueur = arrangement['longueur']
  axe = arrangement['axe']
  dimensions_carton = arrangement['dimensions du carton']
  origine_carton = arrangement['origine du carton']
  n_cyl = np.shape(centres)[0]
  if reglages['scale depassement'] is None:
    reglages['scale depassement'] = rayon
  reglages_fonction_de_penalisation = {'scale': reglages['scale depassement'], 'derivee au bord': reglages['derivee au bord depassement'], 'portee penalite': reglages['portee penalite depassement']*rayon}
  pen = reglages['penalisation depassement']

   #on extrait les vecteurs qui portent le plan, ainsi que leurs indices
  axes = np.eye(3)
  axes_du_plan= np.zeros((3,3))
  ind= 0
  indices_direction_plan = np.zeros(2, np.int16)
  for i in range(3):
    if  not np.array_equal(axes[i], axe):
      axes_du_plan[ind]=axes[i]
      indices_direction_plan[ind]=i
      ind+=1


  marges= np.zeros((n_cyl, 2,2)) # marges[i] = [[marge à gauche, marge à droite], [marge devant, marge derrière]]
  for i in range(n_cyl):
    marges[i][0][0] =  centres[i][indices_direction_plan[0]] - origine_carton[indices_direction_plan[0]] - rayon
    marges[i][0][1] = dimensions_carton[indices_direction_plan[0]] + origine_carton[indices_direction_plan[0]] - centres[i][indices_direction_plan[0]] - rayon
    marges[i][1][0] =  centres[i][indices_direction_plan[1]] - origine_carton[indices_direction_plan[1]] - rayon
    marges[i][1][1] = dimensions_carton[indices_direction_plan[1]] + origine_carton[indices_direction_plan[1]] - centres[i][indices_direction_plan[1]] - rayon

  perte_depassement_2d.marges = marges
  perte_depassement_2d.axes_du_plan = axes_du_plan

  penalite = 0
  s=0
  for i in range(n_cyl):
    for j in range(2):
      for k in range(2):
        penalite+= pen(-marges[i][j][k], reglages_fonction_de_penalisation)
        if marges[i][j][k] <0 :
          s=s- marges[i][j][k]
  return(reglages['coefficient depassement']*penalite/n_cyl)
 
def perte_densite_2d(arrangement, reglages = None ):
  reglages_par_defaut = {'penalisation densite' : id , 'coefficient densite' : 0}
  if reglages is None :
    reglages = reglages_par_defaut
  else :
    reglages = reglages_par_defaut | reglages
  centres = np.array(arrangement['centres'])
  dimensions= arrangement['dimensions du carton']
  n = np.shape(centres)[0]
  rayon = arrangement['rayon']
  d = 0
  for i in range(n):
    for j in range (i+1,n):
      d += np.linalg.norm(centres[i]-centres[j])- 2*rayon
  distance_moyenne = d/(n*(n)/2)
  distance_moyenne /= np.mean(dimensions)
  return(reglages['coefficient densite']*reglages['penalisation densite'](distance_moyenne))

def perte_alignement(arrangement, reglages = None):
  reglages_par_defaut = {'Ponderation' : exp_neg, 'penalisation alignement': mesure_dispersion, 'portee alignement' : 1, 'coefficient alignement' : 0}
  if reglages is None :
    reglages = reglages_par_defaut
  else :
    reglages = reglages_par_defaut | reglages

  centres = arrangement['centres']
  axe = arrangement['axe']
  rayon = arrangement['rayon']
  ponderation = reglages['Ponderation']
  penalisation_alignement = reglages['penalisation alignement']
  coeff_align = reglages['coefficient alignement']

  n_centres = len(centres)
  angles_ponderations= [] # ((angle, ponderation) for angle in angles)
  reglages_int = {'scale' : reglages['portee alignement']*rayon}

  for i in range(n_centres):
    for j in range(i+1,n_centres) :
      distance, angle = calculs_spatiaux(centres[i], centres[j], axe) #angle entre 0 et pi/2
      pond = ponderation(distance/rayon, reglages = reglages_int)
      angles_ponderations = ajout_angle(angles_ponderations, angle, pond)

  perte = penalisation_alignement(angles_ponderations, reglages)
  return coeff_align*perte

def perte_total_2d(arrangement, reglages= None ):
  reglages_par_defaut = {}
  if reglages is not None:
    reglages = reglages_par_defaut | reglages
  else:
    reglages = reglages_par_defaut
  s= 0
  a= perte_chevauchement_2d(arrangement, reglages)
  b= perte_depassement_2d(arrangement, reglages)
  c= perte_densite_2d(arrangement, reglages)
  d= perte_alignement(arrangement, reglages)
  s= a+b+c+d
  perte_total_2d.perte_de_depassement = b
  perte_total_2d.perte_de_chevauchement = a
  perte_total_2d.perte_de_densite = c
  perte_total_2d.perte_de_alignement = d
  return(s)

def perte_totale_1d_2d(centres_cylindres_1d, rayon, longueur, dimensions_carton, axe=[1,0,0], origine_carton = [0,0,0], reglages=None):
  '''
  Prends un tableau 1d de centre de cylindres et renvoie une perte totale
  '''
  reglages_par_defaut = {}
  if reglages is not None:
    reglages = reglages_par_defaut | reglages
  else:
    reglages = reglages_par_defaut

  centres_cylindres = t_2_d(centres_cylindres_1d)
  arrangement = {'centres': centres_cylindres, 'rayon' : rayon, 'longueur' : longueur, 'axe' : axe, 'dimensions du carton' : dimensions_carton, 'origine du carton': origine_carton }
  perte = (perte_total_2d(arrangement, reglages ))
  perte_totale_1d_2d.perte_de_depassement = perte_total_2d.perte_de_depassement
  perte_totale_1d_2d.perte_de_chevauchement = perte_total_2d.perte_de_chevauchement
  perte_totale_1d_2d.perte_de_densite = perte_total_2d.perte_de_densite
  perte_totale_1d_2d.perte_de_alignement = perte_total_2d.perte_de_alignement
  return(perte)

def perte_non_alignement(arrangement_bibos, reglages = None):
  reglages_defaut = {
      'reglages_fonction_pen' : None,
                     'coeff_pen' : 0
  }
  if reglages is not None:
    reglages = reglages_defaut | reglages
  else:
    reglages = reglages_defaut
  orientations = arrangement_bibos['orientations']
  orientations = orientations/np.linalg.norm(orientations, axis=1, keepdims=True)
  angles = np.zeros(len(orientations))
  for i in range(len(orientations)):
    angles[i] = np.arctan2(orientations[i][1],orientations[i][0])
    if angles[i] < 0 :
      angles[i] += 2*np.pi
  rec = rec_noyaux(angles, np.ones(len(orientations)), reglages['reglages_fonction_pen'])
  perte  =  1/convol(rec, reglages={'p':10, 'bande passante': 0.1}) + 1/convol(rec, reglages={'p':10, 'bande passante': 0.01})
  return(perte)
