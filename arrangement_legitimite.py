# -*- coding: utf-8 -*-


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math
import scipy

from outils_courants import distance_deux_cylindres



def legitime_chevauchements(centres_cylindres,  longueur, rayon, axe, reglages = None):
    """
    Calcule les distances entre les paires de cylindres en 3D avec une métrique au choix.

    - centres_cylindres : (N,3) array des centres des cylindres.
    - longueur : longueur du cylindre (commune à tous les cylindres).
    - rayon : rayon du cylindre (commun à tous les cylindres).
    - axe : axe de rotation du cylindre (commune à tous les cylindres). (par défaut l'axe des x)
    - distance : fonction qui calcule la distance d'un point (x,y,z) au cylindre. (par défaut distance euclidienne entre deux cylindres)
    - erreur : tolérance sur la distance calculée avant d'annoncer qu'il y a un chevauchement. (par défaut 0)

    Retourne :
    - est_legitime_chevauchement : booleen indiquant si la configuration est physiquement possibles (en considérant seulement les chevauchements)
    """
    reglages_par_defaut = {'distance entre cylindres' : distance_deux_cylindres}
    if reglages is None :
      reglages = reglages_par_defaut
    else :
      reglages = reglages_par_defaut | reglages
    distance= reglages['distance entre cylindres']     

    n_cyl = centres_cylindres.shape[0]
    est_legitime_chevauchement = True
    for ind_1 in range(n_cyl-1):
      for ind_2 in range(ind_1+1,n_cyl):
        dist = distance(centres_cylindres[ind_1],
                        centres_cylindres[ind_2] ,rayon = rayon ,  longueur = longueur, axe = axe, reglages = reglages)
        if dist == -1:
          est_legitime_chevauchement = False
          #print("il y a un chevauchement entre les cylindres ", ind_1, " et ", ind_2)

    return(est_legitime_chevauchement)

def legitime_depassement(centres_cylindres,  longueur, rayon, dimensions_carton , axe, origine_carton , reglages = None):
  '''
  Détermine si une collection de cylindres est dans les limites d'un carton

    - centres_cylindres : (N,3) array des centres des cylindres
    - longueur : longueur du cylindre (commune à tous les cylindres).
    - rayon : rayon du cylindre (commun à tous les cylindres).
    - dimensions_carton : tableau (3) des dimensions du carton (longueur, largeur, hauteur) = (x,y,z)
    - axe : axe de rotation du cylindre (commun à tous les cylindres). 
    - distance : fonction qui calcule la distance d'un point (x,y,z) au cylindre. (par défaut distance euclidienne entre deux cylindres)
    - origine_carton : origine du carton
    - erreur : tolérance sur la distance calculée avant d'annoncer qu'il y a un dépassement. (par défaut 0)

  Retourne :
  - est_legitime_depassement : booleen indiquant si la configuration est physiquement possibles (en considérant seulement les dépassements)
    '''
  reglages_par_defaut = { 'fit_carton' : False , 'erreur acceptee depassement' : 0.001 * (rayon+longueur)}
  if reglages is None :
    reglages = reglages_par_defaut
  else :
    reglages = reglages_par_defaut | reglages    
  L=longueur
  R=rayon
  fit_carton = reglages['fit_carton']
  erreur = reglages['erreur acceptee depassement']
  axe=axe/np.dot(axe, axe)**0.5
  if abs(np.linalg.norm(axe)-1)>0.1:
    return("bleeeeeark")


  dim_carton = dimensions_carton
  u=np.random.uniform(0,1)
  if u<0.000001:
    print ("feur")
  point_extremes = np.zeros((3, 2, 3)) # [[point de gauche, point de droite][point du bas, ...]...]
  marges = np.zeros((3,2)) # [[marge à gauche, marge à droite], [marge en bas]]
  est_legitime_depassement = True
  directions = np.zeros((3, 3))
  for i in range(3):
    directions[i][i]=1
    #on cherche le point de coordonnée minimale selon cette direction
    #ainsi que les centres des cylindres correspondant
    point_extremes[i][0] = centres_cylindres[0]
    point_extremes[i][1] = centres_cylindres[0]

    for j in range(len(centres_cylindres)):
      coord_i = centres_cylindres[j][i]
      if coord_i < point_extremes[i][0][i]:
        point_extremes[i][0] = centres_cylindres[j]
      if coord_i > point_extremes[i][1][i]:
        point_extremes[i][1] = centres_cylindres[j]

    #côté min
    direction = - directions[i]
    cos_alpha = abs(np.dot(axe, direction))
    sin_alpha = np.sqrt(1-cos_alpha**2)
    elongement_vers_le_bord = sin_alpha*R +cos_alpha*L/2 #calcul de à quel point le cylindre va loin dans cette direction. Si la formule paraît obscure, cf dessin
    if elongement_vers_le_bord > point_extremes[i][0][i] - origine_carton[i]+erreur: #le terme  de droite correspond à la distance entre le centre du cylindre et la paroi selon la direction i
      est_legitime_depassement = False
    marges[i][0] =   (point_extremes[i][0][i] - origine_carton[i])-elongement_vers_le_bord

    #côté max
    direction =  directions[i]
    cos_alpha = abs(np.dot(axe, direction))
    sin_alpha = np.sqrt(1-cos_alpha**2)
    elongement_vers_le_bord = sin_alpha*R + cos_alpha*L/2
    if elongement_vers_le_bord > -point_extremes[i][1][i] + origine_carton[i] + dim_carton[i]+erreur:
      est_legitime_depassement = False
    marges[i][1] =   (origine_carton[i]+dim_carton[i] -point_extremes[i][1][i]) - elongement_vers_le_bord

  if fit_carton: #Si quelqu'un a eu le malheur de demander fit_carton = True, alors on décide de placer l'origine de manière à "coller" les cylindres à gauche, en bas... et puis on relance le programme
    org_carton = np.array( [0]*3)
    for i in range(3) :
      org_carton[i] = marges[i][0]-erreur
    est_legitime_depassement = legitime_depassement(centres_cylindres, longueur, rayon, dim_carton, axe, org_carton,erreur=erreur)
    point_extremes=legitime_depassement.point_extremes
    marges=legitime_depassement.marges
    directions=legitime_depassement.directions


  legitime_depassement.point_extremes = point_extremes
  legitime_depassement.marges = marges
  legitime_depassement.directions = directions

  return(est_legitime_depassement)

def est_legitime(arrangement, reglages = None):
  centres_cylindres = np.array(arrangement['centres'])
  rayon = arrangement['rayon']
  longueur = arrangement['longueur']
  dimensions_carton = np.array(arrangement['dimensions du carton'])
  origine_carton = np.array(arrangement['origine du carton'])
  axe = arrangement['axe']

  reglages_defaut = {'fit_carton': False, 'erreur_depassement' : 0.001*rayon, 'distance': distance_deux_cylindres, 'erreur_chevauchement' : 0.001*rayon}
  if reglages is None:
    reglages = reglages_defaut
  else:
    reglages = reglages_defaut | reglages
  leg_dep = legitime_depassement(centres_cylindres,  longueur, rayon, dimensions_carton, axe, origine_carton, reglages)
  leg_chev = legitime_chevauchements(centres_cylindres,  longueur, rayon, axe, reglages)

  est_legitime.legitime_depassement = leg_dep
  est_legitime.legitime_chevauchements = leg_chev
  est_legitime.points_extremes = legitime_depassement.point_extremes
  est_legitime.marges = legitime_depassement.marges

  return leg_dep and leg_chev