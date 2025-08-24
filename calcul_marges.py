

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math
from math import *
import scipy

from outils_courants import *

def carton_min(centres_cylindres, longueur, rayon, axe , reglages = None): #calcule le carton minimale englobant la configuration
  L=longueur
  R=rayon
  axe=axe/np.dot(axe, axe)**0.5

  directions = np.zeros((3,3))
  point_extremes = np.zeros((3, 2, 3)) # [[point de gauche, point de droite][point du bas, ...]...]
  longueurs_min = np.zeros(3) #longueur du carton min selon les axes des x, des y, des z
  origine = np.zeros(3)
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

    longueurs_min[i] = point_extremes[i][1][i] - point_extremes[i][0][i]

    #côté min
    direction = - directions[i]
    cos_alpha = abs(np.dot(axe, direction))
    sin_alpha = np.sqrt(1-cos_alpha**2)
    elongement_vers_le_bord = sin_alpha*R +cos_alpha*L/2 #calcul de à quel point le cylindre va loin dans cette direction. Si la formule paraît obscure, cf dessin dans arrangement_legitimite
    #print('elongement vers le bord côté min : ', elongement_vers_le_bord)
    longueurs_min[i] += elongement_vers_le_bord
    origine[i] = point_extremes[i][0][i] + elongement_vers_le_bord*direction[i]


    #côté max
    direction =  directions[i]
    cos_alpha = abs(np.dot(axe, direction))
    sin_alpha = np.sqrt(1-cos_alpha**2)
    elongement_vers_le_bord = sin_alpha*R + cos_alpha*L/2
    #print('elongement vers le bord côté max : ', elongement_vers_le_bord)
    longueurs_min[i] += elongement_vers_le_bord

  carton_min.origine = origine


  return(longueurs_min)

def dilatation_arrangement(centres_cylindres, coefficients, origine_sous_carton, origine_carton, reglages = None):
  translates = np.zeros((len(centres_cylindres), 3))
  origine_sous_carton = np.array(origine_sous_carton)
  origine_carton = np.array(origine_carton)
  for i in range(len(centres_cylindres)): #on translate-dilate-translate pour ne pas dilater le vecteur qui porte l'origine
    translates[i] = centres_cylindres[i] - origine_sous_carton
  dilates = np.zeros((len(centres_cylindres), 3))
  for i in range(len(centres_cylindres)):
    for j in range(3):
      dilates[i][j] = translates[i][j] * coefficients[j]
  for i in range(len(centres_cylindres)): #on retranslate dans l'autre sens
    translates[i] = dilates[i] + origine_carton
  return(translates)

def marges_a_arrangement_fixe(centres_cylindres, longueur, rayon, axe , dimensions_carton , origine_carton, reglages = None ):

  #on commence à calculer le "carton minimal" entourant cette configuration
  dimensions_carton_min = carton_min(centres_cylindres, longueur, rayon, axe, reglages= reglages)
  origine_carton_min = carton_min.origine

  #ce morceau sert à identifer quels axes dépendront du rayon, et quel axe dépendra de la longueur
  tous_axes = np.eye(3)
  axe_du_cylindre = axe #on part du principe ici que les cylindres sont orientés selon un des axes du carton
  autres_axes = np.zeros((2,3))
  ind= 0
  indices_autres_axes = np.zeros(2, np.int64)
  indice_axe_cylindre = 0
  for i in range(3) :
    e = tous_axes[i]
    if not np.array_equal(e, axe_du_cylindre):
      autres_axes[ind] = e
      indices_autres_axes[ind] = i
      ind += 1
    else :
      indice_axe_cylindre = i



  longueur_maximale_proportion = (dimensions_carton[indice_axe_cylindre]/dimensions_carton_min[indice_axe_cylindre]) #si la longueur du carton selon l'axe de rotation du cylindre est deux fois plus grande que la longueur minimale, alors on peut doubler la longueur des cylindres
  longueur_maximale = longueur_maximale_proportion * longueur
  rayon_maximal_proportion_deux_directions = np.array(((dimensions_carton[indices_autres_axes[0]]/dimensions_carton_min[indices_autres_axes[0]]), (dimensions_carton[indices_autres_axes[1]]/dimensions_carton_min[indices_autres_axes[1]])))
  rayon_maximal_proportion= np.min(rayon_maximal_proportion_deux_directions)
  rayon_maximal = rayon_maximal_proportion * rayon

  coefficients_dilatation = np.zeros(3)
  coefficients_dilatation[indice_axe_cylindre] = longueur_maximale_proportion
  coefficients_dilatation[indices_autres_axes[0]] = rayon_maximal_proportion
  coefficients_dilatation[indices_autres_axes[1]] = rayon_maximal_proportion
  arrangement= dilatation_arrangement(centres_cylindres, coefficients_dilatation, origine_carton_min, origine_carton, reglages) #on translate le carton minimal pour faire correpondre les origines et on dilate les coordonées centrées

  marges_a_arrangement_fixe.rayon_maximal_en_proportion_deux_directions = rayon_maximal_proportion_deux_directions
  marges_a_arrangement_fixe.longueur_maximale_en_proportion = longueur_maximale_proportion
  marges_a_arrangement_fixe.rayon_maximal_en_proportion = rayon_maximal_proportion
  marges_a_arrangement_fixe.dimension_du_carton_minimal = dimensions_carton_min
  marges_a_arrangement_fixe.origine_du_carton_minimal = origine_carton_min
  marges_a_arrangement_fixe.longueur_maximale = longueur_maximale
  marges_a_arrangement_fixe.rayon_maximal = rayon_maximal
  marges_a_arrangement_fixe.arrangement = arrangement

def a_la_main(centres_cylindres, ancienne_longueur, ancien_rayon, nouvelle_longueur, nouveau_rayon, axe=[1,0,0], origine_carton=[0,0,0]):

  #ce morceau sert à identifer quels axes dépendront du rayon, et quel axe dépendra de la longueur
  tous_axes = np.eye(3)
  axe_du_cylindre = axe #on part du principe ici que les cylindres sont orientés selon un des axes du carton
  autres_axes = np.zeros((2,3))
  ind= 0
  indices_autres_axes = np.zeros(2, np.int64)
  indice_axe_cylindre = 0
  for i in range(3) :
    e = tous_axes[i]
    if not np.array_equal(e, axe_du_cylindre):
      autres_axes[ind] = e
      indices_autres_axes[ind] = i
      ind += 1
    else :
      indice_axe_cylindre = i

  #on calcule le "carton minimal" entourant cette configuration (juste pour avoir l'orine en fait)
  dimensions_carton_min = carton_min(centres_cylindres, ancienne_longueur, ancien_rayon, axe, dimensions_carton, origine_carton)
  origine_carton_min = carton_min.origine

  coefficients_dilatation = np.zeros(3)
  coefficients_dilatation[indice_axe_cylindre] = nouvelle_longueur/ancienne_longueur
  coefficients_dilatation[indices_autres_axes[0]] = nouveau_rayon/ancien_rayon
  coefficients_dilatation[indices_autres_axes[1]] = nouveau_rayon/ancien_rayon

  return(dilatation_arrangement(centres_cylindres, coefficients_dilatation, origine_carton_min, origine_carton))

def marge(arrangement, reglages):
  centres_cylindres = arrangement['centres']
  longueur = arrangement['longueur']
  rayon = arrangement['rayon']
  axe = arrangement['axe']
  dimensions_carton = arrangement['dimensions du carton']
  origine_carton = arrangement['origine du carton']

  marges_a_arrangement_fixe(centres_cylindres, longueur, rayon, axe , dimensions_carton , origine_carton, reglages)
  marge.rayon_maximal_en_proportion_deux_directions=marges_a_arrangement_fixe.rayon_maximal_en_proportion_deux_directions
  marge.longueur_maximale_en_proportion=marges_a_arrangement_fixe.longueur_maximale_en_proportion
  marge.rayon_maximal_en_proportion=marges_a_arrangement_fixe.rayon_maximal_en_proportion 
  marge.dimension_du_carton_minimal=marges_a_arrangement_fixe.dimension_du_carton_minimal
  marge.origine_du_carton_minimal=marges_a_arrangement_fixe.origine_du_carton_minimal
  marge.longueur_maximale=marges_a_arrangement_fixe.longueur_maximale
  marge.rayon_maximal=marges_a_arrangement_fixe.rayon_maximal
  marge.centres_sortie=marges_a_arrangement_fixe.arrangement