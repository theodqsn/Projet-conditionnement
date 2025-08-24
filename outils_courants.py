

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math
from math import *
import scipy
import csv
import csv

try:
    import Levenshtein
except ImportError:
    import subprocess
    import sys
    subprocess.check_call([sys.executable, "-m", "pip", "install", "python-Levenshtein"])
    import Levenshtein

def copier_attributs(source, cible):
    for k, v in vars(source).items():
        setattr(cible, k, v)

def projection(tab, axe):
  n = len(tab)
  tab_reduit = np.zeros(int(2*n/3))
  ind = 0
  mod = np.dot([0,1,2], axe)
  for i in range(n):
    if i % 3 != mod:
      tab_reduit[ind] = tab[i]
      ind += 1
  return tab_reduit

def injection (tab, axe, valeur):
  l = []
  n = int(1.5*len(tab))
  ind = 0
  mod = np.dot([0,1,2], axe)
  for i in range(n):
    if i % 3 != mod:
      l.append(tab[ind])
      ind += 1
    else:
      l.append(valeur)
  return np.array(l)

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

def chaine_la_plus_proche(chaine, liste_de_chaines):
        dist=[Levenshtein.distance(chaine, liste_de_chaines[i]) for i in range(len(liste_de_chaines))]
        ind_min = np.argmin(dist)
        if dist[ind_min] == 0:
          return(liste_de_chaines[ind_min])
        inp ='a'
        while inp != 'n' or inp != 'o':
          if liste_de_chaines[ind_min]!=chaine:
            print('nom rentré inexacte')
            print('vous avez entré', '"',  chaine, '"')
            print('vouliez vous dire : ', '"', liste_de_chaines[ind_min], '" ?')
            inp=input('tapez o pour oui, n pour non')
            if inp=='Patate':
              return('Vous avez gagné !')
            if inp=='o':
              return(liste_de_chaines[ind_min])
            if inp == 'n' :
              raise ValueError('nom non reconnu')

def distance_cylindre(centre_cylindre, point , rayon, longueur, axe, reglages = None):
  '''
  Calcule la distance entre un point (x,y,z) et un cylindre de rayon r et de longueur l centré en (x0,y0,z0), et d'axe axe
  La distance sera la distance entre le point et son projeté orthogonal sur le cylindre
  Printer_test ne sert que pour
  '''
  r=rayon
  l=longueur
  x,y,z = centre_cylindre
  x0,y0,z0 = point

  #on crée une base orthonormée adaptée au cylindre par gram-schmidt
  epsilon = np.random.normal(0,5, size=3)
  u1=np.array(axe)
  u2=np.array(axe) + epsilon
  epsilon = np.random.normal(0,5, size=3)
  u3=np.array(axe) + epsilon

  e1=u1/(np.linalg.norm(u1))
  e2 = (u2 - np.dot(u2,e1)*e1)/np.linalg.norm(u2 - np.dot(u2,e1)*e1)
  e3 = (u3 - np.dot(u3,e1)*e1 - np.dot(u3,e2)*e2)/np.linalg.norm((u3 - np.dot(u3,e1)*e1 - np.dot(u3,e2)*e2))

  # on definit les ecarts dans chaque direction
  delta = [x-x0, y-y0, z-z0]
  delta1 = np.dot(delta, e1)
  delta2 = np.dot(delta, e2)
  delta3 = np.dot(delta, e3)

  R= np.sqrt(delta3**2 + delta2**2) #distance radiale au point centre
  L= abs(delta1) #distance axiale au point centre

  R_pos=R-r # distance radiale entre le point et le cylindre
  if R_pos<0:
    R_pos=0

  L_pos=L-l/2
  if L_pos<0:
    L_pos=0

  return np.sqrt(R_pos**2 + L_pos**2)

def distance_deux_cylindres(centre_1, centre_2 , rayon, longueur, axe, reglages = None ):
  reglages_par_defaut = {'erreur acceptee distance' : 0.001*rayon}
  if reglages is None :
    reglages = reglages_par_defaut
  else :
    reglages = reglages_par_defaut | reglages
  
  r=rayon
  l=longueur
  x1,y1,z1 = centre_1
  x2,y2, z2 = centre_2
  erreur = reglages['erreur acceptee distance']

  #on crée une base orthonormée adaptée au cylindre par gram-schmidt
  epsilon = np.random.normal(0,5, size=3)
  u1=np.array(axe)
  u2=np.array(axe) + epsilon
  epsilon = np.random.normal(0,5, size=3)
  u3=np.array(axe) + epsilon

  e1=u1/(np.linalg.norm(u1))
  e2 = (u2 - np.dot(u2,e1)*e1)/np.linalg.norm(u2 - np.dot(u2,e1)*e1)
  e3 = (u3 - np.dot(u3,e1)*e1 - np.dot(u3,e2)*e2)/np.linalg.norm((u3 - np.dot(u3,e1)*e1 - np.dot(u3,e2)*e2))

  # on definit les ecarts dans chaque direction
  delta = [x1-x2, y1-y2, z1-z2]
  delta1 = np.dot(delta, e1)
  delta2 = np.dot(delta, e2)
  delta3 = np.dot(delta, e3)

  R= np.sqrt(delta3**2 + delta2**2) #distance radiale au point centre
  L= abs(delta1) #distance axiale au point centre
  distance_deux_cylindres.R= R
  distance_deux_cylindres.L= L
  distance_deux_cylindres.e1= e1
  distance_deux_cylindres.e2= e2
  distance_deux_cylindres.e3= e3
  if R<2*r-erreur and L<l-erreur:
    return(-1)
  else :
    L_pos = L - l
    if L_pos<0:
      L_pos = 0
    R_pos = R - 2*r
    if R_pos<0:
      R_pos = 0
    return( np.sqrt(R_pos**2 + L_pos**2))

def densite(centres_cylindres, rayon, longueur, axe=[1,0,0]):
  '''
  Détermine la densite d'une collection de cylindres
    - centres_cylindres : (N,3) array des centres des cylindres
    - longueur : longueur du cylindre (commune à tous les cylindres).
    - rayon : rayon du cylindre (commun à tous les cylindres).
    - axe : axe de rotation du cylindre (commune à tous les cylindres). (par défaut l'axe des x)
  '''
  R=rayon
  L=longueur
  dim=3
  point_extremes = np.zeros((3,2,3)) # [[point de gauche, point de droite][point du bas, ...]...]
  valeurs_extremes = np.zeros((3,2)) # [[gauchesse maximale, droitesse maximale][hauteur minimale, ...]...]
  directions = np.zeros((dim,dim))
  for i in range(dim):
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


    direction = directions[i]
    cos_alpha = abs(np.dot(axe, direction))
    sin_alpha = np.sqrt(1-cos_alpha**2)
    elongement_vers_le_bord = sin_alpha*R +cos_alpha*L/2

    valeurs_extremes[i][0] = point_extremes[i][0][i] - elongement_vers_le_bord  #on regarde quels sont les points extremaux parmi les cylindres. Si la formule paraît obscure, cf le notebook, Arrangement_Legitimite.ipynb
    valeurs_extremes[i][1] = point_extremes[i][1][i] +  elongement_vers_le_bord #on regarde quels sont les points extremaux parmi les cylindres. Si la formule paraît obscure, cf le notebook, Arrangement_Legitimite.ipynb
  volume_cylindre = np.pi*rayon**2*longueur
  nombre_cylindres = len(centres_cylindres)
  densite.extremites = valeurs_extremes
  densite.volume_cylindre = np.pi*rayon**2*longueur
  densite.nombre_cylindres = len(centres_cylindres)
  return ((volume_cylindre * nombre_cylindres)/((valeurs_extremes[0][1]-valeurs_extremes[0][0])*(valeurs_extremes[1][1]-valeurs_extremes[1][0])*(valeurs_extremes[2][1]-valeurs_extremes[2][0]) ))

def densite_simple(centres_cylindres, longueur, rayon):
  min_x = min(centres_cylindres[:,0])
  max_x = max(centres_cylindres[:,0])
  min_y = min(centres_cylindres[:,1])
  max_y = max(centres_cylindres[:,1])
  min_z = min(centres_cylindres[:,2])
  max_z = max(centres_cylindres[:,2])
  n_cylindre = np.shape(centres_cylindres)[0]
  vol = n_cylindre * pi * rayon**2 * longueur
  l= min(rayon, longueur)/2
  return vol/ ((max_x-min_x+l)*(max_y-min_y+l)*(max_z-min_z+l))

def conversion(s): #j'avoue c'est une fonction chatgpt
    # Détection automatique du séparateur
    dialect = csv.Sniffer().sniff(s)
    separator = dialect.delimiter

    # Conversion en liste de floats
    return list(map(float, s.split(separator)))

def t_1_d(T): #ici T est de taille (n,3)
  n= np.shape(T)[0]
  t= np.zeros(3*n)
  ind=0 #parce que j'ai la flemme d'utiliser et de justifier des formules compliquées avec des modulos
  for i in range(n):
    t[ind]=T[i][0]
    ind+=1
    t[ind]=T[i][1]
    ind+=1
    t[ind]=T[i][2]
    ind+=1
  return t

def t_2_d(t): #ici t est de taille (3n)
  n=int(np.shape(t)[0]/3)
  T= np.zeros((n,3))
  ind=0
  for i in range(n):
    T[i][0]=t[ind]
    ind+=1
    T[i][1]=t[ind]
    ind+=1
    T[i][2]=t[ind]
    ind+=1
  return T

def distance_bibo(point, bibo):
  grand_centre = bibo['centre']
  orientation = bibo['orientation']/np.linalg.norm(bibo['orientation'])
  grand_rayon = bibo['grand_rayon']
  petit_rayon = bibo['petit_rayon']
  longueur = bibo['longueur']
  axe = bibo['axe']

  petit_centre = grand_centre + orientation*(grand_rayon+petit_rayon)
  distance_petit_centre = distance_deux_cylindres(point[0], point[1], point[2], petit_centre[0], petit_centre[1], petit_centre[2], petit_rayon, longueur, axe)
  distance_grand_centre = distance_deux_cylindres(point[0], point[1], point[2], grand_centre[0], grand_centre[1], grand_centre[2], grand_rayon, longueur, axe)
  distance_bibo.distance_petit_centre = distance_petit_centre
  distance_bibo.distance_grand_centre = distance_grand_centre
  return(min(distance_petit_centre, distance_grand_centre))

def distance_bibo(point, bibo):
  grand_centre = bibo['centre']
  orientation = bibo['orientation']/np.linalg.norm(bibo['orientation'])
  grand_rayon = bibo['grand_rayon']
  petit_rayon = bibo['petit_rayon']
  longueur = bibo['longueur']
  axe = bibo['axe']

  petit_centre = grand_centre + orientation*(grand_rayon+petit_rayon)

  
  distance_petit_centre = distance_cylindre(point[0], point[1], point[2], petit_centre[0], petit_centre[1], petit_centre[2], petit_rayon, longueur, axe)
  distance_grand_centre = distance_cylindre(point[0], point[1], point[2], grand_centre[0], grand_centre[1], grand_centre[2], grand_rayon, longueur, axe)
  distance_bibo.distance_petit_centre = distance_petit_centre
  distance_bibo.distance_grand_centre = distance_grand_centre
  return(min(distance_petit_centre, distance_grand_centre))

def distance_deux_bibos(bibo1, bibo2):
  grand_centre_1 = bibo1['centre']
  orientation_1 = bibo1['orientation']/np.linalg.norm(bibo1['orientation'])
  grand_rayon_1 = bibo1['grand_rayon']
  petit_rayon_1 = bibo1['petit_rayon']
  longueur_1 = bibo1['longueur']
  axe_1 = bibo1['axe']

  petit_centre_1 = grand_centre_1 + orientation_1*(grand_rayon_1+petit_rayon_1)


  grand_centre_2 = bibo2['centre']
  orientation_2 = bibo2['orientation']/np.linalg.norm(bibo2['orientation'])
  grand_rayon_2 = bibo2['grand_rayon']
  petit_rayon_2 = bibo2['petit_rayon']
  longueur_2 = bibo2['longueur']
  axe_2 = bibo2['axe']

  petit_centre_2 = grand_centre_2 + orientation_2*(grand_rayon_2+petit_rayon_2)

  a= distance_deux_cylindres(grand_centre_1[0], grand_centre_1[1], grand_centre_1[2], grand_centre_2[0], grand_centre_2[1], grand_centre_2[2], 0.5*(grand_rayon_1 + grand_rayon_2), 0.5*(longueur_1+ longueur_2), axe_1)
  b= distance_deux_cylindres(petit_centre_1[0], petit_centre_1[1], petit_centre_1[2], petit_centre_2[0], petit_centre_2[1], petit_centre_2[2], 0.5 * (petit_rayon_1 + petit_rayon_2), 0.5*(longueur_1+ longueur_2), axe_1)
  c= distance_deux_cylindres(petit_centre_1[0], petit_centre_1[1], petit_centre_1[2], grand_centre_2[0], grand_centre_2[1], grand_centre_2[2], 0.5*(petit_rayon_1+grand_rayon_2), 0.5*(longueur_1+longueur_2), axe_1 )
  d = distance_deux_cylindres(grand_centre_1[0], grand_centre_1[1], grand_centre_1[2], petit_centre_2[0], petit_centre_2[1], petit_centre_2[2], 0.5*(grand_rayon_1+petit_rayon_2), 0.5*(longueur_1+longueur_2), axe_1)
  return (np.min([a,b,c,d]))

def densite_bibo(arrangement_bibo):
  # on regarde densité.valeurs_extremes pour les grands cylindres et pour les petits cylindres
  grands_cylindres = arrangement_bibo['centres']
  orientations = arrangement_bibo['orientations']
  orientations = orientations/np.linalg.norm(orientations, axis=1, keepdims=True)
  grand_rayon = arrangement_bibo['grand_rayon']
  petit_rayon = arrangement_bibo['petit_rayon']
  longueur = arrangement_bibo['longueur']
  axe = arrangement_bibo['axe']
  nombre = arrangement_bibo['nombre']
  origine_carton = arrangement_bibo['origine du carton']
  dimensions_carton = arrangement_bibo['dimensions du carton']

  #on calcul les centres des petits cylindres
  petits_centres =  [grand_centre + orientation*(grand_rayon+petit_rayon) for grand_centre, orientation in zip(grands_cylindres, orientations)]

  #on calcul les valeurs extrêmes pour la "place minimale" prise par les petits cylindres
  densite(petits_centres,petit_rayon, longueur, axe)
  extremites_petits = densite.extremites

  #on calcul les valeurs extrêmes pour la "place minimale" prise par les grands cylindres
  densite(grands_cylindres, grand_rayon, longueur, axe)
  extremites_grands = densite.extremites

  extremites_totales = np.zeros((3,2))
  for i in range(3):
    extremites_totales[i][0] = np.min([extremites_petits[i][0], extremites_grands[i][0]])
    extremites_totales[i][1] = np.max([extremites_petits[i][1], extremites_grands[i][1]])

  volume_grands_cylindres = nombre * pi * grand_rayon**2 * longueur
  volume_petits_cylindres = nombre * pi * petit_rayon**2 * longueur
  volume_carton_minimale = (extremites_totales[0][1] - extremites_totales[0][0])*(extremites_totales[1][1] - extremites_totales[1][0])*(extremites_totales[2][1] - extremites_totales[2][0])
  volume_cylindres_totales = volume_grands_cylindres + volume_petits_cylindres

  return(volume_cylindres_totales/volume_carton_minimale)

def t_1_d_bibo(arrangement_bibo):
  """
  Convertit les centres et orientations d'un arrangement de bibos en un vecteur 1D.

  Args:
    arrangement_bibo: Un dictionnaire représentant l'arrangement de bibos.
                       Doit contenir les clés 'centres' (liste de centres) et
                       'orientations' (liste d'orientations).

  Returns:
    Un vecteur 1D contenant les coordonnées x, y, z de chaque centre et les
    composantes x, y, z de chaque orientation.
  """
  centres = arrangement_bibo['centres']
  orientations = arrangement_bibo['orientations']
  orientations = orientations/np.linalg.norm(orientations, axis=1, keepdims=True)
  nombre = arrangement_bibo['nombre']
  t = np.zeros(6 * nombre)
  for i in range(nombre):
    t[6*i] = centres[i][0]
    t[6*i + 1] = centres[i][1]
    t[6*i + 2] = centres[i][2]
    t[6*i + 3] = orientations[i][0]
    t[6*i + 4] = orientations[i][1]
    t[6*i + 5] = orientations[i][2]
  return t

def t_2_d_bibo(t):
  """
  Convertit un vecteur 1D en centres et orientations d'un arrangement de bibos.

  Args:
    t: Un vecteur 1D contenant les coordonnées x, y, z de chaque centre et les
       composantes x, y, z de chaque orientation.
    nombre: Le nombre de bibos dans l'arrangement.

  Returns:
    Un dictionnaire contenant une partie de l'arrangement de bibos avec les clés 'centres'
    et 'orientations'.
  """
  nombre = int(len(t)/6)
  centres = []
  orientations = []
  for i in range(nombre):
    centres.append([t[6*i], t[6*i + 1], t[6*i + 2]])
    orientations.append([t[6*i + 3], t[6*i + 4], t[6*i + 5]])
  return {'centres': centres, 'orientations': orientations}
