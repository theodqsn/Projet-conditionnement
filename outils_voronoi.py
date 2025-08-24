
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math
import scipy

from outils_courants import *
from outils_dessin import *

def voronoi_3d_calcul(arrangement, reglages=None):
    """
    Calcule les cellules de Voronoi dans le plan orthogonal à l'axe,
    à la coupe passant par le centre du carton selon cet axe.
    """
    # Réglages par défaut
    reglages_par_defaut = {'mesure de distance': distance_cylindre, 'taille de la grille': 10}
    if reglages is None:
        reglages = reglages_par_defaut
    else:
        reglages = reglages_par_defaut | reglages

    # Données
    grid_size = reglages['taille de la grille']
    distance = reglages['mesure de distance']
    centres_cylindres = arrangement['centres']
    longueur = arrangement['longueur']
    rayon = arrangement['rayon']
    axe = np.array(arrangement['axe'])
    origine_carton = np.array(arrangement['origine du carton'])
    dimensions_carton = np.array(arrangement['dimensions du carton'])

    # Trouver les 2 directions orthogonales à l'axe
    directions = [np.array(e) for e in np.eye(3)]
    axes_ortho = [i for i, d in enumerate(directions) if not np.isclose(np.abs(np.dot(d, axe)), 1)]
    a1, a2 = axes_ortho
    a3 = 3 - a1 - a2  # l'axe aligné avec `axe`

    # Coordonnée fixe selon axe
    centre_a3 = origine_carton[a3] + dimensions_carton[a3] / 2

    # Coordonnées réelles dans le plan orthogonal à l'axe
    val_1 = np.linspace(origine_carton[a1], origine_carton[a1] + dimensions_carton[a1], grid_size)
    val_2 = np.linspace(origine_carton[a2], origine_carton[a2] + dimensions_carton[a2], grid_size)

    # Grilles de sortie
    voronoi_cells = np.zeros((grid_size, grid_size, 3))
    distances = np.zeros((grid_size, grid_size))

    for i in range(grid_size):
        for j in range(grid_size):
            # Construction du point dans l'espace réel
            point = [0.0, 0.0, 0.0]
            point[a1] = val_1[i]
            point[a2] = val_2[j]
            point[a3] = centre_a3

            # Calcul de la distance au premier centre
            x0, y0, z0 = centres_cylindres[0]
            min_dist = distance(point, [x0, y0, z0], rayon, longueur, axe, reglages)
            voronoi_cells[i, j] = [x0, y0, z0]

            # Parcours des autres centres
            for x0, y0, z0 in centres_cylindres[1:]:
                dist = distance(point, [x0, y0, z0], rayon, longueur, axe, reglages)
                if dist < min_dist:
                    min_dist = dist
                    voronoi_cells[i, j] = [x0, y0, z0]

            distances[i, j] = min_dist

    # Enregistrement des distances pour usage externe
    voronoi_3d_calcul.distances = distances
    return voronoi_cells


def plus_gros_trou(arrangement, reglages = None):
  reglages_par_defaut = {'coefficient mesure trou' : 1}
  if reglages is None : 
    reglages=  reglages_par_defaut
  else :
    reglages = reglages_par_defaut |reglages
  coeff_trou = reglages['coefficient mesure trou']
  n_cyl = len(arrangement['centres'])
  centres = arrangement['centres'].copy()
  dimensions_carton = arrangement['dimensions du carton']
  origine_carton = arrangement['origine du carton']
  attributions = voronoi_3d_calcul(arrangement, reglages)

  #on récupère la matrice des distances de chaque point à son centre le plus proche, calculée dans voronoi_3d
  distances = voronoi_3d_calcul.distances
  n_grid = np.shape(distances)[0]
  taille_cellules = np.zeros(n_cyl) #on essaie de trouve quelle est la plus grande cellule (celle qui laissera le plus grand trou)
  cpt=np.zeros(n_cyl)
  for ind in range(n_cyl): #en moyenne, mettre cette ligne ici plutôt que plus bas double le temps de calcul, mais c'est plus clair. On pourrait aussi bien plus optimiser en triant centres
    for i in range(n_grid):
      for j in range(n_grid):
        for k in range(n_grid):
          if (attributions[i][j][k][0], attributions[i][j][k][1], attributions[i][j][k][2])== (centres[ind][0], centres[ind][1], centres[ind][2]):
            taille_cellules[ind]+=distances[i][j][k]**coeff_trou
            cpt[ind]+=1
  cellule_max = np.argmax(taille_cellules)# on récupère l'indice du cylindre correspondant à la plus grande cellule


  #on recherche ici le barycentre de la cellule de voronoi, pondéré par une fonction de la distance au cylindre le plus proche, afin d'inserer notre cylindre à l'endroit le plus pertinent
  barycentre_pondere = np.zeros(3)
  poids_cellule =  taille_cellules[cellule_max]

  l=[]
  for i in range(n_grid):
    for j in range(n_grid):
      for k in range(n_grid):
        if (attributions[i][j][k][0], attributions[i][j][k][1], attributions[i][j][k][2])== (centres[cellule_max][0], centres[cellule_max][1], centres[cellule_max][2]):
          barycentre_pondere += (distances[i][j][k]**coeff_trou)*np.array([i/n_grid*dimensions_carton[0] + origine_carton[0] ,j/n_grid * dimensions_carton[1] + origine_carton[1], k/n_grid*dimensions_carton[2] + origine_carton[2]])


  barycentre_pondere = barycentre_pondere/poids_cellule

  plus_gros_trou.cellule_max = cellule_max
  plus_gros_trou.card_par_classe = cpt
  plus_gros_trou.arrangement = arrangement
  plus_gros_trou.taille_cellules = taille_cellules
  plus_gros_trou.distances = np.array(l)
  return barycentre_pondere