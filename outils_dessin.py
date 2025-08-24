# -*- coding: utf-8 -*-


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math
import scipy
from arrangement_legitimite import est_legitime


def tracer_cylindre(n_pts, x0, y0, z0, rayon, longueur, axe=[1,0,0]):#npts est la racine carrée du nombre de points
  theta= np.linspace(0,2*np.pi,3*n_pts)
  l = np.linspace(-longueur/2,longueur/2,n_pts)

  #on crée une base orthonormée adaptée au cylindre par gram-schmidt
  epsilon = np.random.normal(0,5, size=3)
  u1=np.array(axe)
  u2=np.array(axe) + epsilon
  epsilon = np.random.normal(0,5, size=3)
  u3=np.array(axe) + epsilon

  e1=u1/(np.linalg.norm(u1))
  e2 = (u2 - np.dot(u2,e1)*e1)/np.linalg.norm(u2 - np.dot(u2,e1)*e1)
  e3 = (u3 - np.dot(u3,e1)*e1 - np.dot(u3,e2)*e2)/np.linalg.norm((u3 - np.dot(u3,e1)*e1 - np.dot(u3,e2)*e2))

  pts=np.zeros((3*n_pts**2,3))
  for i in range(n_pts):
    for j in range(3*n_pts):
      pts[3*i*n_pts+j]= np.array([x0,y0,z0]) + l[i]*e1 + rayon*np.cos(theta[j])*e2 + rayon*np.sin(theta[j])*e3
  return pts


def tracer_carton(dimensions_carton, origine_carton=[0,0,0], n_pts=30):
  points = np.zeros((6*(n_pts**2),3))
  print(np.shape(points))
  x_d = origine_carton[0] # coordonnées au "début" de chaque axe
  y_d = origine_carton[1]
  z_d = origine_carton[2]

  x_f = origine_carton[0]+  dimensions_carton[0]
  y_f = origine_carton[1]+  dimensions_carton[1]
  z_f = origine_carton[2]+  dimensions_carton[2]

  x =np.linspace(x_d,x_f,n_pts)
  y =np.linspace(y_d,y_f,n_pts)
  z =np.linspace(z_d,z_f,n_pts)

  ind=0
  for i in range(n_pts):
    for j in range(n_pts):
      points[ind] = ([x[i],y[j],z_d])
      ind+=1
      points[ind] = [x[i],y[j],z_f]
      ind+=1
      points[ind] = [x[i],y_d,z[j]]
      ind+=1
      points[ind] = [x[i],y_f,z[j]]
      ind+=1
      points[ind] = [x_d,y[i],z[j]]
      ind+=1
      points[ind] = [x_f,y[i],z[j]]
      ind+=1
  return(points)

def tracer_cercle(x0, y0, z0, rayon, longueur, axe=[1,0,0], n_pts=30):
  theta= np.linspace(0,2*np.pi,3*n_pts)
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
  
  u=np.dot(axes_du_plan[0],[x0,y0,z0])
  v=np.dot(axes_du_plan[1],[x0,y0,z0])

  pts=np.zeros((3*n_pts,2))
  for i in range(3*n_pts):
    pts[i]= np.array([u,v]) + rayon*np.cos(theta[i])*np.array([1,0]) + rayon*np.sin(theta[i])*np.array([0,1])

  return pts


def tracer_carton_2d(dimensions_carton, origine_carton, axe=[1,0,0], n_pts=30):
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

  u_0 = np.dot(axes_du_plan[0],origine_carton)
  v_0 = np.dot(axes_du_plan[1],origine_carton)
  print(u_0, v_0)
  points = np.zeros((4*n_pts, 2))
  dimensions_plates = np.array([dimensions_carton[indices_direction_plan[0]], dimensions_carton[indices_direction_plan[1]]])
  ind= 0
  for i in range(n_pts):
    points[ind] = np.array([u_0,v_0]) + dimensions_plates[0]*(i/n_pts)*np.array([1,0]) 
    ind+=1
    points[ind] = np.array([u_0,v_0]) + dimensions_plates[0]*(i/n_pts)*np.array([1,0]) + dimensions_plates[1]*np.array([0,1])
    ind+=1
    points[ind] = np.array([u_0,v_0]) + dimensions_plates[1]*(i/n_pts)*np.array([0,1]) 
    ind+=1
    points[ind] = np.array([u_0,v_0]) + dimensions_plates[1]*(i/n_pts)*np.array([0,1]) + dimensions_plates[0]*np.array([1,0])
    ind+=1
  return points


def tracer_arrangement_2d(arrangement, reglages=None):
  import matplotlib.pyplot as plt

  reglages_defaut = {
      'transparence carton' : 1,
      'transparence cercles' : 1,
      'n_pts_cercles' : 30,
      'n_pts_carton' : 100
  }
  if reglages is None:
    reglages_int = reglages_defaut
  else :
    reglages_int = reglages_defaut.copy()
    reglages_int.update(reglages)

  transparence_carton = reglages_int['transparence carton']
  transparence_cercles = reglages_int['transparence cercles']
  n_pts_cercles = reglages_int['n_pts_cercles']
  n_pts_carton = reglages_int['n_pts_carton']

  centres = arrangement['centres']
  rayon = arrangement['rayon']
  longueur = arrangement['longueur']
  axe = arrangement['axe']

  dimensions_carton = arrangement['dimensions du carton']
  origine_carton = arrangement['origine du carton']

  pts_carton = tracer_carton_2d(dimensions_carton=dimensions_carton, origine_carton=origine_carton, axe=axe, n_pts = n_pts_carton)
  plt.scatter(pts_carton[:,0], pts_carton[:,1], alpha=transparence_carton)
  for centre in centres :
    x0,y0,z0 = centre
    pts_cercle= tracer_cercle(x0, y0, z0, rayon, longueur, axe, n_pts_cercles)
    plt.scatter(pts_cercle[:,0], pts_cercle[:,1], alpha=transparence_cercles)
    
  print('NOM : ', arrangement['nom'])
  print('rayon  = ', rayon)
  print('longueur = ', longueur)
  print('axe = ', axe)
  if est_legitime(arrangement):
    print('cet arrangement est légitime')
  else:
    print('cet arrangement n\'est pas légitime')
  plt.axis('equal')
  plt.show()


import json

def print_matrice(json_path = '/Workspace/Users/theo.duquesne@armor-iimak.com/empilements des cylindres2/armor.json'):

    with open(json_path, 'r') as f:
        binary_list = json.load(f)
    
    for row in binary_list:
        line = ''.join('🐡' if val == 1 else ' ' for val in row)
        print(line)

from PIL import Image
import numpy as np
import json

def image_en_matrice(image_path ='/Workspace/Users/theo.duquesne@armor-iimak.com/empilements des cylindres2/armor.jpg', json_path='/Workspace/Users/theo.duquesne@armor-iimak.com/empilements des cylindres2/armor.json', threshold=200):
    """
    Convertit une image en matrice binaire (0 = blanc, 1 = coloré),
    puis enregistre cette matrice dans un fichier JSON.

    :param image_path: Chemin de l'image (ou un objet PIL.Image.Image).
    :param json_path: Chemin de sauvegarde du fichier JSON.
    :param threshold: Seuil de luminance (0-255) pour considérer un pixel comme blanc.
    """
    # Charger l'image si un chemin est donné
    if isinstance(image_path, str):
        image = Image.open(image_path)
    else:
        image = image_path

    # Convertir en niveaux de gris
    gray_image = image.convert("L")

    # Convertir en tableau numpy
    gray_array = np.array(gray_image)

    # Créer la matrice binaire
    binary_matrix = np.where(gray_array < threshold, 1, 0)

    # Convertir en liste pour JSON
    binary_list = binary_matrix.tolist()

    # Sauvegarder dans un fichier JSON
    with open(json_path, 'w') as f:
        json.dump(binary_list, f)

    print(f"Matrice binaire enregistrée dans {json_path}")


def reduire_resolution_json(fichier_entree = '/Workspace/Users/theo.duquesne@armor-iimak.com/empilements des cylindres2/armor.json', fichier_sortie = '/Workspace/Users/theo.duquesne@armor-iimak.com/empilements des cylindres2/armor.json', p=30, q=30):

  # Lecture
  with open(fichier_entree, 'r') as f:
      data = json.load(f)

  # On suppose que la matrice est une liste de listes de 0/1
  mat = np.array(data)

  # Réduction
  mat_reduite = reduire_resolution(mat, p, q)

  # Conversion en listes classiques (float) pour JSON
  mat_reduite_list = mat_reduite.tolist()

  # Écriture
  with open(fichier_sortie, 'w') as f:
      json.dump(mat_reduite_list, f)


def reduire_resolution(mat, p, q):
    m, n = mat.shape
    h_block = m // p + 1
    w_block = n // q + 1

    res = np.zeros((p, q), dtype=float)

    for i in range(p):
        for j in range(q):
            center_i = int(m * i / p)
            center_j = int(n * j / q)

            start_i = max(0, center_i - h_block // 2)
            end_i = min(m, start_i + h_block)
            start_j = max(0, center_j - w_block // 2)
            end_j = min(n, start_j + w_block)

            if end_i - start_i < h_block:
                start_i = max(0, end_i - h_block)
            if end_j - start_j < w_block:
                start_j = max(0, end_j - w_block)

            block = mat[start_i:end_i, start_j:end_j]
            res[i, j] =  0
            if block.mean() >= 0.5:
                res[i, j] = 1

    return res
