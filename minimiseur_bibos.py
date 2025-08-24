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

def transformer_distances_2_bibos(i,j,arrangement, reglages = None):
    r = arrangement['petit rayon']
    axe = arrangement['axe']
    egg = np.array(projeter_sur_plan(arrangement['centres'][j], axe)) - np.array(projeter_sur_plan(arrangement['centres'][i], axe))
    e = np.linalg.norm(egg)**2
    theta_1 = arrangement['orientations'][i]
    theta_2 = arrangement['orientations'][j]
    ex = np.array([1,0])
    ey = np.array([0,1])
    ur1 = cos(theta_1)*ex + sin(theta_1)*ey
    ur2 = cos(theta_2)*ex + sin(theta_2)*ey
    alpha = np.linalg.norm(ur1-ur2 )**2
    beta = np.dot(egg, ur2-ur1)
    delta = np.dot(egg, ur1)
    epsilon = np.dot(egg, ur2)
    b = 2*beta + 2*r*alpha
    c = e + 2*r*beta + r**2 * alpha - 4 * r**2
    if  (b**2 - 4*c*alpha) >= 0  and alpha !=0: #Si alpha = 0, les deux ont les mêmes orientations, et les deux petits ne pourront pas se toucher, sauf superposition (traitée dans la partie d'après)
        Delta = (b**2 - 4*c*alpha)**0.5
        d1 = (-b - Delta)/(2*alpha)
        d2 = (-b + Delta)/(2*alpha)
        I_1 = Fermes( [[float('-inf'),d1],[d2, float('inf')]]) #petit petit
    else :
        I_1 = Fermes([float('-inf'), float('inf')])
        d1 = float('inf')
        d2 = float('inf')
    if delta >= 0 :  #petit grand
        I_2 = Fermes([float('-inf'),0.5* e /delta - r] )
    else:
        I_2 = Fermes([0.5*e/delta - r, float('inf')])
    if epsilon < 0 : #grand petit 
        I_3 = Fermes([float('-inf'), -0.5*e/epsilon - r])
    else:
        I_3 = Fermes([-0.5*e/epsilon - r, float('inf')])
    I_4 = Fermes([-0.5*np.sqrt(e) ,0.5*np.sqrt(e) ] ) #grand grand
    I = Fermes.intersection([I_1,I_2,I_3,I_4])
    
    s= len(I.bornes)
    if s<= 0 : # il n'y a pas de solution convenable, ie il y a superposition
        d_pp = np.linalg.norm(egg - r*ur1 + r*ur2) - 2*r
        d_pg = np.linalg.norm(egg - r*ur1) - r
        d_gp = np.linalg.norm(egg + r*ur2) - r
        d_gg = np.linalg.norm(egg)
        directions_gradient = [egg, egg + r*ur2 ,egg - r*ur1, egg - r*ur1 + r*ur2]
        directions_gradient = np.array([-u/np.linalg.norm(u) for u in directions_gradient])
        transformer_distances_2_bibos.warning = " ⚠️ Superposition"
        transformer_distances_2_bibos.directions_gradient = directions_gradient
        transformer_distances_2_bibos.I_1 = I_1
        return([d_gg, d_gp, d_pg, d_pp])
    
    if  (b**2 - 4*c*alpha) >= 0 and d2 > min(I_2.bornes[-1]['sup'],I_3.bornes[-1]['sup'],I_4.bornes[-1]['sup']) and d1 < min(I_2.bornes[-1]['sup'],I_3.bornes[-1]['sup'],I_4.bornes[-1]['sup']): # Il faut faire attention, on peut être dans un cas où le rayon maximal autorisé n'est pas le min des rayons maximaux autorisés
        directions_gradient = [egg, egg + r*ur2 ,egg - r*ur1, egg - r*ur1 + r*ur2]
        directions_gradient = np.array([-u/np.linalg.norm(u) for u in directions_gradient])
        transformer_distances_2_bibos.directions_gradient = directions_gradient
        transformer_distances_2_bibos.warning = " Cas chelou  (ʘ‿ʘ🌺) "
        transformer_distances_2_bibos.I_1 = I_1
        return([0.5*np.sqrt(e), I_3.bornes[-1]['sup'], I_2.bornes[-1]['sup'], d1])
    
    else :
        directions_gradient = [egg, egg + r*ur2 ,egg - r*ur1, egg - r*ur1 + r*ur2]
        directions_gradient = np.array([-u/np.linalg.norm(u) for u in directions_gradient])
        transformer_distances_2_bibos.directions_gradient = directions_gradient
        transformer_distances_2_bibos.warning = "Felicitations 🎉, vous êtes dans le cas normal "
        transformer_distances_2_bibos.I_1 = I_1
        return([I_4.bornes[-1]['sup'],I_3.bornes[-1]['sup'],I_2.bornes[-1]['sup'],I_1.bornes[-1]['sup']])

def transformer_distances_bibo_bord(i, dims, arrangement, reglages = None):
    x,y = dims
    r= arrangement['petit rayon']
    axe = arrangement['axe']
    petit_centre_2d = np.array(projeter_sur_plan(arrangement['petits centres'][i], axe))
    grand_centre_2d = np.array(projeter_sur_plan(arrangement['centres'][i], axe))
    theta = arrangement['orientations'][i]
    ex = np.array([1,0])
    ey = np.array([0,1])
    er = cos(theta)*ex + sin(theta)*ey

    #Grand centre
    d_gg = grand_centre_2d[0] # à gauche
    d_gb = grand_centre_2d[1] # en bas
    d_gd = (np.array(dims) - np.array(grand_centre_2d))[0] # à droite
    d_gh = (np.array(dims) - np.array(grand_centre_2d))[1] # en haut

    # à gauche :
    if np.dot(-ex,er) <= 0 :
        d_pg = float('inf')
    else :
        d_pg = (d_gg - r)/np.dot(-ex,er) -r 

    # à droite :
    if np.dot(ex,er) <= 0 :
        d_pd = float('inf')
    else :
        d_pd = (d_gd - r)/np.dot(ex,er) -r 

    # en bas :
    if np.dot(-ey,er) <= 0 :
        d_pb = float('inf')
    else :
        d_pb = (d_gb - r)/np.dot(-ey,er) -r 

    # en haut :
    if np.dot(ey,er) <= 0 :
        d_ph = float('inf')
    else :
        d_ph = (d_gh - r)/np.dot(ey,er) -r 

    directions_gradient  = np.array([ex, ey, -ex, -ey,ex, ey, -ex, -ey ])
    transformer_distances_bibo_bord.directions_gradient = directions_gradient
    return([d_gg, d_gb, d_gd, d_gh, d_pg, d_pb, d_pd, d_ph])
   
def projeter_bords(point, dimensions):
    x, y = point
    L, H = dimensions

    # Projection sur bord bas (y = 0)
    bas = [x, 0]

    # Projection sur bord haut (y = H)
    haut = [x, H]

    # Projection sur bord gauche (x = 0)
    gauche = [0, y]

    # Projection sur bord droit (x = L)
    droite = [L, y]

    return [gauche, bas, droite, haut]

def resol_a_la_main_bibos(centres_orientations, fonction, rayon, petit_rayon, axe, valeur, dims, reglages = None):
    valeurs_intermediaires = []
    n = len(centres_orientations) // 3  # centres_2d est un tableau 1D (x,y,θ pour chaque bibo)

    alpha = 0.8
    gradient_precedent = np.zeros_like(centres_orientations)

    def appliquer_gradient(centres, gradient_effectif, pas, pas_angle):
        resultat = np.copy(centres)
        for j in range(3 * n):
            if j >= 2 * n:  # Composante d’angle
                resultat[j] += (pas_angle * gradient_effectif[j]) 
                resultat[j] = resultat[j] % (2 * pi)
            else:
                resultat[j] += pas * gradient_effectif[j]
        return resultat

    for boucle, (n_tours, pas, pas_angle) in enumerate([(2 * n * 20, 1.0, 1.0),
                                                         (2 * n * 30, 0.3,1),
                                                         (2 * n * 30, 0.03, 1)]):

        for i in range(n_tours):
            perte, gradient = fonction(centres_orientations)

            # Normalisation uniquement sur les 2n premières composantes
            norme_originale = np.linalg.norm(gradient[:2 * n]) + 1e-8
            gradient[:2 * n] /= norme_originale

            gradient_effectif = (1 - alpha) * gradient + alpha * gradient_precedent

            centres_orientations = clip_tab_bibos(centres_orientations, dims, petit_rayon)
            centres_orientations = appliquer_gradient(centres_orientations, gradient_effectif, pas, pas_angle)

            if boucle == 2 and i % n == 0:
                pas *= 0.9

            if i % 20 == 0:
                valeurs_intermediaires.append({
                    'centres': t_2_d_bibos(injection_bibos(np.copy(centres_orientations), axe, valeur)),
                    'rayon': rayon,
                    'code couleur': 0,
                    'texte': 'En cours de rangement',
                    'perte': perte,
                    'iteration du minimiseur': i
                })

    resol_a_la_main.valeurs_intermediaires = valeurs_intermediaires
    return centres_orientations

def clip_tab_bibos(tableau_xy, dimensions, petit_rayon):
    
    tableau = np.array(tableau_xy)
    x_max, y_max = dimensions
    epsilon = 1e-8
    limite = (2 * len(tableau)) // 3  # deux tiers de la longueur
    n=len(tableau) // 3

    for i in range(0, n):
        if tableau[2*i] < epsilon or tableau[2*i] > x_max - epsilon:
            tableau[2*i] = x_max / 2 + random.uniform(-0.1, 0.1)

        if tableau[2*i+1] < epsilon or tableau[2*i+1] > y_max - epsilon:
            tableau[2*i+1] = y_max / 2 + random.uniform(-0.1, 0.1)

        intervalle = intervalles_angles(tableau, i, dimensions, petit_rayon)

        if len(intervalle.bornes) == 0:

            #Si on ne peut pas cliper en déplaçant l'angle, on déplace un peu les rouleaux
            cx,cy =  tableau[2*i], tableau[2*i + 1]
            centre = np.array([x_max/2,y_max/2])
            vect = -np.array([cx, cy]) + centre
            tableau[2*i], tableau[2*i+1] = np.array([cx, cy]) +petit_rayon*vect #0.43 = \sqrt{2} - 1 + petit qquechose

            intervalle = intervalles_angles(tableau, i, dimensions, petit_rayon)
            if len(intervalle.bornes) == 0:
                centres, orientations = t_2_d_bibos(injection_bibos(tableau, [0,0,1], 0.5))
                x,y = dimensions
                dims = x,y,1
                sortie = {
                    'nom' : 'feur', 
                    'centres' : centres,
                    'orientations' : orientations,
                    'dimensions du carton' : dims,
                    'petit rayon' : petit_rayon,
                    'grand rayon' : 0,
                    'axe' : [0,0,1]

                }
                dessiner_gradient(sortie)
                raise Exception('oskour')


        angle = tableau[2*n+ i] %(2*pi)
        if not intervalle.contient(angle):
            intervalle_interdit = Fermes([0,2*pi]) 
            intervalle_interdit.exclusion(intervalle)
            inf = intervalle_interdit.bornes[0]['inf'] 
            if inf ==0 :
                inf = intervalle_interdit.bornes[0]['sup']
                if inf == 2*pi :
                    inf = 0
            sup = intervalle_interdit.bornes[-1]['sup']
            if sup == 2*pi :
                sup = intervalle_interdit.bornes[-1]['inf']
                if sup == 0:
                    sup = 2*pi
            if abs(inf - angle)%(2*pi) < abs(sup - angle)%(2*pi):
                tableau[2*n+i] = inf
                #(intervalle).afficher()
            else:
                #(intervalle).afficher()
                tableau[2*n+i] = sup


    return tableau

def inch_allah_bibos(arrangement, reglages=None):

    # Réglages par défaut
    reglages_par_defaut = {
        'fonction pertes inch': perte_inch_v2_bibos,
        'epsilon': 1,
        'resol_main' : resol_2dir
    }
    if reglages is not None:
        reglages = reglages_par_defaut | reglages
    else:
        reglages = reglages_par_defaut

    perte_inch = reglages['fonction pertes inch']
    solveur = reglages['resol_main']
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
    def a_minimiser(x_2d):
        centres_projetes,orientations_projetees = t_2_d_bibos(injection_bibos(x_2d, axe, point_sert_a_rien))
        arrangement_modifiable['centres'] = centres_projetes
        arrangement_modifiable['orientations'] = orientations_projetees
        perte, gradient = perte_inch(arrangement_modifiable)
        a_minimiser.bibo_min = perte_inch.bibo_min
        a_minimiser.bibos = perte_inch.points
        return perte, gradient

    # Optimisation
    dimensions_proj = projeter_sur_plan(arrangement_modifiable['dimensions du carton'], axe)
    res = solveur(x0, a_minimiser, rayon, petit_rayon, axe, point_sert_a_rien, dimensions_proj, reglages)


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

def intervalles_angles(tableau_xy,i, dimensions, petit_rayon):
    t= tableau_xy 
    x_max, y_max = dimensions
    r = petit_rayon
    d1 = t[2*i]
    d2 = x_max - t[2*i]
    d3 = t[2*i+1]
    d4 = y_max - t[2*i+1]
    theta_1, theta_2, theta_3, theta_4 = 0,0,0,0
    I1, I2, I3, I4 = Fermes({'inf' : 0, 'sup': 2*pi}), Fermes({'inf' : 0, 'sup': 2*pi}), Fermes({'inf' : 0, 'sup': 2*pi}), Fermes({'inf' : 0, 'sup': 2*pi})
    if d1 < 2*r:
        if d1 < 0: 
            print("erreur, d1 = ", d1)
            print('le rouleau est en dehors du carton')
            print('on fait comme si il était à la limite')
            d1 = 0
        theta_1 = acos(d1/r  -1)
        I1 = Fermes([{'inf' : 0, 'sup': pi-theta_1}, {'inf' : pi + theta_1, 'sup': 2*pi } ])
    if d2 <2*r:
        if d2 < 0:
            print("erreur, d2 = ", d2)
            print('le rouleau est en dehors du carton')
            print('on fait comme si il était à la limite')
            d2 = 0
        theta_2 = acos(d2/r -1)
        I2 = Fermes([{'inf' : theta_2, 'sup': 2*pi - theta_2}])
    if d3 < 2*r:
        if d3 < 0:
            print("erreur, d3 = ", d3)
            print('le rouleau est en dehors du carton')
            print('on fait comme si il était à la limite')
            d3 = 0
        theta_3 = acos(d3/r -1)
        I3 = Fermes([{'inf' : 0, 'sup': 1.5*pi-theta_3}, {'inf' : 1.5*pi + theta_3, 'sup': 2*pi } ] )
    if d4 < 2*r:
        if d4 < 0:
            print("erreur, d4 = ", d4)
            print('le rouleau est en dehors du carton')
            print('on fait comme si il était à la limite')
            d4 = 0
        theta_4 =acos(d4/r -1)
        I4 = Fermes([{'inf' : 0, 'sup': .5*pi-theta_4}, {'inf' : .5*pi + theta_4, 'sup': 2*pi } ])
    I = Fermes.intersection([I1,I2,I3,I4])
    
    return(I)

def init_bibo(i):
    bibo = {'indice': i ,
     
     'distance bibo' : float('inf'),

     'distance bord' : float('inf'),       
      
     'distance mandrin' : float('inf'),
      
     'bibos_proches' : 
        [] , 

     'bords_proches' :
        [], 

     'direction gradient bibos' :
        [],

     'direction gradient bords' :
        [],

     'direction gradient mandrin' :
        [] #on ajoutera les np.dot(el, e_theta au fur et à mesure)
         
         }
    return(bibo)

def bibos_proches (bibo, arrangement, reglages = None): 
    reglages_par_defaut = {
        'epsilon_bibo': 3e-3 }
    if reglages is not None:
        reglages = reglages_par_defaut | reglages
    else:
        reglages = reglages_par_defaut
    epsilon = reglages['epsilon_bibo']

    i = bibo['indice']
    axe = arrangement['axe']
    n = len(arrangement['centres'])
    grands_centres = arrangement['centres']
    petits_centres = arrangement['petits centres']
    

    indicatrice_grand_petit = [['g', 'g'], ['g', 'p'], ['p', 'g'], ['p', 'p']] 

    for j in range(n):
        if j != i :
            distances = transformer_distances_2_bibos(i,j,arrangement)
            directions_gradient = transformer_distances_2_bibos.directions_gradient
            for k, dist in enumerate(distances):  #d1,d2,d3,d4, correspond à gg, gp, pg, pp

                if dist <= bibo['distance bibo']+ epsilon :

                    bibo_proche ={
                        'point_int' : None,
                        'est_petit_int' : None,
                        'point_ext' : None,
                        'est_petit_ext' : None,
                       }

                    if  (indicatrice_grand_petit[k][0] == 'p') : # le point du bibo i est le centre du petit cylindre
                        bibo_proche['point_int'] = projeter_sur_plan(petits_centres[i], axe )
                        bibo_proche['est_petit_int'] = True
                        
               
                    else : # le point du bibo i est le centre du grand cylindre
                        bibo_proche['point_int'] = projeter_sur_plan(grands_centres[i], axe)
                        bibo_proche['est_petit_int'] = False

                    if  (indicatrice_grand_petit[k][1] == 'p') : # le point du bibo j est le centre du petit cylindre
                        bibo_proche['point_ext'] = projeter_sur_plan(petits_centres[j], axe )
                        bibo_proche['est_petit_ext'] = True

                    else : # le point du bibo j est le centre du grand cylindre
                        bibo_proche['point_ext'] = projeter_sur_plan(grands_centres[j], axe)
                        bibo_proche['est_petit_ext'] = False


                if dist < bibo['distance bibo'] - epsilon :

                    bibo['bibos_proches'] = [bibo_proche]
                    bibo['distance bibo'] = dist
                    bibo['direction gradient bibos'] = [directions_gradient[k]]

                elif dist <= bibo['distance bibo'] + epsilon :
                    
                    if 'direction gradient bibos' not in bibo :
                        print('distance bibo actuelle : ', bibo['distance bibo'])
                        print('distance bibo proche : ', dist)

                    bibo['bibos_proches'].append(bibo_proche)
                    bibo['direction gradient bibos'].append(directions_gradient[k])


                #On traite à part le cas du mandrin
                if  (indicatrice_grand_petit[k][0] == 'p') :
                    if dist <= bibo['distance mandrin']+ epsilon :
                        bibo['distance mandrin'] = dist

                        if  (indicatrice_grand_petit[k][1] == 'p') : # le point du bibo j est le centre du petit cylindre
                            point_ext = projeter_sur_plan(petits_centres[j], axe )


                        else : # le point du bibo j est le centre du grand cylindre
                            point_ext = projeter_sur_plan(grands_centres[j], axe)
                    
                        theta = arrangement['orientations'][i]
                        e_theta = np.array([-sin(theta), cos(theta)])
                        pseudo_petit_centre = np.array(projeter_sur_plan(grands_centres[i], axe)) + arrangement['petit rayon']*np.array([cos(theta), sin(theta)])
                        el = point_ext - pseudo_petit_centre
                        bibo['direction gradient mandrin'] = -np.dot(el, e_theta)/abs(np.dot(el, e_theta))

def bords_proches(bibo, arrangement, reglages=None):
    reglages_par_defaut = {
        'epsilon_bord': 1e-5
    }
    if reglages is not None:
        reglages = reglages_par_defaut | reglages
    else:
        reglages = reglages_par_defaut
    epsilon = reglages['epsilon_bord']

    i = bibo['indice']
    axe = arrangement['axe']
    carton_proj = projeter_sur_plan(arrangement['dimensions du carton'], axe)
    ptit_centre_i = projeter_sur_plan(arrangement['petits centres'][i], axe)
    grand_centre_i = projeter_sur_plan(arrangement['centres'][i], axe)

    distances = transformer_distances_bibo_bord(i, carton_proj, arrangement, reglages) # GrandGauche GrandBas GrandDroite GrandHaut PetitGauche PetitBas PetitDroite PetitHaut
    directions_gradient = transformer_distances_bibo_bord.directions_gradient
    points_de_projection = projeter_bords(grand_centre_i, carton_proj) + projeter_bords(ptit_centre_i, carton_proj) # Même ordre
    x,y = ptit_centre_i
    a,b = carton_proj
    projete_petit = [[0,y],[x,0], [a, y], [x, b]]
    indicatrice_petits_centres_bords = [False, False, False, False, True, True, True, True]

    for k, dist in enumerate(distances):
        if dist < bibo['distance bord'] - epsilon:
            bibo['distance bord'] = dist
            bibo['bords_proches'] = []
            bibo['direction gradient bords'] = [directions_gradient[k]]

        if abs(dist - bibo['distance bord']) <= epsilon:
            bord_proche = {
                'point_int': None,
                'est_petit_int': None,
                'point_ext': None
            }

            if indicatrice_petits_centres_bords[k]: # Si c'est un petit centre qui est le point le plus proche
                bord_proche['point_int'] = ptit_centre_i
                bord_proche['est_petit_int'] = True
            else:
                bord_proche['point_int'] = grand_centre_i
                bord_proche['est_petit_int'] = False

            bord_proche['point_ext'] = points_de_projection[k]

            bibo['bords_proches'].append(bord_proche)
            bibo['direction gradient bords'].append( directions_gradient[k]) 

        #On traite à part le cas du mandrin
        if  (indicatrice_petits_centres_bords[k]) :
            if dist <= bibo['distance mandrin']+ epsilon :
                bibo['distance mandrin'] = dist
                point_ext = projete_petit[k-4]
                theta = arrangement['orientations'][i]
                e_theta = np.array([-sin(theta), cos(theta)])
                el = point_ext - ptit_centre_i
                bibo['direction gradient mandrin'] = -np.dot(el, e_theta)/abs(np.dot(el, e_theta)+1e-8)

def perte_inch_v2_bibos(arrangement, reglages=None):
    perte_inch_v2_bibos.arrangement = arrangement
    ptit_centres = petits_centres(arrangement)
    arrangement['petits centres'] = ptit_centres
    grands_centres = arrangement['centres']
    n = len(ptit_centres)
    alpha = 0.5

    plus_petite_distance_min = float('inf')
    plus_grande_distance_min = float('-inf')
    bibos = []

    for i in range (0,n):
        bibo_i = init_bibo(i)    
       #on calcule les points les plus proches
        bibos_proches(bibo_i, arrangement, reglages)
        bords_proches(bibo_i, arrangement, reglages)
        bibos.append(bibo_i)
       #calcul de la plus petite et de la plus grande distance minimale
        d= min(bibo_i['distance bibo'], bibo_i['distance bord'])
        if d < plus_petite_distance_min:
            plus_petite_distance_min = d
            bibo_min = i
        if d > plus_grande_distance_min:
            plus_grande_distance_min = d

    
   #Calcul du gradient
    grad1 = np.zeros(3*n)
    eloignement = arrangement['petit rayon'] + arrangement['grand rayon']
    for i in range(0,n):
        bibo_i = bibos[i]
        coeff = plus_grande_distance_min - min(bibo_i['distance bibo'], bibo_i['distance bord'])
        if (bibo_i['distance bibo'] < bibo_i['distance bord']):
            points_proches = bibo_i['bibos_proches']
        else:
            points_proches = bibo_i['bords_proches']

        if bibo_i['distance bibo'] < bibo_i['distance bord']:
            grad = np.mean(bibo_i['direction gradient bibos'], axis = 0)
            grad1[2*i] += grad[0]
            grad1[2*i+1] += grad[1] 
        else : 
            grad = np.mean(bibo_i['direction gradient bords'], axis = 0)
            grad1[2*i] += grad[0] 
            grad1[2*i+1] += grad[1] 

    if np.linalg.norm(grad1) == 0 : #ça peut arriver, par exemple avec deux cylindres
            for i in range(0,n):

                bibo_i = bibos[i]
                if bibo_i['distance bibo'] < bibo_i['distance bord']:
                    grad = np.mean(bibo_i['direction gradient bibos'], axis = 0)
                    grad1[2*i] += grad[0] 
                    grad1[2*i+1] += grad[1]
                else : 
                    grad = np.mean(bibo_i['direction gradient bords'], axis = 0)
                    grad1[2*i] += grad[0]
                    grad1[2*i+1] += grad[1]

    if np.linalg.norm(grad1) == 0:
        raise Exception('⚠️ gradient nul, on doit s\'arrêter')       
    grad1 = (1-alpha)*grad1/np.linalg.norm(grad1)
    
    grad2 = np.zeros(3*n)
    for i in range(n):
        bibo_i = bibos[i]
        coeff = plus_grande_distance_min - min(bibo_i['distance bibo'], bibo_i['distance bord'])
        if (bibo_i['distance bibo'] < bibo_i['distance bord']):
            points_proches = bibo_i['bibos_proches']
        else:
            points_proches = bibo_i['bords_proches']

        #Dans tous les cas, on fait tourner les serviettes. On trouve l'orientation (+ ou - 1, orienté selon le sens trigo) dans le bibo
        grad2[2*n+ i] += 10*bibo_i['direction gradient mandrin'] 
    grad2 = alpha* grad2/np.linalg.norm(grad2)
    gradient = grad1+grad2

    perte_inch_v2_bibos.points = bibos
    perte_inch_v2_bibos.plus_grande_distance_minimale = plus_grande_distance_min
    perte_inch_v2_bibos.bibo_min = bibo_min
    return -plus_petite_distance_min , gradient

class Fermes:
    def __init__(self, borne_inf=None, borne_sup=None):
        if borne_inf is None and borne_sup is None:
            self.bornes = [{'inf': 0, 'sup': 2*pi}]
            self.proprer()
            return

        if isinstance(borne_inf, dict):
            if 'inf' in borne_inf and 'sup' in borne_inf:
                self.bornes = [borne_inf]
                self.proprer()
                return
            else:
                raise ValueError("Le dictionnaire doit contenir les clés 'inf' et 'sup'.")

        if isinstance(borne_inf, list):
            if len(borne_inf) == 2 and all(isinstance(x, (int, float)) for x in borne_inf):
                self.bornes = [{'inf': float(borne_inf[0]), 'sup': float(borne_inf[1])}]
                self.proprer()
                return

            if all(isinstance(x, dict) and 'inf' in x and 'sup' in x for x in borne_inf):
                self.bornes = borne_inf
                self.proprer()
                return

            if all(isinstance(x, list) and len(x) == 2 and all(isinstance(v, (int, float)) for v in x) for x in borne_inf):
                self.bornes = [{'inf': float(pair[0]), 'sup': float(pair[1])} for pair in borne_inf]
                self.proprer()
                return
        

        raise TypeError("Entrée invalide : attendu None, un dict, une liste [a,b], ou une liste de listes/dicts.")

    def proprer(self):
        # Trie les intervalles
        self.bornes.sort(key=lambda borne: borne['inf'])
        sortie = []

        while self.bornes:
            cc = self.bornes.pop(0)
            borne_inf_cc = cc['inf']
            borne_sup_cc = cc['sup']
            i = 0
            while i < len(self.bornes):
                borne = self.bornes[i]
                if borne['inf'] <= borne_sup_cc:  # Recouvrement ou contact
                    borne_inf_cc = min(borne_inf_cc, borne['inf'])
                    borne_sup_cc = max(borne_sup_cc, borne['sup'])
                    self.bornes.pop(i)
                else:
                    i += 1
            sortie.append({'inf': borne_inf_cc, 'sup': borne_sup_cc})
        self.bornes = sortie

    @staticmethod
    def union(liste_fermes):
        resultat = Fermes()
        liste = []
        for ferme in liste_fermes:
            liste+=(ferme.bornes)
        resultat.bornes = liste
        resultat.proprer()
        return resultat

    def __repr__(self):
        return f"Fermes({self.bornes})"
      
    def intersection_2_elements(self, autre):
      bornes_self = self.bornes
      bornes_autre= autre.bornes
      bornes_final = []
      for borne_s in bornes_self :
        for borne_a in bornes_autre :
          b_a_i = borne_a['inf']
          b_a_s = borne_a['sup']
          b_s_i = borne_s['inf']
          b_s_s = borne_s['sup']
          b_t_i = max(b_a_i, b_s_i)
          b_t_s = min(b_a_s, b_s_s)
          if not b_t_i > b_t_s:
            bornes_final.append({'inf' : b_t_i, 'sup' : b_t_s})
      self.bornes = bornes_final
      self.proprer()
      
    @staticmethod
    def intersection(liste):
      liste = copy.deepcopy(liste)
      if not liste :
        print('Erreur : aucun élément dans la liste d\'éléments à intersecter')
      sortie = liste.pop(0)
      while(liste):
        autre = liste.pop(0)
        sortie.intersection_2_elements(autre)
      return(sortie)
    
    def afficher(self):
      self.proprer()
      l = self.bornes.copy()
      chaine = 'Ce fermé est : '
      flag = True
      for borne in  l :
        if flag:
          flag = False
        else:
          chaine += '\u222A'
        i = borne['inf']
        s= borne['sup']
        chaine+= '[' + str(i) + ',' + str(s)+ ']'
      print(chaine)
    
    def contient(self, a):
        for borne in self.bornes:
            if borne['inf'] <= a <= borne['sup']:
                return True
        return False
    
    def exclusion(self, autre):
        resultat = []
        for borne in self.bornes:
            inf1, sup1 = borne['inf'], borne['sup']
            parties = [{'inf': inf1, 'sup': sup1}]
            for excl in autre.bornes:
                inf2, sup2 = excl['inf'], excl['sup']
                nouvelles_parties = []
                for p in parties:
                    a, b = p['inf'], p['sup']
                    if sup2 <= a or inf2 >= b:
                        nouvelles_parties.append({'inf': a, 'sup': b})
                    else:
                        if a < inf2:
                            nouvelles_parties.append({'inf': a, 'sup': min(inf2, b)})
                        if b > sup2:
                            nouvelles_parties.append({'inf': max(sup2, a), 'sup': b})
                parties = nouvelles_parties
            resultat.extend(parties)
        self.bornes = resultat
        self.proprer()

def dessiner_gradient(arrangement, reglages=None, beta=0.1, fleches = True):
    reglages_par_defaut = {'fonction de perte inch' : perte_inch_v2_bibos}
    if reglages is not None :
        reglages = reglages_par_defaut | reglages
    else :
        reglages = reglages_par_defaut
    fonction_perte_inch = reglages["fonction de perte inch"] 
    import numpy as np
    import matplotlib.pyplot as plt
    from matplotlib.patches import Circle, Rectangle
    if not 'nom' in arrangement :
        arrangement['nom'] = 'arrangement anonyme'

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

    # === Projection des centres en 2D ===
    centres_grands_2d = [projeter_sur_plan(c, axe) for c in centres_grands]
    centres_petits_2d = [projeter_sur_plan(c, axe) for c in centres_petits]

    # === Calcul du gradient ===
    _, gradient = fonction_perte_inch(arrangement, reglages)
    gradient = gradient / (np.linalg.norm(gradient) + 1e-8)
    n = len(centres_grands)

    # === Création de la figure ===
    fig, ax = plt.subplots(figsize=(6, 4), dpi=80)
    ax.set_aspect('equal')

    # Limites du graphique
    if axe[0] == 1:
        ax.set_xlim(0, dimensions_carton[1])
        ax.set_ylim(0, dimensions_carton[2])
    elif axe[1] == 1:
        ax.set_xlim(0, dimensions_carton[0])
        ax.set_ylim(0, dimensions_carton[2])
    else:
        ax.set_xlim(0, dimensions_carton[0])
        ax.set_ylim(0, dimensions_carton[1])

    # === Tracé de l'arrangement ===
    ajouter_carton(ax, dimensions_carton, axe)
    ajouter_disques(ax, centres_grands, rayon_grand, axe, couleur='blue', alpha=0.6)
    ajouter_disques(ax, centres_petits, rayon_petit, axe, couleur='orange', alpha=0.6)

    # === Flèches du gradient + numéros ===
    for i in range(n):
        x, y = centres_grands_2d[i]
        dx = beta * gradient[2 * i]
        dy = beta * gradient[2 * i + 1]
        if fleches :
            ax.arrow(x, y, dx, dy,
                    head_width=0.2 * rayon_grand, head_length=0.3 * rayon_grand,
                    fc='red', ec='red', length_includes_head=True)
        # Affichage du numéro du centre
        ax.text(x, y, str(i), fontsize=9, ha='center', va='center', color='white', weight='bold')

    # === Flèches du gradient orientation (petits centres) ===
    for i in range(n):
        petit_centre_i = centres_petits[i]
        theta_i = orientations[i]
        e_theta = -np.sin(theta_i) * np.array([1, 0]) + np.cos(theta_i) * np.array([0, 1])
        base_2d = projeter_sur_plan(petit_centre_i, axe)
        dir_2d = e_theta
        intensite = beta * gradient[2 * n + i]
        if intensite != 0:
            dx = intensite * dir_2d[0]
            dy = intensite * dir_2d[1]
            if fleches : 
                ax.arrow(base_2d[0], base_2d[1], dx, dy,
                        head_width=0.2 * rayon_petit, head_length=0.3 * rayon_petit,
                        fc='purple', ec='purple', length_includes_head=True)

    # === Affichage ===
    plt.title(f"{arrangement['nom']} avec gradient")
    plt.tight_layout()
    plt.show()

def afficher_points(liste_de_points, arrangement=None):
    def arrondir(pt):
        return [round(float(x), 3) for x in pt]

    def bool_str(b):
        return "petit" if b else "grand"

    for i, bibo in enumerate(liste_de_points):
        print(f"\n=== Bibo {bibo['indice']} ===")

        if arrangement is not None:
            axe = arrangement['axe']
            centre_proj = projeter_sur_plan(arrangement['centres'][i], axe)
            orientation_i = arrangement['orientations'][i]
            print(f"  Centre projeté : {arrondir(centre_proj)}")
            print(f"  Orientation : {round(orientation_i, 3)}")

        print(f"  Distance bibo : {round(bibo['distance bibo'], 3)}")
        print(f"  Distance bord : {round(bibo['distance bord'], 3)}")

        print("\n  Bibos proches :")
        if bibo['bibos_proches']:
            for point in bibo['bibos_proches']:
                p_int = arrondir(point['point_int'])
                p_ext = arrondir(point['point_ext'])
                print(f"    - point_int : {p_int} ({bool_str(point['est_petit_int'])})")
                print(f"      point_ext : {p_ext} ({bool_str(point['est_petit_ext'])})")
        else:
            print("    (aucun)")

        print("\n  Bords proches :")
        if bibo['bords_proches']:
            for point in bibo['bords_proches']:
                p_int = arrondir(point['point_int'])
                p_ext = arrondir(point['point_ext'])
                print(f"    - point_int : {p_int} ({bool_str(point['est_petit_int'])})")
                print(f"      point_ext : {arrondir(p_ext)}")
        else:
            print("    (aucun)")

def resol_main_moins_foireuse(centres_orientations, fonction, rayon, petit_rayon, axe, valeur, dims, reglages = None):
  reglages_par_defaut= {
    'alpha resol': 0.9, #change strictement de direction 3 iterations après s'être cogné contre le mur
    'beta resol' : 0.7937,
    'n_dernieres_it' :10,
    'stab' : 0.1,
    'epsilon_arret' : 0.001,
    'gamma resol': 0,
    'gamma resol angle': 0,
    'decalage accepte angle': 0.03,
    'decalage accepte position': 0.05

  }
  if reglages is None:
    reglages = reglages_par_defaut
  else :
    reglages = reglages | reglages_par_defaut

  historique = []
  cpt = 0
  n = len(centres_orientations) // 3  # centres_2d est un tableau 1D (x,y,θ pour chaque bibo)

 #Update_histo
  def update_histo(gradient, histo):
    while len(histo) >= reglages['n_dernieres_it']:
      histo.pop(0)
    histo.append(gradient)
    var = np.var(histo, axis= 1)
    moy = np.mean(histo, axis=0)
    update_histo.variance_gradient = var
    update_histo.moyenne_gradient = moy

 #Condition_arret
  def condition_arret(perte,cpt):
    arret_perte = False
    arret_gradient = False
    # le gradient nous fait tourner en rond :
    if np.linalg.norm(update_histo.moyenne_gradient) <= reglages['stab'] :
      arret_gradient = True
    # les distances sont toutes equivalentes :
    ppdm = -perte #plus petite distance minimale entre des objets
    pgdm = perte_inch_v2_bibos.plus_grande_distance_minimale
    d= pgdm - ppdm
    if d <= reglages['epsilon_arret']:
      arret_perte = True
    condition_arret.arret_gradient = update_histo.moyenne_gradient
    condition_arret.tours = cpt
    arret = arret_gradient and cpt >= reglages["n_dernieres_it"]
    return(arret)

 #Calcul_pas
  def calcul_pas(gradient):
      # --- Direction inertielle unitaire ---
      if not hasattr(calcul_pas, 'direction'):
          direction = gradient / (np.linalg.norm(gradient) + 1e-8)
      else:
          α = reglages['alpha resol']
          raw = (1 - α) * gradient + α * calcul_pas.direction
          direction = raw / (np.linalg.norm(raw) + 1e-8)
      calcul_pas.direction = direction

      # --- Gradient précédent stocké dans prev_grad ---
      prev_grad = getattr(calcul_pas, 'prev_grad', np.zeros_like(gradient))
      delta = gradient - prev_grad
      calcul_pas.prev_grad = gradient.copy()

      # --- Séparation ---
      n_pos = 2 * n
      grad_pos, grad_ang = delta[:n_pos], delta[n_pos:]

      # --- Pas précédents ---
      p0 = getattr(calcul_pas, 'pas_precedent_pos', 1.0)
      pa = getattr(calcul_pas, 'pas_precedent_ang', 1.0)

      # --- Calcul des nouveaux pas ---
      base_pos = reglages['decalage accepte position']
      base_ang = reglages['decalage accepte angle']
      β = reglages['beta resol']

      norm_dp = np.linalg.norm(grad_pos) + 1e-8
      norm_da = np.linalg.norm(grad_ang) + 1e-8

      # Pas proposés sans inertie
      nouveau_pos = base_pos * p0 / norm_dp
      nouveau_ang = base_ang * pa / norm_da

      # Application de l'inertie sur la longueur du pas
      pas_pos = (1 - β) * nouveau_pos + β * p0
      pas_ang = (1 - β) * nouveau_ang + β * pa

      # Clamp pour éviter les valeurs extrêmes
      pas_pos = np.clip(pas_pos, reglages.get('pas_min', 1e-4), reglages.get('pas_max', 0.05))
      pas_ang = np.clip(pas_ang, reglages.get('pas_min_ang', 1e-4), reglages.get('pas_max_ang', 0.05))

      # Sauvegarde
      calcul_pas.pas_precedent_pos = pas_pos
      calcul_pas.pas_precedent_ang = pas_ang

      # --- Sortie ---
      pas_total = np.concatenate((
          pas_pos * direction[:n_pos],
          pas_ang * direction[n_pos:]
      ))
      return pas_total

 #descendre
  def descendre(centres, pas):
      resultat = np.copy(centres)
      resultat += pas
      for j in range(3 * n):
          if j >= 2 * n:  # Composante d’angle
              resultat[j] = resultat[j] % (2 * pi)
      return resultat

 # Boucle
  arret =False
  while not arret:
    perte, gradient = fonction(centres_orientations)
    gradient = gradient/np.linalg.norm(gradient)
    pas = calcul_pas(gradient)
    update_histo(gradient, historique)
    centres_orientations = descendre(centres_orientations, pas)
    arret = condition_arret(perte, cpt)
    cpt+= 1 
    if hasattr(sauvegarde, 'historique'):
      infos = {
          'centres_orientations':centres_orientations,
          'variance gradient': np.linalg.norm(update_histo.variance_gradient),
          'moyenne gradient': np.linalg.norm(update_histo.moyenne_gradient),
          'pas positions': calcul_pas.pas_precedent_pos,
          'pas angles': calcul_pas.pas_precedent_ang,

      }
      sauvegarde.historique.append(infos)

  return (centres_orientations )
    
def resol_main_autre(centres_orientations, fonction, rayon, petit_rayon, axe, valeur, dims, reglages = None):
 # Initialisation et réglages
  reglages_par_defaut= {
    'alpha resol':0.1, # alpha gros => faible inertie
    'beta resol' : 0.7937,
    'n_dernieres_it' :10,
    'stab' : 0.1,
    'epsilon_arret' : 0.001,
    'gamma resol': 0,
    'gamma resol angle': 0,
    'decalage accepte angle': 0.01,
    'decalage accepte position': 0.01,
    'pas_min' :0.001,
    'pas_max' : 0.01

  }
  if reglages is None:
    reglages = reglages_par_defaut
  else :
    reglages = reglages | reglages_par_defaut

  historique = []
  cpt = 0
  n = len(centres_orientations) // 3  # centres_2d est un tableau 1D (x,y,θ pour chaque bibo)

 #  Mise à jour de l'historique 
  def update_histo(gradient, histo):
    while len(histo) >= reglages['n_dernieres_it']:
      histo.pop(0)
    histo.append(gradient)
    var = np.var(histo, axis= 1)
    moy = np.mean(histo, axis=0)
    update_histo.variance_gradient = var
    update_histo.moyenne_gradient = moy

 # Condition d'arrêt
  def condition_arret(perte,cpt):
    arret_perte = False
    arret_gradient = False
    # le gradient nous fait tourner en rond :
    if np.linalg.norm(update_histo.moyenne_gradient) <= reglages['stab'] :
      arret_gradient = True
    # les distances sont toutes equivalentes :
    ppdm = -perte #plus petite distance minimale entre des objets
    pgdm = perte_inch_v2_bibos.plus_grande_distance_minimale
    d= pgdm - ppdm
    if d <= reglages['epsilon_arret']:
      arret_perte = True
    condition_arret.arret_gradient = update_histo.moyenne_gradient
    condition_arret.tours = cpt
    arret = arret_gradient and cpt >= reglages["n_dernieres_it"]
    return(arret)

 # Calcul du pas
  def calcul_pas(gradient):
    n = gradient.size // 3
    alpha = reglages['alpha resol']
    base = reglages['decalage accepte position']

    if not hasattr(calcul_pas, 'stockage'):
        calcul_pas.stockage = []


    stockage = calcul_pas.stockage
    nouvelle_stockage = []

    # --- Traitement des positions (2D) ---
    for i in range(n):
        idx = 2 * i
        g = gradient[idx:idx + 2]

      #récupération des infos dans stockage
        if len(stockage) >= 2 * n:
            entree = stockage[i]
            g_prec = entree['gradient']
            pas_lisse_prec = entree['pas lisse']
            long_prec = np.linalg.norm(pas_lisse_prec) + 1e-8
        else:
            g_prec = np.zeros_like(g)
            long_prec = reglages['pas_min']
            pas_lisse_prec = np.zeros_like(g)

      #calcul du pas de position
        diff = g - g_prec
        norme_diff = np.linalg.norm(diff) + 1e-8
        long_brut = base * long_prec / norme_diff
        long_brut = np.clip(long_brut, reglages.get('pas_min', 1e-4), reglages.get('pas_max', 0.05))
        direction_lissee = (1 - alpha) * g_prec + alpha * g
        direction_lissee = direction_lissee / (np.linalg.norm(direction_lissee)+1e-8)
        pas_lisse = long_brut * direction_lissee #on s'assure d'avoir une longueur interessante. Sinon, si le gradient varie bcp, on penalise deux fois et on est quasi nul. On a déjà long_brute pour p&naliser si on est dans une zone de forte variations du gradient

      # Réécriture du stockage
        nouvelle_stockage.append({
            'gradient': g.copy(),
            'gradient precedent': g_prec.copy(),
            'longueur pas brute': long_brut,
            'longueur pas brute precedente': long_prec,
            'pas lisse': pas_lisse.copy(),
            'pas lisse precedent': pas_lisse_prec.copy(),
        })

    # --- Traitement des angles (1D) ---
    for i in range(n):
        idx = 2 * n + i
        g = gradient[idx:idx + 1]

       # Récupération des informations dans le stockage 
        if len(stockage) >= 2 * n:
            entree = stockage[n + i]
            g_prec = entree['gradient']
            pas_lisse_prec = entree['pas lisse']
            long_prec = np.linalg.norm(pas_lisse_prec)
        else:
            g_prec = np.zeros_like(g)
            long_prec = reglages['pas_min']
            pas_lisse_prec = np.zeros_like(g)

      #calcul du pas d'angle
        diff = g - g_prec
        norme_diff = np.linalg.norm(diff) + 1e-8
        long_brut = base * long_prec / norme_diff
        long_brut = np.clip(long_brut, reglages.get('pas_min', 1e-4), reglages.get('pas_max', 0.05))
        direction_lissee = (1 - alpha) * g_prec + alpha * g
        direction_lissee = direction_lissee / (np.linalg.norm(direction_lissee)+1e-8)
        pas_lisse = long_brut * direction_lissee


      # Réécriture du stockage
        nouvelle_stockage.append({
            'gradient': g.copy(),
            'gradient precedent': g_prec.copy(),
            'longueur pas brute': long_brut,
            'longueur pas brute precedente': long_prec,
            'pas lisse': pas_lisse.copy(),
            'pas lisse precedent': pas_lisse_prec.copy(),
        })

    calcul_pas.stockage = nouvelle_stockage

 # Ajustement du pas (pour eviter de trop osciller)
  def ajuster_pas(centres):
    stockage = calcul_pas.stockage
    n = len(stockage) // 2
    epsilon = reglages['epsilon_arret']

    perte_actuelle, _ = fonction(centres)

    while True:
        centres_essai = descendre(centres)
        perte_nouvelle, _ = fonction(centres_essai)

        if perte_nouvelle <= perte_actuelle:
            break

        i = perte_inch_v2_bibos.bibo_min

        # Division par deux des pas lissés
        stockage[i]['pas lisse'] *= 0.5
        stockage[n + i]['pas lisse'] *= 0.5

        # Si l’un des deux pas devient trop petit, on arrête
        if (np.linalg.norm(stockage[i]['pas lisse']) < epsilon or
            np.linalg.norm(stockage[n + i]['pas lisse']) < epsilon):
          stockage[i]['pas lisse'] = np.array([0,0])
          stockage[n + i]['pas lisse'] = np.array([0])
          break

 # Descente d'un pas
  def descendre(centres):
    stockage = calcul_pas.stockage
    n = len(stockage) // 2

    pas_total = np.zeros_like(centres)

    # Positions
    for i in range(n):
        idx = 2 * i
        pas_total[idx:idx + 2] = stockage[i]['pas lisse']

    # Angles
    for i in range(n):
        idx = 2 * n + i
        pas_total[idx] = stockage[n + i]['pas lisse']

    resultat = centres + pas_total

    # Réduction des angles
    for j in range(2 * n, 3 * n):
        resultat[j] = resultat[j] % (2 * pi)

    return resultat
  
 # Boucle
  arret =False
  while not arret:
    perte, gradient = fonction(centres_orientations)
    gradient = gradient/np.linalg.norm(gradient)
    calcul_pas(gradient)
    ajuster_pas(centres_orientations)
    update_histo(gradient, historique)
    centres_orientations = descendre(centres_orientations)
    arret = condition_arret(perte, cpt)
    cpt+= 1 

    # Sauvegarde des informations
    if hasattr(sauvegarde, 'historique'):
      infos = {
          'tour' : cpt,
          'centres_orientations': centres_orientations,
          'variance gradient': np.linalg.norm(update_histo.variance_gradient),
          'moyenne gradient': np.linalg.norm(update_histo.moyenne_gradient),
          'pas positions': np.mean([calcul_pas.stockage[i]['longueur pas brute'] for i in range(n)]),
          'pas angles': np.mean([calcul_pas.stockage[i+n]['longueur pas brute'] for i in range(n)]),
          'stockage' : copy.deepcopy(calcul_pas.stockage)

      }
      sauvegarde.historique.append(infos)
      resol_main_autre.stockage = calcul_pas.stockage

 # Return
  return (centres_orientations )

def resol_main_deux_directions_ancien(fonction, derivee, centres_orientations, reglages, sauvegarde):
    reglages_par_defaut= {
    'inertie':0.1, # 
    'n_dernieres_it' :10,
    'stab' : 0.1,
    'epsilon_arret' : 0.001,
    'decalage accepte': 0.01,
    'pas_min' :0.001,
    'pas_max' : 0.01

  }
    if reglages is None :
      reglages = reglages_par_defaut
    else:
      reglages = reglages_par_defaut | reglages

    pas_max = reglages["pas_max"]
    epsilon_arret = reglages["epsilon_arret"]
    decalage_accepte = reglages["decalage accepte"]
    inertie = reglages["inertie"]

    n = centres_orientations.shape[0] // 3
    stockage = [{} for _ in range(2 * n)]

    update_histo = type('', (), {})()
    update_histo.moyenne_gradient = np.zeros_like(centres_orientations)
    update_histo.variance_gradient = np.zeros_like(centres_orientations)

    def normalise(v):
        norme = np.linalg.norm(v)
        if norme < 1e-12:
            return np.zeros_like(v)
        return v / norme

    def get_sous_vecteurs(vecteur):
        return [vecteur[2 * i:2 * i + 2] for i in range(n)] + [vecteur[2 * n + i:2 * n + i + 1] for i in range(n)]

    def set_sous_vecteurs(sous_vecteurs):
        return np.concatenate(sous_vecteurs)

    def condition_arret(moyenne_gradient):
        return np.linalg.norm(moyenne_gradient) < epsilon_arret

    def met_a_jour_moyenne_et_variance(nouveau_gradient):
        diff = nouveau_gradient - update_histo.moyenne_gradient
        update_histo.moyenne_gradient += diff / (cpt + 1)
        update_histo.variance_gradient += diff * (nouveau_gradient - update_histo.moyenne_gradient)

    def tester_direction(direction, centres_actuels):
        longueur = pas_max
        while longueur > epsilon_arret:
            test = centres_actuels + longueur * direction
            perte_test = fonction(test)
            if perte_test < perte:
                return test, perte_test, longueur
            longueur /= 2
        return centres_actuels, perte, 0.0

    cpt = 0
    while True:
        gradient = derivee(centres_orientations)
        sous_gradients = get_sous_vecteurs(gradient)
        directions = []

        for i, g in enumerate(sous_gradients):
            g_norm = normalise(g)
            stockage[i]["gradient"] = g_norm
            if "direction inertielle" in stockage[i]:
                direction_lissee = normalise((1 - inertie) * g_norm + inertie * stockage[i]["direction inertielle"])
            else:
                direction_lissee = g_norm.copy()
            stockage[i]["direction inertielle"] = direction_lissee
            directions.append(g_norm)
            directions.append(direction_lissee)

        directions_1 = directions[:2 * n:2]
        directions_2 = directions[1:2 * n:2]

        dir1 = set_sous_vecteurs(directions_1)
        dir2 = set_sous_vecteurs(directions_2)

        perte = fonction(centres_orientations)
        point1, perte1, longueur1 = tester_direction(dir1, centres_orientations)
        point2, perte2, longueur2 = tester_direction(dir2, centres_orientations)

        if perte1 < perte2:
            direction_choisie = dir1
            centres_intermediaire = point1
            perte_inter = perte1
            longueur = longueur1
        else:
            direction_choisie = dir2
            centres_intermediaire = point2
            perte_inter = perte2
            longueur = longueur2

        gradient_suivant = derivee(centres_intermediaire)
        diff_gradient = gradient - gradient_suivant
        norme_diff = np.linalg.norm(diff_gradient)

        if norme_diff < 1e-12:
            pas_final = 0.0
        else:
            pas_final = pas_max * decalage_accepte / norme_diff

        direction_finale = normalise(direction_choisie)
        centres_orientations = centres_orientations - pas_final * direction_finale

        nouveau_gradient = derivee(centres_orientations)
        met_a_jour_moyenne_et_variance(nouveau_gradient)

        nouveaux_sous_gradients = get_sous_vecteurs(nouveau_gradient)
        for i, g in enumerate(nouveaux_sous_gradients):
            stockage[i]["longueur pas brut"] = pas_final
            stockage[i]["gradient"] = normalise(g)

        sauvegarde.historique.append({
            'tour': cpt,
            'centres_orientations': centres_orientations.copy(),
            'variance gradient': np.linalg.norm(update_histo.variance_gradient),
            'moyenne gradient': np.linalg.norm(update_histo.moyenne_gradient),
            'pas positions': np.mean([stockage[i]["longueur pas brut"] for i in range(n)]),
            'pas angles': np.mean([stockage[i + n]["longueur pas brut"] for i in range(n)]),
            'stockage': copy.deepcopy(stockage)
        })

        if condition_arret(update_histo.moyenne_gradient):
            break

        cpt += 1

    return centres_orientations

"""def resol_main_deux_directions_brouillon(centres_orientations, fonction, rayon, petit_rayon, axe, valeur, dims, reglages = None):
 # Gestion des reglages
  reglages_par_defaut= {
    'inertie':0.1, # 
    'n_dernieres_it' :10,
    'stab' : 0.1,
    'epsilon_arret' : 0.001,
    'decalage accepte': 0.01,
    'pas_min' :0.001,
    'pas_max' : 0.01

  }
  if reglages is None :
    reglages = reglages_par_defaut
  else:
    reglages = reglages_par_defaut | reglages
 
  
  n = centres_orientations.shape[0] // 3
 # Fonctions internes
  def init_stockage():
    direction_pos_i = {
      'gradient courant' : np.zeros(2),
      'ancien gradient' : np.zeros(2),
      'longueur pas' : 0.0,
      'proposition direction inertielle' : np.zeros(2),
      'proposition direction gradient' : np.zeros(2),
      'coordonnees proposition inertielle' : np.zeros(2),
      'gradient proposition inertielle': np.zeros(2),
      'gradient proposition gradient' : np.zeros(2),
      'coordonnees proposition gradient' : np.zeros(2),
      'deplacement inertiel' : np.zeros(2)
    }
    direction_ang_i = {
      'gradient courant' : np.zeros(1),
      'ancien gradient' : np.zeros(1),
      'longueur pas' : 0.0,
      'proposition direction inertielle' : np.zeros(1),
      'proposition direction gradient' : np.zeros(1),
      'pas proposition inertielle' : reglages['pas_max'],
      'pas proposition gradient' : reglages['pas_max'],
      'gradient proposition inertielle': np.zeros(1),
      'gradient proposition gradient' : np.zeros(1),
      'coordonnees proposition inertielle' : np.zeros(1),
      'coordonnees proposition gradient' : np.zeros(1),
      'deplacement inertiel' : np.zeros(1)
    }
    stockage = [direction_pos_i for i in range(n)] + [direction_ang_i for i in range(n)]
    resol.stockage = stockage

  def update_histo():
    #garde en mémoire les n_dernieres_it valeurs de gradients (et eventuellement de positions). Calcule des indicateurs (comme la moyenne et variance) de ces dernières positions et/ou des derniers gradients et les garde en mémoire dans des attributs tels que update_histo.historique, update_histo.moyenne...

  def calculer_directions():
    # si resol.premier_tour est vrai, alors on calcule les directions inertielles et de gradient de la manière suivante : 
    #perte, gradient = fonction(centres_orientations)
    #ensuite, on tronçonne en sous vecteurs, que l'on normalise et que l'on ajoute à stockage dans perte courante et gradient courant
    #de même, on ajoute la perte dans resol.perte_courante
    #sinon, on peut les récupérer dans stockage à gradient courant et deplacement inertiel .
    #on met à jour les champs propositions direction inertielle et gradient dans stockage 

  def tester_directions():
    #récupère les propositions de directions inertielles et gradient dans resol.stockage, évalue leur gradient et leur perte grâce à fonction(), fais un pas de longueur pas_max dans cette direction, calcul la perte et le gradient dans cette direction . Cette étape concerne tous les vecteurs à la fois.
    # Ensuite, on effectue les calculs habituels (en y ajoutant la stratégie hybride) pour choisir la longueur du pas. Cette étape se fait sous vecteur par sous vecteur
    # Une fois les calculs fait, on les (gradients, coordonnées et pertes des propositions) stocke dans stockage, avec les clé "... proposition inertielle/gradient" qui sont assez explicites

  def descendre(centres_orientations):
    #choisit la direction qui fait le plus baisser la perte 
    #va dans cette direction (vecteur par vecteur, chacun de la longueur "longueur pas" associée)
    #met à jour le stockage:
    #-(ancien gradient prend la valeur de gradient courant)
    #-déplacement inertiel est mis à alpha*deplacement_choisit + (1-alpha)*deplacement inertiel
    # gradient courant prend la valeur de gradient proposition inertielle/gradient suivant la direction choisie
    #perte courante prend la valeur de perte proposition inertielle/gradient suivant la direction choisie
    return centres_orientations_updated

  def maj_historique():
    if hasattr(sauvegarde,'historique'):
      sauvegarde.historique.append({
            'tour': cpt,
            'centres_orientations': centres_orientations.copy(),
            'variance gradient': np.linalg.norm(update_histo.variance_gradient),
            'moyenne gradient': np.linalg.norm(update_histo.moyenne_gradient),
            'pas positions': np.mean([stockage[i]["longueur pas"] for i in range(n)]),
            'pas angles': np.mean([stockage[i + n]["longueur pas"] for i in range(n)]),
            'stockage': copy.deepcopy(stockage)
        })
      
  def conditions_arret():
    moyenne_gradient = update_histo.moyenne_gradient
    return moyenne_gradient < reglages['epsilon_arret'] and cpt >= reglages['n_dernieres_it']


 # Initialisation des variables avant la boucle
  resol.premier_tour = True
  init_stockage()
  cpt = 0
  arret = False
  cent_or  = centres_orientations.copy()
 ### Boucle principale
  while not arret:
   # On calcul les deux directions concurrentes par lesquelles on pourrait descendre.
    calcul_directions()
   # On teste chacune des directions afin de savoir laquelle est la meilleure
    tester_directions()
   # On descend dans la meilleure direction
    cent_or = descendre(cent_or)
   # On met à jour tous les indicateurs
    update_histo()
    maj_historique()
    cpt += 1
    arret = conditions_arret()
    resol.premier_tour = False

 # On renvoie la solution finale ainsi que d'éventuels attributs
  return cent_or """

def resol_main_deux_directions(centres_orientations, fonction, rayon, petit_rayon, axe, valeur, dims, reglages=None):
   # Gestion des reglages
    reglages_par_defaut = {
        'alpha': 0.1, #alpha faible => inertie forte
        'n_dernieres_it': 10,
        'stab': 0.1,
        'epsilon_arret': 0.001,
        'decalage accepte': 0.05,
        'pas_min': 0.001,
        'pas_max': 0.01,
        'acceleration' : 2
    }
    if reglages is None:
        reglages = reglages_par_defaut
    else:
        reglages = reglages_par_defaut | reglages

    n = centres_orientations.shape[0] // 3
    if hasattr(sauvegarde,'infos_arrangement'):
      sauvegarde.infos_arrangement = {'petit rayon' : petit_rayon,
         'axe' : axe,
         'valeur' : valeur,
         'dimensions' : dims}
      
   # Variables pour l’historique des gradients
    resol = {'je ne suis là que pour stocker des attributs'}
    cent_or = np.copy(centres_orientations)
    
   # Pour mesurer le temps des sous-fonctions
    resol = type('', (), {})()
    resol.stockage = None
    resol.perte_courante = None
    resol.perte_prop_inert = None
    resol.perte_prop_grad = None
    resol.premier_tour = True
    resol.tempos = {
        'calcul_directions': 0,
        'tester_directions': 0,
        'descendre': 0,
        'update_histo': 0,
        'maj_historique': 0
    }

   # Definition de fonctions internes
    def init_stockage():
        direction_pos_i = {
            'gradient courant': np.zeros(2),
            'ancien gradient': np.zeros(2),
            'longueur pas': reglages['pas_max'],
            'longueur pas gradient' : reglages['pas_max'],
            'longueur pas inertielle' : reglages['pas_max'],
            'proposition direction inertielle': np.zeros(2),
            'proposition direction gradient': np.zeros(2),
            'coordonnees proposition inertielle': np.zeros(2),
            'gradient proposition inertielle': np.zeros(2),
            'gradient proposition gradient': np.zeros(2),
            'coordonnees proposition gradient': np.zeros(2),
            'deplacement inertiel': np.zeros(2)
        }
        direction_ang_i = {
            'gradient courant': np.zeros(1),
            'ancien gradient': np.zeros(1),
            'longueur pas': reglages['pas_max'],
            'longueur pas gradient' : reglages['pas_max'],
            'longueur pas inertielle' : reglages['pas_max'],
            'proposition direction inertielle': np.zeros(1),
            'proposition direction gradient': np.zeros(1),
            'pas proposition inertielle': reglages['pas_max'],
            'pas proposition gradient': reglages['pas_max'],
            'gradient proposition inertielle': np.zeros(1),
            'gradient proposition gradient': np.zeros(1),
            'coordonnees proposition inertielle': np.zeros(1),
            'coordonnees proposition gradient': np.zeros(1),
            'deplacement inertiel': np.zeros(1)
        }
        stockage = [copy.deepcopy(direction_pos_i) for _ in range(n)] + [copy.deepcopy(direction_ang_i) for _ in range(n)]
        resol.stockage = stockage

    def update_histo():
        t0 = time.perf_counter()
        if not hasattr(update_histo, 'historique_gradients'):
          setattr(update_histo, 'historique_gradients' , [])
          setattr(update_histo, 'moyenne_gradient' , np.inf )
          setattr(update_histo, 'variance_gradient',  np.inf )
        # On garde seulement les gradients des sous vecteurs (normalisés) sur les n dernières itérations
        # Extraire les gradients courants normalisés
        grads_normes = []
        for i in range(2 * n):
            g = resol.stockage[i]['gradient courant']
            norm_g = np.linalg.norm(g)
            if norm_g > 0:
                g_norm = g / norm_g
            else:
                g_norm = g.copy()  # vecteur nul
            grads_normes.append(g_norm)

        grads_normes = np.array(grads_normes)  # forme (2n, dim_i)

        # On stocke la moyenne des gradients normalisés sur les derniers tours
        update_histo.historique_gradients.append(grads_normes)
        if len(update_histo.historique_gradients) > reglages['n_dernieres_it']:
            update_histo.historique_gradients.pop(0)
            # Calcul moyenne et variance (seulement norme de la moyenne, variance pas utilisée ici mais stockée)
            moyenne = np.mean(update_histo.historique_gradients, axis=0)  # moyenne sur les dernières itérations
            norme_moyenne = np.linalg.norm(np.mean(moyenne, axis=0))  # moyenne globale norme
            variance = np.var(update_histo.historique_gradients, axis=0).mean()
            update_histo.moyenne_gradient = norme_moyenne
            update_histo.variance_gradient = variance
        resol.tempos['update_histo'] += time.perf_counter() - t0

    def calculer_directions():
        t0 = time.perf_counter()
        if resol.premier_tour:
            # calcul perte et gradient
            perte, gradient = fonction(centres_orientations)
            # découpage et normalisation des sous vecteurs
            for i in range(n):
                # positions (2 dim)
                g_pos = gradient[2 * i:2 * i + 2]
                norm_g_pos = np.linalg.norm(g_pos)
                g_pos_norm = g_pos / norm_g_pos if norm_g_pos > 0 else g_pos.copy()
                resol.stockage[i]['gradient courant'] = g_pos_norm
                resol.stockage[i]['ancien gradient'] = np.zeros_like(g_pos_norm)
                # angles (1 dim)
                g_ang = gradient[2 * n + i:2 * n + i + 1]
                norm_g_ang = np.linalg.norm(g_ang)
                g_ang_norm = g_ang / norm_g_ang if norm_g_ang > 0 else g_ang.copy()
                resol.stockage[n + i]['gradient courant'] = g_ang_norm
                resol.stockage[n + i]['ancien gradient'] = np.zeros_like(g_ang_norm)
            resol.perte_courante = perte
            # initialiser les propositions direction inertielle et gradient (identiques au départ)
            for i in range(2 * n):
                resol.stockage[i]['proposition direction inertielle'] = resol.stockage[i]['gradient courant'].copy()
                resol.stockage[i]['proposition direction gradient'] = resol.stockage[i]['gradient courant'].copy()
        else:
          for i in range(2*n):
            dep_in = copy.copy(resol.stockage[i]['deplacement inertiel'] )
            dep_in = dep_in/( np.linalg.norm(dep_in)+1e-8)
            resol.stockage[i]['proposition direction inertielle'] = dep_in
            resol.stockage[i]['proposition direction gradient'] = resol.stockage[i]['gradient courant'].copy()


        resol.tempos['calcul_directions'] += time.perf_counter() - t0

    def tester_directions():
        t0 = time.perf_counter()
        # On crée deux propositions globales :
        # Proposition inertielle
        pas_max = reglages['pas_max']
        alpha = reglages['alpha']
        decalage_accepte = reglages['decalage accepte']
        pas_min = reglages['pas_min']

        def appliquer_pas(direction_cle, longueur_pas_cle):
            coords_test = cent_or.copy()
            for i in range(n):
                # positions
                dir_vec = resol.stockage[i][direction_cle]
                norm_dir = np.linalg.norm(dir_vec)
                if norm_dir > 0:
                    dir_norm = dir_vec / norm_dir
                else:
                    dir_norm = dir_vec.copy()
                pas_i = resol.stockage[i][longueur_pas_cle]
                coords_test[2 * i:2 * i + 2] += pas_i * dir_norm
            for i in range(n):
                # angles
                dir_vec = resol.stockage[n + i][direction_cle]
                norm_dir = np.linalg.norm(dir_vec)
                if norm_dir > 0:
                    dir_norm = dir_vec / norm_dir
                else:
                    dir_norm = dir_vec.copy()
                pas_i = resol.stockage[n + i][longueur_pas_cle]
                coords_test[2 * n + i] += pas_i * dir_norm[0]
            # modulo angles
            for i in range(n):
                coords_test[2 * n + i] = coords_test[2 * n + i] % (2 * np.pi)
            return coords_test

       # Initialisation pré-coup de sonde
        for i in range(2*n):
          for choix in ['gradient', 'inertielle']:
            if resol.stockage[i]['longueur pas']<= pas_min:
              resol.stockage[i]['longueur pas '+choix]=pas_min
            else :
              resol.stockage[i]['longueur pas '+choix]=(resol.stockage[i]['longueur pas'])
          
       #coup de sonde

        for choix in ['gradient', 'inertielle']:
          direction_cle = f'proposition direction {choix}'
          proposition = appliquer_pas(direction_cle,'longueur pas '+choix)
          perte_proposition, gradient_proposition = fonction(proposition) 
          setattr(resol, 'perte_proposition_'+choix, perte_proposition)
          gradients_pos, gradients_ang = np.split(gradient_proposition, [2*n])
          gradients_pos = np.split(gradients_pos, n)
          gradients_ang = np.split(gradients_ang,n)
          for i in range(n):
            resol.stockage[i]['gradient proposition ' + choix] = gradients_pos[i]/(np.linalg.norm(gradients_pos[i])+1e-8)
            resol.stockage[n+i]['gradient proposition ' + choix] = gradients_ang[i]/(np.linalg.norm(gradients_ang[i])+1e-8)

       #choix
        if resol.perte_proposition_inertielle < resol.perte_proposition_gradient:
          resol.inert = True
          resol.perte_courante = resol.perte_proposition_inertielle
        else:
          resol.inert = False
          resol.perte_courante = resol.perte_proposition_gradient

        if resol.inert :
          choix = 'inertielle'
        else :
          choix = 'gradient'

       # On calcule les pas 
        for i in range(2*n):
          diff = np.linalg.norm(resol.stockage[i]['gradient proposition '+choix]-resol.stockage[i]['gradient courant']) +1e-8
          resol.stockage[i]['longueur pas'] = np.clip(resol.stockage[i]['longueur pas '+choix]*reglages['decalage accepte']/diff, pas_min , min(pas_max, reglages['acceleration']*resol.stockage[i]['longueur pas '+choix] )) 

        resol.tempos['tester_directions'] += time.perf_counter() - t0

    def descendre(cent_or):
        t0 = time.perf_counter()
        # Choix direction par sous vecteur
        for i in range(2 * n):
            pas_i = resol.stockage[i]['longueur pas']
            dir_inert = resol.stockage[i]['proposition direction inertielle']
            dir_grad = resol.stockage[i]['proposition direction gradient']

            norm_inert = np.linalg.norm(dir_inert)
            norm_grad = np.linalg.norm(dir_grad)

            # Cas vecteur nul : garder vecteur nul, pas affecté
            if norm_inert > 0:
                dir_inert_norm = dir_inert / norm_inert
            else:
                dir_inert_norm = dir_inert.copy()
            if norm_grad > 0:
                dir_grad_norm = dir_grad / norm_grad
            else:
                dir_grad_norm = dir_grad.copy()

            # Calcul perte des propositions (calculés à tester_directions)
            perte_inert = resol.perte_proposition_inertielle
            perte_grad = resol.perte_proposition_gradient

            # Choix de la direction la plus favorable (on descend donc selon le minimum)
            if resol.inert : 
              direction_choisie = dir_inert_norm
            else:
              direction_choisie = dir_grad_norm
            # Calcul déplacement et maj de 'deplacement inertiel'
            deplacement = pas_i * direction_choisie
            resol.stockage[i]['deplacement inertiel'] = reglages['alpha']*direction_choisie + (1-reglages['alpha'])*resol.stockage[i]['deplacement inertiel']

            # Mise à jour centres_orientations
            if i < n:
                cent_or[2 * i:2 * i + 2] += deplacement
            else:
                cent_or[2 * n + i-n] += 2*np.pi*deplacement[0]
                cent_or[2 * n + i-n] %= 2 * np.pi

        resol.tempos['descendre'] += time.perf_counter() - t0
        return(cent_or)

    def maj_historique():
        t0 = time.perf_counter()
        # Mettre à jour gradient courant et ancien gradient avec lissage inertie
        for i in range(2 * n):
            resol.stockage[i]['ancien gradient'] = resol.stockage[i]['gradient courant'].copy()
            if resol.inert :
              resol.stockage[i]['gradient courant'] = resol.stockage[i]['gradient proposition inertielle']
            else :
              resol.stockage[i]['gradient courant'] = resol.stockage[i]['gradient proposition gradient']

        if hasattr(sauvegarde,'historique'):
          sauvegarde.historique.append({
            'tour': iteration,
            'centres_orientations': cent_or.copy(),
            'variance gradient': (update_histo.variance_gradient),
            'moyenne gradient': (update_histo.moyenne_gradient),
            'pas positions': np.mean([resol.stockage[i]["longueur pas"] for i in range(n)]),
            'pas angles': np.mean([resol.stockage[i + n]["longueur pas"] for i in range(n)]),
            'perte proposition inertielle' : resol.perte_proposition_inertielle,
            'perte proposition gradient' : resol.perte_proposition_gradient,
            'perte' : resol.perte_courante,
            'stockage': copy.deepcopy(resol.stockage)
        })
      
        resol.tempos['maj_historique'] += time.perf_counter() - t0

    def conditions_arret():
        if not hasattr(conditions_arret, 'bloques'):
          setattr(conditions_arret, 'bloques', {'inertielle':[False]*2*n, 'gradient':[False]*2*n})
        if not hasattr(conditions_arret, 'arret_gradient'):
          setattr(conditions_arret, 'arret_gradient', False)
        # On arrête si norme moyenne des derniers gradients < epsilon_arret
        conditions_arret.arret_gradient = update_histo.moyenne_gradient < reglages['epsilon_arret']
        arret_gradient = conditions_arret.arret_gradient
        t = conditions_arret.bloques
        arret_bloques = True
        for B in t.values():
          for b in B:
            if not b :
              arret_bloques = False
              break
        if arret_gradient or arret_bloques:
          if arret_gradient :
            print('arret_gradient')
          if arret_bloques :
            print('arret_bloques')
          return True
        conditions_arret.bloques = {'inertielle':[False]*2*n, 'gradient':[False]*2*n}

   # Initialisation
    init_stockage()
    max_iterations = 1000
    iteration = 0

   # Boucle
    while iteration < max_iterations:
        calculer_directions()
        update_histo()
        tester_directions()
        if conditions_arret():
            break
        cent_or = descendre(cent_or)
        maj_historique()
        iteration += 1
        resol.premier_tour = False

   #Retour
    copier_attributs(resol, resol_main_deux_directions)
    resol_main_deux_directions.historique_gradients = update_histo.historique_gradients
    return cent_or

def resol_2dir(centres_orientations, fonction, rayon, petit_rayon, axe, valeur, dims, reglages=None):
   # Gestion des reglages
    reglages_par_defaut = {
        'alpha': 0.5, #alpha faible => inertie forte
        'n_dernieres_it': 6,
        'seuil': 0.1,
        'epsilon_arret': 0.001,
        'decalage accepte': 0.05,
        'pas_min': 0.001,
        'pas_max': 0.01,
        'acceleration position' : 2,
        'acceleration angle' : 1.5,
        'stop inertie': True,
        'choix direction' : 'choix_direction_ps',
        'vers_gradient' : np.sqrt(2)/2,
        'opp_gradient' : - np.sqrt(2)/2,
        'stabilite perte' : 0.003,
        'pas_max_angles' : 0.05,
        'marge bipoints' : 0,
        'inertie forcee' : True,
        'coeff_frein' : 0.5
              }
    if reglages is None:
        reglages = reglages_par_defaut
    else:
        reglages = reglages_par_defaut | reglages

    n = centres_orientations.shape[0] // 3

    if hasattr(sauvegarde,'infos_arrangement'):
      sauvegarde.infos_arrangement = {'petit rayon' : petit_rayon,
         'axe' : axe,
         'valeur' : valeur,
         'dimensions' : dims}
      
    cent_or = np.copy(centres_orientations)
    
   # Pour stocker des trucs
    resol = type('', (), {})()
    resol.stockage = None
    resol.cent_or = centres_orientations
    resol.histo_pertes = [float('inf')]
    resol.tempos = {
        'calcul_directions': 0,
        'tester_directions': 0,
        'descendre': 0,
        'update_histo': 0,
        'maj_historique': 0
    }

   # Definition de fonctions internes
    def init_stockage():
        direction_pos_i = {
            'gradient courant': np.zeros(2),
            'gradient sonde': np.zeros(2),
            'longueur pas': reglages['pas_max'],
            'deplacement inertiel': np.zeros(2),
            'historique des choix' : ['gradient'],
            'historique des gradients' : [np.zeros(2)],
            'perte individuelle' : [float('inf')],
            'perte sonde' : float('inf')
        }
        direction_ang_i = {
            'gradient courant': np.zeros(1),
            'gradient sonde': np.zeros(1),
            'longueur pas': reglages['pas_max'],
            'deplacement inertiel': np.zeros(1),
            'historique des choix' : ['gradient'],
            'historique des gradients' : [np.zeros(1)],
            'perte individuelle' : [float('inf')],
            'perte sonde' : float('inf')
        }
        stockage = [copy.deepcopy(direction_pos_i) for _ in range(n)] + [copy.deepcopy(direction_ang_i) for _ in range(n)]
        resol.stockage = stockage

    def init_direction():
        # calcul perte et gradient
        perte, gradient = fonction(centres_orientations)
        # découpage et normalisation des sous vecteurs
        for i in range(n):
            # positions (2 dim)
            g_pos = gradient[2 * i:2 * i + 2]
            norm_g_pos = np.linalg.norm(g_pos)
            g_pos_norm = g_pos / norm_g_pos if norm_g_pos > 0 else g_pos.copy()
            resol.stockage[i]['gradient sonde'] = None
            resol.stockage[i]['gradient courant'] = g_pos_norm
            # angles (1 dim)
            g_ang = gradient[2 * n + i:2 * n + i + 1]
            norm_g_ang = np.linalg.norm(g_ang)
            g_ang_norm = g_ang / norm_g_ang if norm_g_ang > 0 else g_ang.copy()
            resol.stockage[n + i]['gradient sonde'] = None
            resol.stockage[n + i]['gradient courant'] = g_ang_norm
        resol.perte_courante = perte

    def appliquer_pas():
            coords_test = resol.cent_or.copy()
            for i in range(n):
                # positions
                if resol.stockage[i]['historique des choix'][-1] ==  'gradient':
                    dir_vec = resol.stockage[i]['gradient courant']
                else:
                    dir_vec = resol.stockage[i]['deplacement inertiel']

                norm_dir = np.linalg.norm(dir_vec)
                if norm_dir >  0:
                    dir_norm = dir_vec / norm_dir
                else:
                    dir_norm = dir_vec.copy()
                pas_i = resol.stockage[i]['longueur pas']
                coords_test[2 * i:2 * i + 2] += pas_i * dir_norm
            for i in range(n):
                # angles
                if resol.stockage[n+i]['historique des choix'][-1] ==  'gradient':
                    dir_vec = resol.stockage[n+i]['gradient courant']
                else:
                    dir_vec = resol.stockage[n+i]['deplacement inertiel']

                norm_dir = np.linalg.norm(dir_vec)
                if norm_dir > 0:
                    dir_norm = dir_vec / norm_dir
                else:
                    dir_norm = dir_vec.copy()
                pas_i = 2*pi*resol.stockage[n + i]['longueur pas']
                coords_test[2 * n + i] += pas_i * dir_norm[0]
            # modulo angles
            for i in range(n):
                coords_test[2 * n + i] = coords_test[2 * n + i] % (2 * np.pi)
            return coords_test

    def choix_direction():
        if reglages['choix direction'] == 'choix_direction_std':
            choix_direction_std()
        elif reglages['choix direction'] == 'choix_direction_ps':
            choix_direction_ps()
        else:
            raise ValueError("choix direction non reconnu")

    def choix_direction_std():
        resol.variances_gradient = []
        for i in range(2*n): 
            histo_g = resol.stockage[i]['historique des gradients']
            histo_c = resol.stockage[i]['historique des choix']
            # On regarde si on a une variance récente déconnante
            var_g = np.var(histo_g, axis=0).mean()
            resol.variances_gradient.append(var_g)


            # Si l'inertie a le contrôle et qu'on a déconné en terme de perte
            if resol.stockage[i]['historique des choix'][-1] == 'inertie' and resol.perte_courante > resol.histo_pertes[-1]:
                #on calme le jeu :
                (resol.stockage[i]['longueur pas']) = reglages['pas_min']
                (resol.stockage[i]['historique des choix']).append('gradient')

            #Si on oscille comm des débiles dans une position de pseudo équilibre
            elif var_g >= reglages['seuil'] and i < n: # On se change que pour les positions
                #Si c'est le cas, deux cas de figure
                inertie_recente = True
                for choix in histo_c:
                    if choix == 'gradient':
                        inertie_recente = False
                # 1er cas de figure : on était dans une config "gradient" :
                if not inertie_recente :
                    #on donne le pouvoir à l'inertie
                    (resol.stockage[i]['historique des choix']).append('inertie')
                    if reglages['stop inertie']:
                      (resol.stockage[i]['longueur du pas']) = reglages['pas_min']
                      if histo_c[-1] == 'gradient':
                        (resol.stockage[i]['deplacement inertiel']) = np.zeros(2)
                # 2eme cas de figure : ça fait déjà un moment qu'on essaie avec inertie
                else:
                    #on stoppe ce bibo pour l'instant
                    resol.stockage[i]['longueur pas'] = reglages['pas_min']
                    (resol.stockage[i]['historique des choix']).append('gradient')

            else :
                #Si la variance recente est pas déconnante, on suit le gradient
                (resol.stockage[i]['historique des choix']).append('gradient')

    def choix_direction_ps():
      resol.variances_gradient = []
      for i in range(n): 
          histo_g = resol.stockage[i]['historique des gradients']
          histo_c = resol.stockage[i]['historique des choix']
          # On regarde si on a une variance récente déconnante
          var_g = np.var(histo_g, axis=0).mean()
          resol.variances_gradient.append(var_g)

          gradient = resol.stockage[i]['gradient courant']
          gradient = gradient/(np.linalg.norm(gradient)+1e-8)
          inertie = resol.stockage[i]['deplacement inertiel']
          inertie = inertie/(np.linalg.norm(inertie)+1e-8)

          ps = np.dot(gradient,inertie)

          # Si l'inertie est orientee dans la même direction que le gradient, on laisse le pouvoir au gradient
          if ps >= reglages['vers_gradient'] :
            (resol.stockage[i]['historique des choix']).append('gradient')

          # Si ils sont opposés, c'est qu'on a suivi l'inertie et qu'on va se cogner contre un obstacle. On redonne alors le pourvoir au gradient et on calme le jeu 
          elif ps <= reglages['opp_gradient'] :
            (resol.stockage[i]['historique des choix']).append('gradient')
            resol.stockage[i]['longueur pas'] = reglages['pas_min']

          # Sinon, c'est que l'inertie est orientée est orthogonale au gradient, donc on est probablement coincés dans un min local. On laisse le pouvoir à l'inertie pour s'en sortir
          else :
            (resol.stockage[i]['historique des choix']).append('inertie')
            resol.stockage[i]['longueur pas'] = reglages['pas_max']

          (resol.stockage[n+i]['historique des choix']).append('gradient')

    def calculer_directions():
        t0 = time.perf_counter()

        for i in range(2*n):
          if not reglages['inertie forcee'] : 
            dep_in = copy.copy(resol.stockage[i]['deplacement inertiel'] )
            dep_in = dep_in/( np.linalg.norm(dep_in)+1e-8)
            grad  = resol.stockage[i]['gradient courant']
            grad = grad/(np.linalg.norm(grad)+1e-8)
            alpha = reglages['alpha']
            dep_in  = alpha*grad + (1-alpha)*dep_in            
            resol.stockage[i]['deplacement inertiel'] = dep_in
          else : 
            resol.stockage[i]['deplacement inertiel'] = np.mean(resol.stockage[i]['historique des gradients'], axis = 0)/ (np.linalg.norm(np.mean(resol.stockage[i]['historique des gradients'], axis = 0))+1e-8)



        resol.tempos['calcul_directions'] += time.perf_counter() - t0

    def coup_sonde():
        coup_de_sonde = appliquer_pas()
        perte_sonde, gradient_sonde = fonction(coup_de_sonde)
        resol.perte_sonde = perte_sonde
        gradients_pos, gradients_ang = np.split(gradient_sonde, [2*n])
        gradients_pos = np.split(gradients_pos, n)
        gradients_ang = np.split(gradients_ang, n)
        for i in range(n):
            resol.stockage[i]['gradient sonde'] = gradients_pos[i]/(np.linalg.norm(gradients_pos[i])+1e-8)
            resol.stockage[i+n]['gradient sonde'] = gradients_ang[i]/(np.linalg.norm(gradients_ang[i])+1e-8)

        bibos = fonction.bibos
        for i in range(len(bibos)):
          resol.stockage[i]['perte sonde']= - min(bibos[i]['distance bibo'], bibos[i]['distance bord'])

    def calculer_pas():
        
        for i in range(n):
          # Si le pouvoir est au gradient, on avance de manière à ne pas trop le faire varier d'une iteration sur l'autre 
          if resol.stockage[i]['historique des choix'][-1] == 'gradient':

            diff = np.linalg.norm(resol.stockage[i]['gradient sonde']-resol.stockage[i]['gradient courant']) +1e-8
            resol.stockage[i]['longueur pas'] = np.clip(resol.stockage[i]['longueur pas']*reglages['decalage accepte']/diff, reglages['pas_min'] , min(reglages['pas_max'], reglages['acceleration position']*resol.stockage[i]['longueur pas'] )) 

          #Sinon, on avance comme un bourrin. On s'inquiète pas de ne pas trop faire varier le gradient puisque dans tous les cas on est dans une zone ou le gradient varie beaucoup
          else :
            resol.stockage[i]['longueur pas'] = reglages['pas_max']
          

          diff = np.linalg.norm(resol.stockage[n+i]['gradient sonde']-resol.stockage[n+i]['gradient courant']) +1e-8
          resol.stockage[n+i]['longueur pas'] = np.clip(resol.stockage[n+i]['longueur pas']*reglages['decalage accepte']/diff, reglages['pas_min'] , min(reglages['pas_max_angles'], reglages['acceleration angle']*resol.stockage[n+i]['longueur pas'] )) 


          # Par contre, avec le bloc d'avant on a pu échapper au bipoints. Le truc, c'est que si on essaie d'échapper aux tripoints on va passer en mode inertie et bourriner dans l'obstacle suivant. Pour mesurer ça, on va utiliser essayer de determiner si récemment on a eu qque chose qui s'oppose au mouvement de l'inertie
          tripoint = False
          hist = resol.stockage[i]['historique des gradients']
          din = resol.stockage[i]['deplacement inertiel']
          for g in hist:
            if np.dot(g,din) <= 0 - reglages['marge bipoints']:
              tripoint = True
          if tripoint:
            resol.stockage[i]['longueur pas'] = reglages['pas_min']
            resol.stockage[i]['tripoint'] = True
          else :
            resol.stockage[i]['tripoint'] = False

          if resol.stockage[i]['perte sonde']> resol.stockage[i]['perte individuelle'][-1] and resol.stockage[i]['perte sonde'] <= resol.perte_sonde +  3*reglages['pas_min']:
            resol.stockage[i]['longueur pas'] = reglages['pas_min']
            resol.stockage[n+i]['longueur pas'] = reglages['pas_min']
            resol.stockage[i]['historique des choix'][-1] = 'gradient'

    def descendre():
        cent_or  = appliquer_pas()
        perte, gradient = fonction(cent_or)
        resol.cent_or = cent_or
        resol.histo_pertes.append(perte) 
        resol.perte_courante = perte
        gradients_pos, gradients_ang = np.split(gradient, [2*n])
        gradients_pos = np.split(gradients_pos, n)
        gradients_ang = np.split(gradients_ang, n)
        for i in range(n):
            resol.stockage[i]['gradient courant'] = gradients_pos[i]/(np.linalg.norm(gradients_pos[i]) +  1e-8)
            resol.stockage[i+n]['gradient courant'] = gradients_ang[i]/(np.linalg.norm(gradients_ang[i]) + 1e-8 )

        bibos = fonction.bibos
        for i in range(len(bibos)):
          resol.stockage[i]['perte individuelle'].append( - min(bibos[i]['distance bibo'], bibos[i]['distance bord']) )

    def update_histo():
        t0 = time.perf_counter()
        resol.moyennes_gradient = []

        for i in range(2 * n):

          # On met à jour l'historique ds gradients
            g = resol.stockage[i]['gradient courant']
            norm_g = np.linalg.norm(g)
            if norm_g > 0:
                g_norm = g / norm_g
            else:
                g_norm = g.copy()  # vecteur nul
            resol.stockage[i]['historique des gradients'].append(g_norm)
            if len(resol.stockage[i]['historique des gradients'])> reglages['n_dernieres_it']:
                resol.stockage[i]['historique des gradients'].pop(0)

          # On met à jour l'historique des choix
            if len(resol.stockage[i]['historique des choix']) >reglages['n_dernieres_it']:
                resol.stockage[i]['historique des choix'].pop(0)

          # On met à jour les moyennes des gradients
            m= np.mean(resol.stockage[i]['historique des gradients'], axis = 0 )
            resol.moyennes_gradient.append(m)



        resol.tempos['update_histo'] += time.perf_counter() - t0

    def conditions_arret():
        resol.arret = False

        stag_perte = True
        if iteration > 3*reglages['n_dernieres_it']:
            #on extrait les reglages['n_dernieres_it'] valeurs de la perte
            pertes = resol.histo_pertes[-3*reglages['n_dernieres_it']:]
            #on verifie que c'est bien stable
            p = np.mean(pertes)
            for i in range(len(pertes)):
                if abs(pertes[i] - p) > reglages['stabilite perte']:
                    stag_perte = False
        else :
            stag_perte = False


        arret_gradient = True
        for i in range(n):
            var = resol.variances_gradient[i]
            if var < reglages['seuil'] or iteration <= reglages['n_dernieres_it']:
                arret_gradient = False

        arret_tripoint = (iteration >=  reglages['n_dernieres_it'])
        for i in range(n):
            if not resol.stockage[i]['tripoint']:
              arret_tripoint = False


        resol.arret = stag_perte and arret_tripoint 

    def maj_historique():
        t0 = time.perf_counter()

        if hasattr(sauvegarde,'historique'):
          sauvegarde.historique.append({
            'tour': iteration,
            'centres_orientations': resol.cent_or.copy(),
            'variance gradient': resol.variances_gradient,
            'moyenne gradient': resol.moyennes_gradient,
            'pas positions':([resol.stockage[i]["longueur pas"] for i in range(n)]),
            'pas angles': ([resol.stockage[i + n]["longueur pas"] for i in range(n)]),
            'perte' : resol.perte_courante,
            'stockage': copy.deepcopy(resol.stockage)
        })
        resol.tempos['maj_historique'] += time.perf_counter() - t0

   # Initialisation
    init_stockage()
    init_direction()
    max_iterations = 200
    iteration = 0
    resol.arret = False
    resol.moyennes_gradient = [0]*2*n

   # Boucle
    while iteration < max_iterations and not resol.arret :
        choix_direction()
        calculer_directions()
        coup_sonde()
        calculer_pas()
        maj_historique()
        descendre()
        update_histo()
        conditions_arret()        
        iteration += 1

   # Retour
    copier_attributs(resol, resol_2dir)   
    return(resol_2dir.cent_or)

def sauvegarde():
  print('à partir de maintenant, on sauvegarde ce qui se passe')
  sauvegarde.historique = []
  sauvegarde.infos_arrangement = []
  print('pour arrêter de sauvegarder, lancer end_sauvegarde')

def end_sauvegarde(sauvegarder_fichier = False):
  import json
  if not hasattr(sauvegarde, 'historique'):
    print('pas de sauvegarde')
  else:
    s= sauvegarde.historique
    delattr(sauvegarde, 'historique')
    if sauvegarder_fichier:
      with open('historique.json', 'w') as f:
        json.dump(s,f)
    return(s)

def frames():
    if not hasattr(sauvegarde, 'historique') or not hasattr(sauvegarde, 'infos_arrangement'):
        print('Aucune sauvegarde disponible')
        return []
    
    frames = []
    historique = sauvegarde.historique
    infos_arrg = sauvegarde.infos_arrangement
    axe = np.array([0, 0, 1])
    x, y = infos_arrg['dimensions'].copy()
    dims = np.array([x, y, 1])
    pr = copy.copy(infos_arrg['petit rayon'])
    
    for h in historique:
        co = copy.copy(h['centres_orientations'])
        c,o = t_2_d_bibos(injection_bibos(co, axe, 0.5))
        it = copy.copy(h['tour'])
        
        gradient = []
        inertie = []
        for j in range(2 * len(c)):
          if 'proposition direction gradient' in h['stockage'][j]:
            gradient.append(h['stockage'][j]['proposition direction gradient']*h['stockage'][j]['longueur pas gradient'])
            inertie.append(h['stockage'][j]['proposition direction inertielle']*h['stockage'][j]['longueur pas inertielle'])
          else :
            g = h['stockage'][j]['gradient courant']*h['stockage'][j]['longueur pas']
            inert = h['stockage'][j]['deplacement inertiel']*h['stockage'][j]['longueur pas']
            gradient.append(g)
            inertie.append(inert)
        
        frame = {
            'rayon': -h['perte'],
            'petit rayon': pr,
            'orientations': o,
            'dimensions du carton': dims,
            'centres': c,
            'axe': axe,
            'iteration': it,
            'gradient': gradient,
            'inertie': inertie
        }
        
        frames.append(frame)
    
    return frames

def afficher_indicateurs(histo):

    # -- Paramètres de base
    n_total = len(histo[0]['centres_orientations'])
    n = (2 * n_total) // 3
    tours = np.array([info['tour'] for info in histo])

    # -- Séries principales
    pas_pos    = np.array([info['pas positions']          for info in histo])
    pas_ang    = np.array([info['pas angles']             for info in histo])
    moy_grad   = np.array([info['moyenne gradient']       for info in histo])
    var_grad   = np.array([info['variance gradient']      for info in histo])
    co_list    = [info['centres_orientations'] for info in histo]
    """est_pos = np.array([info['est_pos'] for info in histo])
    est_ang = np.array([info['est_ang'] for info in histo])"""

    # -- Distance à la dernière frame
    cible = co_list[-1]
    d_pos = [np.linalg.norm((co - cible)[:2*n])   for co in co_list]
    d_ang = [np.linalg.norm((co - cible)[2*n:])   for co in co_list]

    # -- Variations entre étapes
    t_suiv       = tours[1:]
    delta_pas_pos  = np.abs(np.diff(pas_pos))
    delta_pas_ang  = np.abs(np.diff(pas_ang))
    delta_moy_grad = np.abs(np.diff(moy_grad))
    delta_var_grad = np.abs(np.diff(var_grad))

    # === Figure 1 : pas / moy / var / distances ===
    fig, axs = plt.subplots(3, 2, figsize=(12, 10))

    axs[0, 0].plot(tours, pas_pos, 'o-', label='Pas position')
    axs[0, 0].set_title("Évolution du pas de position")
    axs[0, 0].grid(True)

    axs[0, 1].plot(tours, pas_ang, 'o-', color='orange', label='Pas angle')
    axs[0, 1].set_title("Évolution du pas d’angle")
    axs[0, 1].grid(True)

    axs[1, 0].plot(tours, moy_grad, 'o-', color='green', label='Moyenne gradient')
    axs[1, 0].set_title("Évolution de la moyenne du gradient")
    axs[1, 0].grid(True)

    axs[1, 1].plot(tours, var_grad, 'o-', color='red', label='Variance gradient')
    axs[1, 1].set_title("Évolution de la variance du gradient")
    axs[1, 1].grid(True)

    axs[2, 0].plot(tours, d_pos, 'o-', color='purple', label='Distance pos')
    axs[2, 0].plot(tours, d_ang, 's-', color='brown', label='Distance ang')
    axs[2, 0].set_title("Distance à la situation finale")
    axs[2, 0].legend()
    axs[2, 0].grid(True)

    # on laisse axs[2,1] libre
    axs[2, 1].axis('off')

    plt.tight_layout()
    plt.show()

    # === Figure 2 : variations de pas et estimateurs ===
    fig2, axs2 = plt.subplots(3, 2, figsize=(12, 12))

    axs2[0, 0].plot(t_suiv, delta_pas_pos, 'o-', label='Δ pas pos')
    axs2[0, 0].set_title("Variation du pas de position")
    axs2[0, 0].grid(True)

    axs2[0, 1].plot(t_suiv, delta_pas_ang, 'o-', color='orange', label='Δ pas ang')
    axs2[0, 1].set_title("Variation du pas d’angle")
    axs2[0, 1].grid(True)

    axs2[1, 0].plot(t_suiv, delta_moy_grad, 'o-', color='green', label='Δ moyenne grad')
    axs2[1, 0].set_title("Variation de la moyenne du gradient")
    axs2[1, 0].grid(True)

    axs2[1, 1].plot(t_suiv, delta_var_grad, 'o-', color='red', label='Δ variance grad')
    axs2[1, 1].set_title("Variation de la variance du gradient")
    axs2[1, 1].grid(True)
    
    """    axs2[2, 0].plot(tours, est_pos, 'o-', color='magenta', label='Est. var pos')
      axs2[2, 0].set_title("Estimateur de variance glissante (position)")
      axs2[2, 0].grid(True)

      axs2[2, 1].plot(tours, est_ang, 'o-', color='cyan', label='Est. var angle')
      axs2[2, 1].set_title("Estimateur de variance glissante (angle)")
      axs2[2, 1].grid(True)"""

    plt.tight_layout()
    plt.show()

def resol_main_autre(centres_orientations, fonction, rayon, petit_rayon, axe, valeur, dims, reglages = None):
  reglages_par_defaut = {
    'alpha resol': 0.1,
    'n_dernieres_it': 10,
    'stab': 0.1,
    'epsilon_arret': 0.001,
    'pas_max': 0.01,
    'decalage accepte position': 0.01,
  }

  if reglages is None:
    reglages = reglages_par_defaut
  else:
    reglages = reglages | reglages_par_defaut

  historique = []
  cpt = 0
  n = len(centres_orientations) // 3

  def normalise_blocs(vec):
    res = np.zeros_like(vec)
    for i in range(n):
      idx = 2*i
      bloc = vec[idx:idx+2]
      norme = np.linalg.norm(bloc) + 1e-8
      res[idx:idx+2] = bloc / norme
    for i in range(n):
      idx = 2*n + i
      val = vec[idx]
      res[idx] = val / (abs(val) + 1e-8)
    return res

  def update_histo(gradient, histo):
    while len(histo) >= reglages['n_dernieres_it']:
      histo.pop(0)
    histo.append(gradient)
    update_histo.variance_gradient = np.var(histo, axis=1)
    update_histo.moyenne_gradient = np.mean(histo, axis=0)

  def condition_arret(perte, cpt):
    arret_gradient = np.linalg.norm(update_histo.moyenne_gradient) <= reglages['stab']
    ppdm = -perte
    pgdm = perte_inch_v2_bibos.plus_grande_distance_minimale
    d = pgdm - ppdm
    arret_perte = d <= reglages['epsilon_arret']
    return arret_gradient and cpt >= reglages['n_dernieres_it']

  if not hasattr(resol_main_autre, 'gradient_prec'):
    resol_main_autre.gradient_prec = np.zeros_like(centres_orientations)

  def tester_direction(centres, direction, pas_max):
    pas = np.zeros_like(centres)
    for i in range(n):
      idx = 2*i
      dir_xy = direction[idx:idx+2]
      pas[idx:idx+2] = pas_max * dir_xy / (np.linalg.norm(dir_xy) + 1e-8)
    for i in range(n):
      idx = 2*n + i
      val = direction[idx]
      pas[idx] = pas_max * val / (abs(val) + 1e-8)
    centres_test = centres + pas
    for j in range(2 * n, 3 * n):
      centres_test[j] = centres_test[j] % (2 * pi)
    perte, gradient = fonction(centres_test)
    return perte, centres_test, gradient

  while True:
    perte, gradient = fonction(centres_orientations)
    gradient = normalise_blocs(gradient)
    inertiel = normalise_blocs((1 - reglages['alpha resol']) * resol_main_autre.gradient_prec + reglages['alpha resol'] * gradient)

    perte1, centres1, grad1 = tester_direction(centres_orientations, gradient, reglages['pas_max'])
    perte2, centres2, grad2 = tester_direction(centres_orientations, inertiel, reglages['pas_max'])

    longueur = reglages['pas_max']
    while perte1 >= perte and perte2 >= perte and longueur >= reglages['epsilon_arret']:
      longueur *= 0.5
      perte1, centres1, grad1 = tester_direction(centres_orientations, gradient, longueur)
      perte2, centres2, grad2 = tester_direction(centres_orientations, inertiel, longueur)

    if perte1 < perte2:
      centres_orientations = centres1
      nouveau_gradient = grad1
    else:
      centres_orientations = centres2
      nouveau_gradient = grad2

    diff = np.linalg.norm(grad1 - grad2) + 1e-8
    facteur = reglages['pas_max'] * reglages['decalage accepte position'] / diff

    direction_finale = normalise_blocs(nouveau_gradient)
    pas = np.zeros_like(centres_orientations)
    for i in range(n):
      idx = 2*i
      dir_xy = direction_finale[idx:idx+2]
      pas[idx:idx+2] = facteur * dir_xy / (np.linalg.norm(dir_xy) + 1e-8)
    for i in range(n):
      idx = 2*n + i
      val = direction_finale[idx]
      pas[idx] = facteur * val / (abs(val) + 1e-8)

    centres_orientations = centres_orientations + pas
    for j in range(2 * n, 3 * n):
      centres_orientations[j] = centres_orientations[j] % (2 * pi)

    resol_main_autre.gradient_prec = gradient
    update_histo(gradient, historique)
    cpt += 1

    if hasattr(sauvegarde, 'historique'):
      infos = {
        'tour': cpt,
        'centres_orientations': centres_orientations,
        'variance gradient': np.linalg.norm(update_histo.variance_gradient),
        'moyenne gradient': np.linalg.norm(update_histo.moyenne_gradient),
      }
      sauvegarde.historique.append(infos)

    if condition_arret(perte, cpt):
      break

  return centres_orientations

def copier_attributs(source, cible):
    for k, v in vars(source).items():
        setattr(cible, k, v)

def afficher_histo(liste, indice=None, cle=None):
    def filtrer(dico, cle):
        if cle is None:
            # Tout sauf 'stockage'
            return {k: v for k, v in dico.items() if k != 'stockage'}
        elif isinstance(cle, str):
            return {cle: dico.get(cle)}
        elif isinstance(cle, list):
            return {k: dico.get(k) for k in cle if k in dico}
        else:
            raise ValueError("cle doit être None, une string ou une liste de strings.")

    if indice is None:
        for i, dico in enumerate(liste):
            print(f"--- Élément {i} ---")
            extrait = filtrer(dico, cle)
            for k, v in extrait.items():
                print(f"{k} : {v}")
            print()
    else:
        if indice < 0 or indice >= len(liste):
            print("Indice hors limites.")
            return
        print(f"--- Élément {indice} ---")
        extrait = filtrer(liste[indice], cle)
        for k, v in extrait.items():
            print(f"{k} : {v}")

def frame_to_arrangement(frame):
  arrg = copy.deepcopy(frame)
  arrg['grand rayon'] = arrg['rayon']
  arrg['nom'] = 'flvfhve'
  arrg['longueur']=1
  return arrg

def tracer_variances_gradients(frames):
    if not frames:
        print("La liste 'frames' est vide.")
        return

    # On suppose que toutes les listes ont la même longueur
    premiere_variance = frames[0]["variance gradient"]
    n = len(premiere_variance)

    # Initialiser une liste de listes, une par courbe
    courbes = [[] for _ in range(n)]

    for frame in frames:
        variance_gradient = frame["variance gradient"]
        for i in range(n):
            courbes[i].append(variance_gradient[i])

    # Tracer les courbes
    fig, axs = plt.subplots(n, 1, figsize=(8, 2 * n), sharex=True)
    if n == 1:
        axs = [axs]  # garantir l'itérabilité

    for i in range(n):
        axs[i].plot(courbes[i])
        axs[i].set_ylabel(f'Var grad {i}')
        axs[i].grid(True)

    axs[-1].set_xlabel("Itération")

    plt.tight_layout()
    plt.show()

# Projection d'un point sur un plan orthogonal à un axe
def projeter_sur_plan_feur(points, axe):
    l = []
    for i in range(len(points)):
        x,y,z =points[i]
        l.append(np.array([x,y]))
    return np.array(l)
    
# Dessin d’un seul frame
def dessiner_frame(ax, frame, facteur=10):
    centres = np.array(frame["centres"])
    orientations = np.array(frame["orientations"])
    axe = np.array(frame["axe"])
    rayon = frame["rayon"]
    petit_rayon = frame["petit rayon"]
    carton = np.array(frame["dimensions du carton"])
    gradient = frame["gradient"]
    inertie = frame["inertie"]
    iteration = frame["iteration"]

    n = len(orientations)
    grand_centres_2d = projeter_sur_plan_feur(centres, axe)
    arrangement = {
        "centres": centres,
        "orientations": orientations,
        "axe": axe,
        "grand rayon": rayon,
        "petit rayon": petit_rayon
    }
    petit_centres_2d = projeter_sur_plan_feur(petits_centres(arrangement), axe)

    # rectangle du carton
    coin = np.array([0, 0, 0])
    x_vec = np.array([1, 0, 0]) * carton[0]
    y_vec = np.array([0, 1, 0]) * carton[1]
    coin_proj = projeter_sur_plan_feur(np.array([coin]), axe)[0]
    x_proj = projeter_sur_plan_feur(np.array([x_vec]), axe)[0]
    y_proj = projeter_sur_plan_feur(np.array([y_vec]), axe)[0]
    rect_vect1 = x_proj - coin_proj
    rect_vect2 = y_proj - coin_proj
    rect = patches.Polygon([coin_proj,
                            coin_proj + rect_vect1,
                            coin_proj + rect_vect1 + rect_vect2,
                            coin_proj + rect_vect2],
                           closed=True, edgecolor='black', facecolor='pink')
    ax.add_patch(rect)

    # disques
    for i in range(n):
        ax.add_patch(plt.Circle(grand_centres_2d[i], rayon, color='blue', alpha=0.5))
        ax.add_patch(plt.Circle(petit_centres_2d[i], petit_rayon, color='orange', alpha=0.5))
        ax.text(*grand_centres_2d[i], str(i), color='black', ha='center', va='center', fontsize=8)

    # flèches gradients
    for i in range(n):
        v = np.array(gradient[i])
        orig = grand_centres_2d[i]
        ax.arrow(orig[0], orig[1], facteur * v[0], facteur * v[1], head_width=0.005, head_length=0.005, fc='green', ec='green', length_includes_head=True)

        theta = orientations[i]
        e_theta = np.array([np.sin(theta), -np.cos(theta)])
        v_theta = float(gradient[n + i]) * e_theta
        orig_theta = petit_centres_2d[i]
        ax.arrow(orig_theta[0], orig_theta[1], facteur * v_theta[0], facteur * v_theta[1], head_width=0.005, head_length=0.005, fc='green', ec='green', length_includes_head=True)

    # flèches inertie
    for i in range(n):
        v = np.array(inertie[i])
        orig = grand_centres_2d[i]
        ax.arrow(orig[0], orig[1], facteur * v[0], facteur * v[1],head_width=0.005, head_length=0.005, fc='purple', ec='purple', length_includes_head=True)

        theta = orientations[i]
        e_theta = np.array([np.sin(theta), -np.cos(theta)])
        v_theta = float(inertie[n + i]) * e_theta
        orig_theta = petit_centres_2d[i]
        ax.arrow(orig_theta[0], orig_theta[1], facteur * v_theta[0], facteur * v_theta[1],head_width=0.005, head_length=0.005, fc='purple', ec='purple', length_includes_head=True)

    # texte d’itération
    ax.text(0.01, 0.99, f"Itération : {iteration}", ha='left', va='top',transform=ax.transAxes, fontsize=10)

    ax.set_aspect('equal')
    ax.axis('off')

# Affichage statique
def dessiner_descente(debut, longueur=1):
    frames_data = frames()
    for i in range(debut, debut + longueur):
        fig, ax = plt.subplots(figsize=(6, 6))
        dessiner_frame(ax, frames_data[i])
        plt.show()

# Animation
def film_descente(nom="animation.gif"):
    frames_data = frames()

    fig, ax = plt.subplots(figsize=(6, 6))

    def update(i):
        ax.clear()
        dessiner_frame(ax, frames_data[i])

    ani = animation.FuncAnimation(fig, update, frames=len(frames_data), interval=200)
    ani.save(nom, writer='pillow')
    plt.close()


