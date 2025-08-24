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
from minimiseur import *


def generer_grille_centre(N, dimensions_carton, axe, reglages=None):
    ### PRÉPARATION
    reglages_par_defaut = {
        'facteur concentration': 0.2  # portion de la zone utilisée
    }

    if reglages is not None:
        reglages = reglages_par_defaut | reglages
    else:
        reglages = reglages_par_defaut

    facteur = reglages['facteur concentration']

    ### GÉNÉRATION DES CENTRES EN GRILLE CONCENTRÉE
    axe = [int(a) for a in axe]
    direction_index = np.argmax(np.abs(axe))
    dims_2d = [i for i in range(3) if i != direction_index]

    x_dim = dimensions_carton[dims_2d[0]]
    y_dim = dimensions_carton[dims_2d[1]]

    # Zone centrale plus petite
    x_eff = x_dim * facteur
    y_eff = y_dim * facteur

    n_cols = int(np.ceil(np.sqrt(N * x_eff / y_eff)))
    n_rows = int(np.ceil(N / n_cols))

    x_step = x_eff / (n_cols + 1)
    y_step = y_eff / (n_rows + 1)

    x_offset = (x_dim - x_eff) / 2
    y_offset = (y_dim - y_eff) / 2

    centres = []
    count = 0
    for row in range(n_rows):
        for col in range(n_cols):
            if count >= N:
                break
            centre = np.zeros(3)
            centre[direction_index] = dimensions_carton[direction_index] / 2  # centré selon l’axe
            centre[dims_2d[0]] = x_offset + (col + 1) * x_step
            centre[dims_2d[1]] = y_offset + (row + 1) * y_step
            centres.append(centre.tolist())
            count += 1

    ### CONSTRUCTION DE L’ARRANGEMENT COMPLET
    arrangement = {
        'nom': 'Arrangement en grille centrée',
        'axe': axe,
        'nombre': N,
        'centres': centres,
        'rayon': 0.1,
        'longueur': 0.99 * dimensions_carton[direction_index],
        'dimensions du carton': dimensions_carton,
        'origine du carton': [0, 0, 0],
        'nombre de couches': 1
    }

    ### RETOUR
    arrangement = tasser(arrangement, interaction=False)
    generer_grille_centre.valeurs_intermediaires = []
    return arrangement

def generation_aleatoire(N, dimensions_carton, axe, reglages=None):
 ### PRÉPARATION
    ## GESTION DES RÉGLAGES
        # on définit des réglages par défaut
    reglages_par_defaut = {
        # pas de réglages nécessaires pour l’instant
    }

    if reglages is not None:
        reglages_par_defaut = reglages | reglages
    else:
        reglages = reglages_par_defaut

 ### GÉNÉRATION DES CENTRES
    # on s'assure que l'axe est un vecteur d'entiers
    axe = [int(a) for a in axe]
    direction_index = np.argmax(np.abs(axe))

    centres = []
    for _ in range(N):
        centre = np.zeros(3)
        for i in range(3):
            if i == direction_index:
                centre[i] = dimensions_carton[i] / 2  # on centre les cylindres dans leur direction
            else:
                centre[i] = np.random.uniform(0, dimensions_carton[i])
        centres.append(list(centre))

 ### CONSTRUCTION DE L’ARRANGEMENT COMPLET
    arrangement = {
        'nom': 'Arrangement aléatoire',
        'axe': axe,
        'nombre': N,
        'centres': centres,
        'rayon': 0.1,  # valeur initiale, sera ajustée plus tard
        'longueur': 0.99 * dimensions_carton[direction_index],
        'dimensions du carton': dimensions_carton,
        'origine du carton': [0, 0, 0],
        'nombre de couches' : 1
    }

 ### RETOUR
    arrangement = tasser(arrangement, interaction = False)
    generation_aleatoire.valeurs_intermediaires = []
    return arrangement

def film(valeurs_intermediaires, arrangement, nom_fichier="animation_compressed.gif"):
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
    dimensions_carton = arrangement['dimensions du carton']
    axe = arrangement['axe']
    norm = 0.5*np.dot( dimensions_carton, np.array([1, 1, 1])-np.array(axe))
    origine_carton = arrangement['origine du carton']
    frames = valeurs_intermediaires

   # === Création de la figure compressée ===
    fig, ax = plt.subplots(figsize=(6, 4), dpi=80)  # Taille et résolution réduites
    ax.set_aspect('equal')

   # Détermination des limites automatiquement à partir du carton
    if axe[0] == 1:
        ax.set_xlim(origine_carton[1], origine_carton[1] + dimensions_carton[1]/norm)
        ax.set_ylim(origine_carton[2], origine_carton[2] + dimensions_carton[2]/norm)
    elif axe[1] == 1:
        ax.set_xlim(origine_carton[0], origine_carton[0] + dimensions_carton[0]/norm)
        ax.set_ylim(origine_carton[2], origine_carton[2] + dimensions_carton[2]/norm)
    else:
        ax.set_xlim(origine_carton[0], origine_carton[0] + dimensions_carton[0]/norm)
        ax.set_ylim(origine_carton[1], origine_carton[1] + dimensions_carton[1]/norm)

   # Textes d'information
    info_text = ax.text(0.02, 0.98, '', transform=ax.transAxes, va='top', ha='left',
                        bbox=dict(facecolor='white', alpha=0.8, edgecolor='gray'))
    info_text_perte = ax.text(0.02, 0.88, '', transform=ax.transAxes, va='top', ha='left',
                              bbox=dict(facecolor='white', alpha=0.8, edgecolor='gray'))
    info_text_iter = ax.text(0.02, 0.78, '', transform=ax.transAxes, va='top', ha='left',
                             bbox=dict(facecolor='white', alpha=0.8, edgecolor='gray'))
    info_text_rayon  = ax.text(0.02, 0.68, '', transform=ax.transAxes, va='top', ha='left', bbox=dict(facecolor='white', alpha=0.8, edgecolor='gray'))

   #
    patch_carton = ajouter_carton(ax, dimensions_carton/norm, axe)
    cercle_plots = []
   #
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

        centres = np.array(frame['centres'])
        rayon = frame['rayon']
        texte = frame.get('texte', '')
        perte = frame.get('perte', 0)
        couleur = frame.get('code couleur', 0)
        iteration = frame.get('iteration du minimiseur', 'N/A')

        cercle_plots = ajouter_disques(ax, centres, rayon, axe)

        if couleur == 1:
            patch_carton.set_color('red')
        elif couleur == 2:
            patch_carton.set_color('green')
        elif couleur == 5:
            patch_carton.set_color('magenta')
        else :
            patch_carton.set_color('blue')

        info_text.set_text(f"Étape : {texte}")
        info_text_perte.set_text(f"Perte : {perte:.2f}")
        info_text_iter.set_text(f"Itération : {iteration}")
        info_text_rayon.set_text(f"Rayon : {rayon:.2f}")

        return cercle_plots + [info_text, info_text_perte, info_text_iter, patch_carton]

    ani = animation.FuncAnimation(fig, update, frames=frames, init_func=init, blit=False)

    # === Sauvegarde en GIF compressé ===
    writer = PillowWriter(fps=4)  # Tu peux descendre à 10 pour alléger encore
    ani.save(nom_fichier, writer=writer)

    print(f"✅ Animation GIF enregistrée sous '{nom_fichier}'")

def phase_grossiere(base, r_max, reglages):
    ### PHASE GROSSIÈRE : AUGMENTATION EXPONENTIELLE DU RAYON
    valeurs_intermediaires = []
    while True:
        base['rayon'] = r_max
        tentative = minimiseur(base, reglages)
        valeurs_intermediaires += minimiseur.valeurs_intermediaires

        if legitime_bien_aligne(tentative, reglages):
            tentative_test = inch_allah(tentative, reglages)
            if legitime_bien_aligne(tentative_test, reglages):
               if tentative_test['rayon'] > tentative['rayon']:
                   tentative = tentative_test
                   r_max = tentative_test['rayon']
            print('On a trouvé un rayon admissible', r_max) ##########
            r_max *= 2
            print('On essaie donc avec', r_max)
        else:
            break

    phase_grossiere.valeurs_intermediaires = valeurs_intermediaires
    return r_max

def inch_allah_pas_wish(arrangement, reglages=None):
    print("incha'allah lancé")
    debut = time.time()

    # Réglages par défaut
    reglages_par_defaut = {
        'iterations par tour': 100,
        'fonction pertes inch': perte_inch_v3
    }
    if reglages is not None:
        reglages = reglages_par_defaut | reglages
    else:
        reglages = reglages_par_defaut

    n_it = reglages['iterations par tour']
    perte_inch = reglages['fonction pertes inch']

    # Copie pour ne pas modifier l'arrangement d'origine
    arrangement_modifiable = copy.deepcopy(arrangement)
    centres_3d = np.array(arrangement_modifiable['centres'])
    axe = np.array(arrangement_modifiable['axe'], dtype=float)
    origine = np.array(arrangement_modifiable['origine du carton'], dtype=float)
    point_sert_a_rien = np.dot(axe, centres_3d[0])

    # Projection initiale en 2D
    x0 = projection(centres_3d.flatten(), axe)

    # Fonction à minimiser
    def a_minimiser(x_2d):
        centres_projectes = injection(x_2d, axe, point_sert_a_rien).reshape((-1, 3)) + origine
        arrangement_modifiable['centres'] = centres_projectes.tolist()
        perte, gradient = perte_inch(arrangement_modifiable)
        print('norme du gradient : ', np.linalg.norm(gradient))
        return perte, 100000 * gradient  # Le facteur 0.1 peut être ajusté si besoin

    # Optimisation
    res = minimize(fun=a_minimiser,
                   x0=x0,
                   method='BFGS',
                   jac=True,
                   options={'maxiter': n_it, 'disp': True})


    # Reconstruction finale
    x_final = res.x
    centres_optim_3d = injection(x_final, axe, point_sert_a_rien).reshape((-1, 3)) + origine
    arrangement_modifiable['centres'] = centres_optim_3d.tolist()

    # Ajustement du rayon (si requis)
    arrangement_final = ajuster_rayon(arrangement_modifiable, reglages)

    print("incha'allah terminé en", round(time.time() - debut, 2), "s")
    return arrangement_final

def phase_dichotomie(base, r_min, r_max, reglages):
  ##GESTION DES REGLAGES
    reglages_par_defaut = {'max_iterations' : 15}
    if reglages is None:
        reglages = reglages_par_defaut
    else : 
        reglages = reglages_par_defaut | reglages
    import copy  # à inclure si ce n'est pas déjà fait

  ## PRÉPARATION
    valeurs_intermediaires = []
    print('On sait que l\'on doit chercher un rayon entre', r_min, 'et', r_max)
    print('On sait que la précision de notre réponse doit être : ', reglages['epsilon'])

    meilleur_arrangement = None
    meilleur_rayon = 0
    if legitime_bien_aligne(base, reglages):
        meilleur_arrangement = base
        meilleur_rayon = base['rayon']

    base_actuelle = copy.deepcopy(base)

  ## BOUCLE PRINCIPALE
    for _ in range(reglages['max_iterations']):
        r_milieu = 0.5 * (r_min + r_max)
        print('On essaie avec un rayon de', r_milieu)
        base_actuelle['rayon'] = r_milieu

        tentative = minimiseur(base_actuelle, reglages)
        valeurs_intermediaires += minimiseur.valeurs_intermediaires
        indicateur_acceleration = minimiseur.indicateur_acceleration
        

       # Test légitimité de la tentative (dichotomie classique)
        if legitime_bien_aligne(tentative, reglages):
            r_min = r_milieu
            meilleur_rayon = r_milieu
            meilleur_arrangement = tentative
            print("Tentative légitime avec ", r_milieu, ", on monte r_min")
        else:
            if indicateur_acceleration :
                print('on court-circuite la fin du programme en se sentant satisfait de nos resultats actuels')
                meilleur_arrangement = inch_allah(copy.deepcopy(tentative), reglages)
                meilleur_rayon = meilleur_arrangement['rayon']
                break
            r_max = r_milieu
            print("Tentative non légitime avec ",r_milieu,", on baisse r_max")

       # Toujours appliquer inch_allah après la tentative
        tentative_ajuste = inch_allah(copy.deepcopy(tentative), reglages)
        r_ajuste = tentative_ajuste['rayon']

       # Si l'ajustement permet d'améliorer r_min, on l'utilise
        print("On regarde ce que ça donne avec inch_allah")
        if legitime_bien_aligne(tentative_ajuste, reglages):
            print("Ajustement légitime, on obtient un rayon de :", r_ajuste)
        if r_ajuste >= r_min:
            print('on avait un meilleur rayon avec inch_allah')
            r_min = r_ajuste
           

            if r_ajuste >= r_max:
                print("encore meilleur que r_max")
                r_max = r_ajuste
                meilleur_rayon = r_ajuste
                meilleur_arrangement = copy.deepcopy(tentative_ajuste)

            meilleur_arrangement = copy.deepcopy(tentative_ajuste)
            meilleur_rayon = r_ajuste
            base_actuelle = copy.deepcopy(meilleur_arrangement)

            valeurs_intermediaires.append({
                'centres': meilleur_arrangement['centres'],
                'rayon': meilleur_rayon,
                'code couleur': 5,
                'texte': 'INCH\'ALLAH',
                'perte': 0,
                'itération du minimiseur': 0
            })

        print('On a actuellement une précision de ', r_max - r_min)
        print('On a actuellement un meilleur rayon de : ', meilleur_rayon)
        if r_max - r_min < reglages['epsilon']:
            print('Ecart < epsilon : on arrête la dichotomie ici')
            phase_dichotomie.valeurs_intermediaires = valeurs_intermediaires
            return meilleur_arrangement, meilleur_rayon
        base_actuelle = copy.deepcopy(meilleur_arrangement)


  ## SORTIE
    phase_dichotomie.valeurs_intermediaires = valeurs_intermediaires
    return meilleur_arrangement, meilleur_rayon

def phase_retentatives_fines(base, rayon_initial, reglages):
    ### PHASE FINALE : ON ESSAIE D’AUGMENTER LE RAYON AUTANT QUE POSSIBLE
    reglages_par_defaut = {
        'nb_iterations_fines': 10000,
        'reglages_fins' : 
            {
      'n_tours_minimiseur': 10,
      'n_tours_stag': float('inf'), 
      'version acceleree' : False, 
      'approximation version acceleree' :0.005,
      'borne stagnation nulle acceleree' : 20,
      'borne stagnation nulle': 1,
      'n_tours_stag_0' : 10,
      'n_tours_stag_0_acc':6}
    }
    if reglages is not None:
        reglages = reglages_par_defaut | reglages
    else:
        reglages = reglages_par_defaut

    reglages_int = copy.deepcopy(reglages)
    reglages_int['epsilon']= reglages_int['epsilon'] / 10
    reglages_int['iterations par tour'] = reglages_int['nb_iterations_fines']
    essai = inch_allah(base, reglages_int)
    if essai['rayon'] > rayon_initial :
        print('On a reussi à gratter un peu')
        meilleur_arrangement = essai
        meilleur_rayon = essai['rayon']
    else:
        meilleur_arrangement = base
        meilleur_rayon = rayon_initial
    valeurs_intermediaires = []

    r_test = 1.1*rayon_initial
    arr_test = copy.deepcopy(meilleur_arrangement)
    arr_test['rayon'] = r_test
    reglages_fins = copy.deepcopy(reglages) | reglages['reglages_fins']
    arr_test = minimiseur(arr_test, reglages_fins)
    valeurs_intermediaires += minimiseur.valeurs_intermediaires
    arr_test = inch_allah(arr_test, reglages_int) 
    if arr_test['rayon'] > rayon_initial :
        print('On a reussi à gratter un peu')
        meilleur_arrangement = arr_test
        meilleur_rayon = arr_test['rayon']

    print('On a un rayon final de : ', meilleur_rayon )

    ## SORTIE
    phase_retentatives_fines.valeurs_intermediaires = valeurs_intermediaires
    return meilleur_arrangement, meilleur_rayon

def chercher_rayon(arrangement_initial, reglages=None):
    ### PRÉPARATION
    debut=time.time()
    # On initialise la liste des valeurs intermédiaires pour visualisation
    valeurs_intermediaires = []

    # On applique les réglages par défaut si besoin
    reglages_par_defaut = {
        'epsilon': (0.001),
        
        'facteur_retentative': 1.01, 
        
    }
    if reglages is not None:
        reglages = reglages_par_defaut | reglages
    else:
        reglages = reglages_par_defaut
    print('on va travailler avec une précision de ', reglages['epsilon'])
    # On copie l’arrangement initial
    base = copy.deepcopy(arrangement_initial)

    # On fixe un rayon de départ
    rayon_depart = arrangement_initial['rayon']
    r_max = rayon_depart 
    # PHASE 1 : Recherche grossière du rayon maximal acceptable
    print('On entre dans la recherche de phase grossière')   
    r_max = phase_grossiere(base, r_max, reglages)
    print('On sort de la recherche de phase grossière')
    print('On a trouvé une borne superieure de : ' , r_max)
    valeurs_intermediaires += phase_grossiere.valeurs_intermediaires
    int1 = time.time()
    print('temps écoulé pour cette phase : ', int1-debut)

    # PHASE 2 : Recherche par dichotomie du meilleur rayon
    print('On entre dans la recherche de phase dichotomie')
    meilleur_arrangement, meilleur_rayon = phase_dichotomie(base, 0, r_max,  reglages)
    valeurs_intermediaires += phase_dichotomie.valeurs_intermediaires
    print('on a fini la recherche par dichotomie')
    print('Le meilleur rayon trouvé est : ', meilleur_rayon)
    print('On sort de la phase de recherche par dichotomie')
    int2 = time.time()
    print('temps écoulé pour cette phase : ', int2-int1)

    # PHASE 3 : Retentatives fines autour du meilleur rayon
    print('On entre dans la recherche de phase retentatives fines')
    print('On voit jusqu\'où on peut forcer')
    meilleur_arrangement, meilleur_rayon = phase_retentatives_fines(meilleur_arrangement, meilleur_rayon, reglages)
    int3 = time.time()
    print('temps écoulé pour cette phase : ', int3-int2)

    ### ATTRIBUTS ET RETOUR
    chercher_rayon.valeurs_intermediaires = valeurs_intermediaires

    if meilleur_arrangement is not None:
        return meilleur_arrangement
    else:
        raise ValueError("Aucun arrangement légitime acceptable n'a été trouvé.")

def main_axe_fixe(N, dimensions_carton, axe, origine_carton=[0, 0, 0], reglages=None, nom_arrangement = None):
 ### PRÉPARATION
    
    ## GESTION DES RÉGLAGES
        # on définit des réglages par défaut
    reglages_par_defaut = {
        'génération d\'arrangements': generation_aleatoire,'options' : {'maxiter' : 150, 'gtol' : 1e-8, 'eps' : 1e-8}, 'n_tours' : 10, 'coefficients densite' : 0.1, 'coefficient depassement' : 1000, 'pellicule' : 1.15,
      'precision' : 1e-6    
    }


    if reglages is not None:
        reglages = reglages_par_defaut | reglages
    else:
        reglages = reglages_par_defaut

    generateur = reglages['génération d\'arrangements']

 ### Genération de nom aléatoire
    import random
    import string

    def generer_chaine_aleatoire(longueur=5):
        """Génère une chaîne aléatoire de la longueur donnée, composée de lettres et chiffres."""
        caracteres = string.ascii_letters + string.digits
        return ''.join(random.choice(caracteres) for _ in range(longueur))

 ### NORMALISATION :
    dimensions_carton = np.array(dimensions_carton)
    plan = np.array([1,1,1])-np.array(axe)
    norm = 0.5*np.dot(plan, dimensions_carton)
    dimensions_carton = dimensions_carton /norm
    print('nouvelles dimensions pour les calculs : ' , dimensions_carton )

 ### GÉNÉRATION DE L’ARRANGEMENT INITIAL
    arrangement_initial = generateur(N, dimensions_carton, axe, reglages)
    arrangement_initial['origine du carton'] = origine_carton
    arrangement_initial = ajuster_rayon(copy.deepcopy(arrangement_initial), reglages)
    if nom_arrangement is not None:
        arrangement_initial['nom'] = nom_arrangement
    else:
        arrangement_initial['nom'] =  generer_chaine_aleatoire()
 ### CHERCHE RAYON MAXIMAL
    arrangement_final = chercher_rayon(arrangement_initial, reglages)
    arrangement_final['nombre de couches' ] = 1

 ### DENORMALISATION
    print('axe : ', arrangement_final['axe'])
    dimensions_carton_cible = dimensions_carton * norm
    arrangement_final = dilater_arrangement(arrangement_final, dimensions_carton_cible, [0,0,0], interaction = False)

 ### RETOUR
    main_axe_fixe.valeurs_intermediaires = chercher_rayon.valeurs_intermediaires
    return arrangement_final

def main(N, dimensions_carton, origine_carton=[0, 0, 0], reglages=None, mandrin_depassant = None) :
  #Gestion des reglages
  reglages_par_defaut = {
    'recherche de rayon' : main_axe_fixe
  }
  if reglages is not None:
    reglages = reglages_par_defaut | reglages
  else:
    reglages = reglages_par_defaut

  #initialisation des variables
  fonction_recherche = reglages['recherche de rayon'] 
  valeurs_intermediaires = [] 

  #On vérifie si les mandrins sont dépassants
  if mandrin_depassant is None:
    mandrin_depassant = False
    c = input('Les mandrins sont-ils dépassants ? (oui/non)')
    while c != 'oui' and c != 'non':
      c = input('Les mandrins sont-ils dépassants ? (oui/non)')
    if c == 'oui':
      mandrin_depassant = True
    else:
      mandrin_depassant = False

  #Cas où les mandrins dépassent
  if mandrin_depassant:
    r= 0
    solution = None
    for axe in [[0,1,0], [1,0,0]]: # On ne test pas tous les axes
      sol = fonction_recherche(N, dimensions_carton, axe )#axe, origine_carton, reglages )
      valeurs_intermediaires += fonction_recherche.valeurs_intermediaires 
      if sol['rayon'] > r:
        r = sol['rayon']
        solution = sol
  
  #Cas où ils ne dépassent pas
  else: 
    r= 0
    solution = None
    for axe in [[0,0,1], [0,1,0], [1,0,0]]: # On test tous les axes
      sol = fonction_recherche(N, dimensions_carton, axe ) #, axe, origine_carton, reglages)
      valeurs_intermediaires += fonction_recherche.valeurs_intermediaires
      if sol['rayon'] > r:
        r = sol['rayon']
        solution = sol

  main.valeurs_intermediaires = valeurs_intermediaires
  return solution
  
def main_axe_fixe_foireux(N, dimensions_carton, axe, origine_carton=[0, 0, 0], reglages=None, nom_arrangement = None):
    reglages_par_defaut = {'generation' : generation_aleatoire}
    if reglages is not None:
        reglages = reglages_par_defaut | reglages
    else:
        reglages = reglages_par_defaut
    generation = reglages['generation']
    arrg = generation(N, dimensions_carton, axe, reglages)
    solution = minimiseur_foireux(arrg, reglages)
    return solution    

