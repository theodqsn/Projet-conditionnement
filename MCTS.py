import bibos
import minimiseur_bibos_nouvelle_generation
import minimiseur_bibos
import gestion_reglages
import os
import sys
import importlib
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.patches import Rectangle, Circle
from matplotlib.animation import PillowWriter
from IPython.display import HTML
import scipy
from scipy import stats
import random
import copy
from collections import deque
from concurrent.futures import ProcessPoolExecutor
import threading


from bibos import *
from minimiseur_bibos_nouvelle_generation import *
from minimiseur_bibos import *
from gestion_reglages import *

class Noeud:
    
    def __init__(self, etat, parent=None, date_creation=0):
        self.etat = etat                      # L'état complet représenté
        self.parent = parent                  # Lien vers le parent
        self.enfants = []                     # Liste des enfants (Noeud)
        self.visites = 0                      # Compteur de visites
        self.score_total = 0.0                # Somme des scores passés
        self.score_pur = -float('inf')              
        self.actions_non_explorees = actions_possibles(etat)  # Actions restantes à tester
        self.date_creation = date_creation

    def est_exploratoire(self):
        parent = self.parent
        if parent is None :
            return True
        freres = parent.enfants
        visites_freres = [frere.visites for frere in freres]
        med = np.median(visites_freres)
        return self.visites < med / 2

def init_etat_initial(nombre_de_rouleaux, dimensions_du_carton, rayon_mandrin ):
    arg = generation_grille_bibos(nombre_de_rouleaux, dimensions_du_carton, [0, 0, 1], rayon_petit = rayon_mandrin)
    a = Action('tourner groupe', {'orientation': pi/4, 'indice groupe':0, 'profondeur': 0})
    etat = {'arrangement' : arg , 
            'actions entreprises' : [a], 
            'actions possibles' : [], 
            'profondeur' : 0}
    lister_actions(etat)
    return etat

class Action:

    def __init__(self, nom, parametres = None):
        self.nom = nom
        if parametres is None :
            self.parametres = {}
        else :
            self.parametres = parametres

    def in_liste(self, liste):
        nom = self.nom
        for action in liste :
            if action.nom == nom :
                if self.parametres == action.parametres :
                    return True
                elif self.nom == 'tourner groupe' : 
                    if 'saturation' in action.parametres and 'saturation' in self.parametres :
                        if self.parametres['saturation'] == action.parametres['saturation'] :
                            a=1
                            
                            return True
        return False
    
    def appliquer(self, arg):
        if self.nom == 'dephaser vertical':
            return dephaser_vertical(arg)
        elif self.nom == 'dephaser horizontal':
            return dephaser_horizontal(arg)
        elif self.nom == 'tourner groupe':
            profondeur = self.parametres['profondeur']
            indice_gpe = self.parametres['indice groupe']
            angle = self.parametres['orientation']
            groupes =  groupes_bibos(arg)
            groupes.repartir(profondeur+1)
            return tourner_groupe(arg, groupes.liste_groupes[indice_gpe], angle)

def determiner_lignes(arrangement_entree):
  '''renvoie [ligne_1,ligne_2,...]. ligne_i  est un array de n_i*3'''
  if not 'arrange en grille' in arrangement_entree or not arrangement_entree['arrange en grille']:
    raise Exception("L'arrangement n'est pas en grille, on ne peut donc pas déterminer les lignes.")
  i = 1
  arrangement = copy.deepcopy(arrangement_entree)
  lignes = [[arrangement['centres'][0]]]
  ordonnees = [arrangement['centres'][0][1]]
  indices = [[0]]

  while i < len(arrangement['centres']):
    
    ordonnee_actuelle =  arrangement['centres'][i][1]
    ajoutee = False
    for j, ordonnee in enumerate(ordonnees):
      if abs(ordonnee-ordonnee_actuelle)< 0.01 and not ajoutee:
        lignes[j].append(arrangement['centres'][i])
        indices[j].append(i)
        ajoutee = True
    if not ajoutee : 
      lignes.append([arrangement['centres'][i]])
      ordonnees.append(ordonnee_actuelle)
      indices.append([i])
    
    i +=1
  
  indices_tries = np.argsort(ordonnees)
  return({'listes triees': indices_tries, 'indices' : indices, 'ordonnees': ordonnees,  'lignes': lignes})

def determiner_colonnes(arrangement_entree):
  arrangement = copy.deepcopy(arrangement_entree)
  '''renvoie [colonne_1,colonne_2,...]. colonne_i  est un array de n_i*3'''
  if not 'arrange en grille' in arrangement or not arrangement['arrange en grille']:
    raise Exception("L'arrangement n'est pas en grille, on ne peut donc pas déterminer les colonnes.")
  i = 1
  colonnes = [[arrangement['centres'][0]]]
  abscisses = [arrangement['centres'][0][0]]
  indices = [[0]]

  while i < len(arrangement['centres']):
  
    abscisse_actuelle =  arrangement['centres'][i][0]
    ajoutee = False
    for j, abscisse in enumerate(abscisses):
      if abs(abscisse-abscisse_actuelle)< 0.01 and not ajoutee:
        colonnes[j].append(arrangement['centres'][i])
        indices[j].append(i)
        ajoutee = True
    if not ajoutee : 
      colonnes.append([arrangement['centres'][i]])
      abscisses.append(abscisse_actuelle)
      indices.append([i])

    i +=1
  
  indices_tries = np.argsort(abscisses) 
  return({'listes triees': indices_tries, 'indices' : indices, 'abscisses': abscisses,  'colonnes': colonnes})

def dephaser_horizontal(arrangement):
  epsilon = reglages_MCTS('decalage dephasage')
  dico_lignes = determiner_lignes(arrangement)
  if epsilon == 'auto':
    d = float('inf')
    for i in range(len(arrangement['centres'])) :
      for j in range(i+1, len(arrangement['centres'])):
        for k in range(3):
          if abs(arrangement['centres'][i][k]-arrangement['centres'][j][k]) > 1e-3 and abs(arrangement['centres'][i][k]-arrangement['centres'][j][k]) < d:
            d  = abs(arrangement['centres'][i][k]-arrangement['centres'][j][k])
    epsilon = d/3
  for i in range(len(dico_lignes['listes triees'])):
    for j in range(len(dico_lignes['lignes'][dico_lignes['listes triees'][i]])):
      ind_i = dico_lignes['listes triees'][i]
      ind_j = j
      if i % 2 == 0:
        dico_lignes['lignes'][ind_i][ind_j][0] += epsilon
      else:
        dico_lignes['lignes'][ind_i][ind_j][0] -= epsilon

  nouveaux_centres = copy.deepcopy(arrangement['centres'])
  for i in range(len(dico_lignes['lignes'])):
    for j in range(len(dico_lignes['lignes'][i])):
      indice = dico_lignes['indices'][i][j]
      nouveaux_centres[indice] = dico_lignes['lignes'][i][j]

  nouvel_arrangement = copy.deepcopy(arrangement)
  nouvel_arrangement['centres'] = nouveaux_centres
  return(nouvel_arrangement)

def dephaser_vertical(arrangement):
  epsilon = reglages_MCTS('decalage dephasage')
  dico_colonnes = determiner_colonnes(arrangement)
  if epsilon == 'auto':
    d = float('inf')
    for i in range(len(arrangement['centres'])) :
      for j in range(i+1, len(arrangement['centres'])):
        for k in range(3):
          if abs(arrangement['centres'][i][k]-arrangement['centres'][j][k]) > 1e-3 and abs(arrangement['centres'][i][k]-arrangement['centres'][j][k]) < d:
            d  = abs(arrangement['centres'][i][k]-arrangement['centres'][j][k])
    epsilon = d/3
  for i in range(len(dico_colonnes['listes triees'])):
    for j in range(len(dico_colonnes['colonnes'][dico_colonnes['listes triees'][i]])):
      ind_i = dico_colonnes['listes triees'][i]
      ind_j = j
      if i % 2 == 0:
        dico_colonnes['colonnes'][ind_i][ind_j][1] += epsilon
      else:
        dico_colonnes['colonnes'][ind_i][ind_j][1] -= epsilon

  nouveaux_centres = copy.deepcopy(arrangement['centres'])
  for i in range(len(dico_colonnes['indices'])):
    for j in range(len(dico_colonnes['indices'][i])):
      indice = dico_colonnes['indices'][i][j]
      nouveaux_centres[indice] = dico_colonnes['colonnes'][i][j]

  nouvel_arrangement = copy.deepcopy(arrangement)
  nouvel_arrangement['centres'] = nouveaux_centres
  dephaser_vertical.dico_colonnes = dico_colonnes
  return(nouvel_arrangement)

def tourner_groupe(arrangement, indices, angle):
  ''' met l'angle des bibos d'indices dans indices à angle'''
  arg = copy.deepcopy(arrangement)
  for i in indices:
    arg['orientations'][i] = angle
  return(arg)

  return(indices)

def lister_actions(etat):
    tap = reglages_MCTS("types actions possibles")
    actions_possibles = []
    actions_entreprises = etat["actions entreprises"]

    def dephaser_vert():
      a = Action('dephaser vertical')
      if not a.in_liste(actions_entreprises) :
        actions_possibles.append(a)

    def dephaser_hor():
      a = Action('dephaser horizontal')
      if not a.in_liste(actions_entreprises) :
        actions_possibles.append(a)

    def tourner_gpe():
      op = reglages_MCTS("orientations possibles")
      arg = etat['arrangement']
      profondeur = etat['profondeur']
      groupes =  groupes_bibos(arg)
      groupes.repartir(profondeur+1)
      for orientation in op :
          n_gpes = len(groupes.liste_groupes)
          for i in range(n_gpes):
            if min(len(etat['arrangement']['centres']), n_gpes):
                # On est dans un cas de saturation
                a = Action('tourner groupe', {'orientation': orientation, 'indice groupe' : i , 'profondeur' :  etat['profondeur'], 'saturation': {'orientation': orientation, 'indice groupe' : i} })
            else: 
                a = Action('tourner groupe', {'orientation': orientation, 'indice groupe' : i , 'profondeur' :  etat['profondeur']})
            if not a.in_liste(actions_entreprises) :
                actions_possibles.append(a)

    dico_actions = {
        'dephaser vertical': dephaser_vert,
        'dephaser horizontal': dephaser_hor,
        'tourner groupe': tourner_gpe
    }

    for type_action in tap :
        if type_action in dico_actions:
            dico_actions[type_action]()
        else : 
            if hasattr(lister_actions, 'actions_non_reconnues'):
                if not type_action in lister_actions.actions_non_reconnues:
                    lister_actions.actions_non_reconnues.append(type_action)
                    print(f"Type d'action '{type_action}' non reconnu.")
                    print('Veuillez l\'ajouter dans le programme \"lister_actions\".')
            else:
                lister_actions.actions_non_reconnues = [type_action]
                print(f"Type d'action '{type_action}' non reconnu.")
                print('Veuillez l\'ajouter dans le programme \"lister_actions\".')

    etat['actions possibles']  = actions_possibles

def actions_possibles(etat):
    return etat['actions possibles']

def appliquer_action(etat, action):
    nouvel_etat = copy.deepcopy(etat)
    nouvel_etat['actions entreprises'].append(action)
    nouvel_etat['profondeur'] += 1
    nouvel_etat['actions possibles'] = []
    nouvel_etat['arrangement'] = action.appliquer(nouvel_etat['arrangement'])
    lister_actions(nouvel_etat)
    return nouvel_etat

class groupes_bibos :

  def __init__(self, arrangement, limite_voisins = 1.1):
    self.arrangement = arrangement
    self.liste_groupes = []
    self.limite_voisins = limite_voisins
    self.n = len(arrangement['orientations'])
    self.matrice_voisins = np.zeros((self.n,self.n))
    self.matrice_distances = np.zeros((self.n,self.n))
    self.matrice_directions = np.array([[None for i in range(self.n)] for j in range(self.n)])
    self.matrice_distances_voisins = np.zeros((self.n,self.n)) 
    self.non_attribues = [i for i in range(self.n)]

  def distances(self):
    for i in range(self.n):
      for j in range(i+1,self.n):
          v = np.array(self.arrangement['centres'][i])-np.array(self.arrangement['centres'][j])
          a = np.linalg.norm(v)
          self.matrice_distances[i][j] = a
          self.matrice_distances[j][i] = a
          if abs(np.dot(v, np.array([1,0,0]))) > abs(np.dot(v, np.array([0,1,0]))):
            self.matrice_directions[i][j] = 'horizontal'
            self.matrice_directions[j][i] = 'horizontal'
          else:
            self.matrice_directions[i][j] = 'vertical'
            self.matrice_directions[j][i] = 'vertical'

  def voisins (self): 
    self.distances()
    min_dist_v = [float('inf')]*self.n    
    min_dist_h = [float('inf')]*self.n

    for i in range(self.n):
      for j in range(self.n):

        if self.matrice_distances[i][j] < min_dist_v[i] and self.matrice_distances[i][j]>0 and self.matrice_directions[i][j]== 'vertical':
          min_dist_v[i] = self.matrice_distances[i][j]
      for j in range(self.n):
        if self.matrice_distances[i][j] < self.limite_voisins*min_dist_v[i] and self.matrice_directions[i][j]== 'vertical':
          self.matrice_voisins[i][j] = 1
          self.matrice_voisins[j][i] = 1

        if self.matrice_distances[i][j] < min_dist_h[i] and self.matrice_distances[i][j]>0 and self.matrice_directions[i][j]== 'horizontal':
          min_dist_h[i] = self.matrice_distances[i][j]
      for j in range(self.n):
        if self.matrice_distances[i][j] < self.limite_voisins*min_dist_h[i] and self.matrice_directions[i][j]== 'horizontal':
          self.matrice_voisins[i][j] = 1
          self.matrice_voisins[j][i] = 1

  def distances_voisins(self):
    self.voisins()
    matrice_adjacence = self.matrice_voisins
    n = self.n
    distances = np.full((n, n), np.inf)

    for i in range(n):
        distances[i][i] = 0
        queue = deque([i])
        vus = {i}
        while queue:
            courant = queue.popleft()
            for voisin, lien in enumerate(matrice_adjacence[courant]):
                if lien != 0 and voisin not in vus:
                    distances[i][voisin] = distances[i][courant] + 1
                    queue.append(voisin)
                    vus.add(voisin)

    self.matrice_distances_voisins = distances.astype(int)

  def nouveau_gpe(self, d_voisins):

    a_remove = []

    # On ajoute le premier representant du groupe
    premier_representant = self.non_attribues[0]
    a_remove.append(premier_representant)
    self.liste_groupes.append([premier_representant])
    ajout = True
    on_peut_les_retirer = [True for _ in self.non_attribues]
    for j, sommet_seul in enumerate(self.non_attribues):
      if self.matrice_distances_voisins[premier_representant][sommet_seul] < d_voisins:
          # On interdit tous ceux qui sont trop proches de ce premier représentant
          on_peut_les_retirer[j] = False

    i = 0 # permet de savoir à quel sommet de la liste_groupe[-1] on est rendus

    
    while i<len(self.liste_groupes[-1]): #tant qu'il y a des sommets non traités dans le groupe, continue
      sommet = self.liste_groupes[-1][i] # sommet courant
      ajout = False

      for s, sommet_seul in enumerate(self.non_attribues):

        if self.matrice_distances_voisins[sommet][sommet_seul] == d_voisins and on_peut_les_retirer[s] :
          self.liste_groupes[-1].append(sommet_seul)
          on_peut_les_retirer[s]= False
          a_remove.append(sommet_seul)
          ajout = True
          for j, autre_sommet_seul in enumerate(self.non_attribues):
            if self.matrice_distances_voisins[autre_sommet_seul][sommet_seul] < d_voisins:
              on_peut_les_retirer[j] = False
        
      
      if ajout == False:
        i+=1
    for sommet in a_remove:
      self.non_attribues.remove(sommet)

  def repartir(self, d_voisins):
    self.distances_voisins()
    while self.non_attribues:
      self.nouveau_gpe(d_voisins)

def uct_global(noeud, total_visites, C=1.414, alpha=0):
    if noeud.visites == 0:
        return float('inf')
    exploitation = noeud.score_total / noeud.visites
    C_ajuste = C / (1 + alpha * noeud.etat['profondeur'])
    exploration = C_ajuste * math.sqrt(math.log(total_visites + 1) / noeud.visites)
    return exploitation + exploration
  
def evaluer_MCTS(etat, reglages_inch= None) :
    arg_initial = etat['arrangement']
    arg_final = inch_allah_lisse_bibos(arg_initial)
    recompense = arg_final['grand rayon']
    evaluer_MCTS.arrangement = arg_final
    return recompense
  
def dessiner_arbre(racine, nom_fichier_dot='arbre.dot'):
    import uuid
    from io import StringIO

    def label_noeud(noeud):
        return f"[🔗score: {noeud.score_pur:.4f}, créé à t={noeud.date_creation}]"

    def action_multiligne(action):
        if action is None:
            return ["∅"]
        lignes = [f"🛠️ {action.nom}"]
        for k, v in action.parametres.items():
            lignes.append(f"{k}: {v}")
        return lignes

    def extraire_action(noeud):
        try:
            return noeud.etat['actions entreprises'][-1]
        except (KeyError, IndexError):
            return None

    # ==== ASCII ====
    def ascii(noeud, prefix=""):
        label = label_noeud(noeud)
        print(prefix + label)
        for i, enfant in enumerate(noeud.enfants):
            is_last = (i == len(noeud.enfants) - 1)
            connector = "└── " if is_last else "├── "
            extension = "    " if is_last else "│   "
            action = extraire_action(enfant)
            lignes_action = action_multiligne(action)
            print(prefix + connector + lignes_action[0])
            for l in lignes_action[1:]:
                print(prefix + extension + "   " + l)
            ascii(enfant, prefix + extension)

    # ==== DOT ====
    dot_output = StringIO()
    dot_output.write('digraph G {\n')
    node_ids = {}
    counter = [0]

    def add_dot_nodes_edges(noeud):
        node_id = f"n{counter[0]}"
        counter[0] += 1
        node_ids[noeud] = node_id
        label = label_noeud(noeud).replace('"', '\\"')
        dot_output.write(f'    {node_id} [label="{label}"];\n')

        for enfant in noeud.enfants:
            add_dot_nodes_edges(enfant)
            action = extraire_action(enfant)
            lignes = action_multiligne(action)
            label_edge = "\\n".join(lignes).replace('"', '\\"')
            dot_output.write(f'    {node_id} -> {node_ids[enfant]} [label="{label_edge}"];\n')

    print("=== Arbre ASCII ===")
    ascii(racine)

    add_dot_nodes_edges(racine)
    dot_output.write('}\n')
    nom_fichier_dot +='.dot'
    with open(nom_fichier_dot, 'w', encoding='utf-8') as f:
        f.write(dot_output.getvalue())

    print(f"\nFichier .dot exporté : {nom_fichier_dot}")
    print("📎 Pour le visualiser : https://dreampuf.github.io/GraphvizOnline/")

def decrire_etat(etat, dessiner=False):
    """
    Affiche proprement le contenu d’un état :
    - dessine l’arrangement si demandé
    - affiche la profondeur
    - affiche les actions entreprises (nom + paramètres)
    - affiche les actions possibles (nom + paramètres)
    """
    
    # Dessin de l’arrangement si demandé
    if dessiner:
        dessiner_gradient(etat["arrangement"])
    
    # Affichage de la profondeur
    print(f"\n🧭 Profondeur : {etat['profondeur']}\n")
    
    # Affichage des actions entreprises
    print("📜 Actions entreprises :")
    if etat["actions entreprises"]:
        for i, action in enumerate(etat["actions entreprises"], 1):
            print(f"  {i}. {action.nom} — paramètres : {action.parametres}")
    else:
        print("  (Aucune)")
    
    # Affichage des actions possibles
    print("\n🔮 Actions possibles :")
    if etat["actions possibles"]:
        for i, action in enumerate(etat["actions possibles"], 1):
            print(f"  {i}. {action.nom} — paramètres : {action.parametres}")
    else:
        print("  (Aucune)\n")

def init_indicateurs_mtcs():
  init_indicateurs_mtcs.histo = []

def get_indicateurs_mtcs():
  if hasattr(init_indicateurs_mtcs, 'histo'):
    h = init_indicateurs_mtcs.histo
    get_indicateurs_mtcs.histo = h
    delattr(init_indicateurs_mtcs, 'histo')
  elif hasattr(get_indicateurs_mtcs, 'histo'):
    h = get_indicateurs_mtcs.histo
    print('Attention, version probablement périmée')
  else :
    raise Exception('pas d\'indicateurs dans la mémoire')

  return(h)

  def best_leaf_mcts(etat_racine, nb_iterations, objectif_exploration = 1/3, controle_exploration = 3, augmentation_controle_profondeur = 0.5, diminution_controle_profondeur = 0.9):

  ### Préparation des objets 

   ## On crée la racine, et on commence la liste des noeuds existants mais non encore evalués 
    racine = Noeud(etat_racine)
    feuilles = [racine]

   ## On initialise les variables qui nous permettront de controler l'algo
    nombre_explorations = 0
    profondeur_max = 0
    meilleur_score = - float('inf')
    profondeur_precedente = 0

    # C sert à contrôler la priorité accordée à l'exploration par rapport à l'exploitation
    logC = math.log(1.414)  
    C = math.exp(logC)

    #  alpha sert à décourager de creuser une branche si celle ci n'offre pas un potentiel incroyable
    alpha = 0    

   ## On va chercher les réglages choisis dans un fichier séparé, dans modules->reglages_MCTS.py
    reglages_inch = reglages_MCTS('reglages_inch')

  ### Boucle principale
    for iteration in range(nb_iterations):


      ## PHASE 1 : On selectionne le meilleur nœud parmi toutes les feuilles expansibles

        candidates = [n for n in feuilles if n.actions_non_explorees]
        if not candidates:
            break  # Tous les nœuds sont complètement explorés
        total_visites = racine.visites + 1
        # Calcul des scores UCT pour chaque candidat, ainsi que du meilleur score
        uct_scores = [uct_global(n, total_visites, C, alpha) for n in candidates]
        max_score = max(uct_scores)

        # On prend tous les noeuds ayant le meilleur score (à un epsilon près)
        epsilon = 1e-4
        meilleurs = [n for n, s in zip(candidates, uct_scores) if abs(s - max_score) < epsilon or s == max_score]
        # On choisit aléatoirement un noeud à étendre parmi les meilleurs parmi les meilleurs
        noeud = random.choice(meilleurs)

      ## PHASE 2 : Expansion

        # On choisit une action au hasard
        ind = random.randint(0,len(noeud.actions_non_explorees)-1)
        action = noeud.actions_non_explorees.pop(ind)

        # On crée un enfant à partir de cette action
        nouvel_etat = appliquer_action(noeud.etat, action)
        enfant = Noeud(nouvel_etat, parent=noeud, date_creation = iteration)
        enfant.etat['actions entreprises'] = copy.deepcopy(nouvel_etat['actions entreprises'])
        noeud.enfants.append(enfant)
        feuilles.append(enfant)

        # On regarde si on est en train de creuser (pour l'ajustement de alpha)
        if nouvel_etat['profondeur'] > profondeur_precedente:
            profondeur_max = nouvel_etat['profondeur']
            creuse = True
        else :
            creuse = False
        profondeur_precedente = nouvel_etat['profondeur']

      ## PHASE 3 : Ajustement dynamique des coefficients de contrôle C et alpha

        # On controle le coefficient C
        if noeud.est_exploratoire():  
          
          nombre_explorations += 1
          prop_exploration = nombre_explorations / (iteration + 1)
          logC += controle_exploration * (objectif_exploration - prop_exploration)
          C = math.exp(logC)

        # On controle le coefficient alpha 
        if creuse and enfant.score_pur <= noeud.score_pur :
          alpha += augmentation_controle_profondeur
        else : 
          alpha *= diminution_controle_profondeur

      ## PHASE 4 : On calcule le rayon atteignable avec la configuration initiale correspondant au nouveau noeud
        score = evaluer(enfant.etat, reglages_inch)
        enfant.score_pur = score  # On met à jour score_pur ici
        if score > meilleur_score:
            meilleur_score = score
            meilleur_noeud = enfant

      ## PHASE 5 : On remonte jusqu'à la racine pour indiquer qu'on a visité un descendant des ancêtres du noeud
        courant = enfant
        while courant is not None:
            courant.visites += 1
            courant.score_total += score
            courant = courant.parent

      ## PHASE 6 : Gestion des indicateurs (on les ajoute à la liste init_indicateurs_mtcs.histo afin de pouvoir suivre la construction de l'arbre)
        if hasattr(init_indicateurs_mtcs, 'histo'):
          init_indicateurs_mtcs.histo.append({'tour': iteration,'controlabilite exploitation (C)': C, 'profondeur' : copy.deepcopy(enfant.etat['profondeur']), 'proportion exploration': prop_exploration, 'meilleur score': meilleur_score, 'score': score, 'profondeur max': profondeur_max, 'controle profondeur (α)' : alpha })

  ### On retourne le noeud avec le meilleur score_pur atteint

    noeuds_explores = [n for n in feuilles if n.score_pur != float('-inf')]
    best_leaf_mcts.proportion_exploration = prop_exploration
    best_leaf_mcts.arbre = racine
    if noeuds_explores:
        return max(noeuds_explores, key=lambda n: n.score_pur)
    else:
        return racine

def best_leaf_mcts(etat_racine, nb_iterations, objectif_exploration = 1/3, controle_exploration = 3, augmentation_controle_profondeur = 0.5, diminution_controle_profondeur = 0.9):

  ### Préparation des objets 

   ## On crée la racine, et on commence la liste des noeuds existants mais non encore evalués 
    racine = Noeud(etat_racine)
    feuilles = [racine]

   ## On initialise les variables qui nous permettront de controler l'algo
    nombre_explorations = 0
    profondeur_max = 0
    meilleur_score = - float('inf')
    profondeur_precedente = 0

    # C sert à contrôler la priorité accordée à l'exploration par rapport à l'exploitation
    logC = math.log(1.414)  
    C = math.exp(logC)

    #  alpha sert à décourager de creuser une branche si celle ci n'offre pas un potentiel incroyable
    alpha = 0    

   ## On va chercher les réglages choisis dans un fichier séparé, dans modules->reglages_MCTS.py
    reglages_inch = reglages_MCTS('reglages_inch')

  ### Boucle principale
    for iteration in range(nb_iterations):


      ## PHASE 1 : On selectionne le meilleur nœud parmi toutes les feuilles expansibles

        candidates = [n for n in feuilles if n.actions_non_explorees]
        if not candidates:
            break  # Tous les nœuds sont complètement explorés
        total_visites = racine.visites + 1
        # Calcul des scores UCT pour chaque candidat, ainsi que du meilleur score
        uct_scores = [uct_global(n, total_visites, C, alpha) for n in candidates]
        max_score = max(uct_scores)

        # On prend tous les noeuds ayant le meilleur score (à un epsilon près)
        epsilon = 1e-4
        meilleurs = [n for n, s in zip(candidates, uct_scores) if abs(s - max_score) < epsilon or s == max_score]
        # On choisit aléatoirement un noeud à étendre parmi les meilleurs parmi les meilleurs
        noeud = random.choice(meilleurs)

      ## PHASE 2 : Expansion

        # On choisit une action au hasard
        ind = random.randint(0,len(noeud.actions_non_explorees)-1)
        action = noeud.actions_non_explorees.pop(ind)

        # On crée un enfant à partir de cette action
        nouvel_etat = appliquer_action(noeud.etat, action)
        enfant = Noeud(nouvel_etat, parent=noeud, date_creation = iteration)
        enfant.etat['actions entreprises'] = copy.deepcopy(nouvel_etat['actions entreprises'])
        noeud.enfants.append(enfant)
        feuilles.append(enfant)

        # On regarde si on est en train de creuser (pour l'ajustement de alpha)
        if nouvel_etat['profondeur'] > profondeur_precedente:
            profondeur_max = nouvel_etat['profondeur']
            creuse = True
        else :
            creuse = False
        profondeur_precedente = nouvel_etat['profondeur']

      ## PHASE 3 : Ajustement dynamique des coefficients de contrôle C et alpha

        # On controle le coefficient C
        if noeud.est_exploratoire():  
          
          nombre_explorations += 1
          prop_exploration = nombre_explorations / (iteration + 1)
          logC += controle_exploration * (objectif_exploration - prop_exploration)
          C = math.exp(logC)

        # On controle le coefficient alpha 
        if creuse and enfant.score_pur <= noeud.score_pur :
          alpha += augmentation_controle_profondeur
        else : 
          alpha *= diminution_controle_profondeur

      ## PHASE 4 : On calcule le rayon atteignable avec la configuration initiale correspondant au nouveau noeud
        score = evaluer_MCTS(enfant.etat, reglages_inch)
        enfant.score_pur = score  # On met à jour score_pur ici
        if score > meilleur_score:
            meilleur_score = score
            meilleur_noeud = enfant

      ## PHASE 5 : On remonte jusqu'à la racine pour indiquer qu'on a visité un descendant des ancêtres du noeud
        courant = enfant
        while courant is not None:
            courant.visites += 1
            courant.score_total += score
            courant = courant.parent

      ## PHASE 6 : Gestion des indicateurs (on les ajoute à la liste init_indicateurs_mtcs.histo afin de pouvoir suivre la construction de l'arbre)
        if hasattr(init_indicateurs_mtcs, 'histo'):
          init_indicateurs_mtcs.histo.append({'tour': iteration,'controlabilite exploitation (C)': C, 'profondeur' : copy.deepcopy(enfant.etat['profondeur']), 'proportion exploration': prop_exploration, 'meilleur score': meilleur_score, 'score': score, 'profondeur max': profondeur_max, 'controle profondeur (α)' : alpha })

  ### On retourne le noeud avec le meilleur score_pur atteint

    noeuds_explores = [n for n in feuilles if n.score_pur != float('-inf')]
    best_leaf_mcts.proportion_exploration = prop_exploration
    best_leaf_mcts.arbre = racine
    if noeuds_explores:
        return max(noeuds_explores, key=lambda n: n.score_pur)
    else:
        return racine



















