

import numpy as np

from outils_courants import *
from arrangement_legitimite import *

def creer_bdd_arrangement(arrangements, nom_fichier):
  import json
  with open(nom_fichier, 'w') as f:
    json.dump(arrangements, f)

def ajouter_arrangement(arrangement, nom_fichier='Base de données des arrangements.json'):
  '''
  Lorsque l'on dispose d'un arrangement, cettte fonction l'ajoute à la base de données
   - arrangement : dictionnaire contenant l'arrangement
      Dans un arrangement, on doit trouver :
        - nom : nom de l'arrangement
        - axe : l'axe selon lequel les cylindres sont rangés
        - nombre : nombre de cylindres
        - centres : liste des centres des cylindres. Ce doit être un tableau de dimensions (nombre,3)
        - rayon : rayon commun aux cylindres
        - longueur : longueur commune aux cylindres
        - dimensions du carton : dimensions du carton dans lequel les cylindres sont rangés
        - origine du carton : origine du carton dans lequel les cylindres sont rangés
   - nom_fichier : nom du fichier contenant la base de données. Par défaut, celui de la base de données des arrangements déjà créé, qui s'appelle 'Base de données des arrangements.json'
  '''
  
  #ce morceau sert à corriger les eventuelles fautes de frappe lorsque l'utilisateur entre des couples cles/valeurs
  liste_des_cles =['nom','axe' ,'nombre' ,'centres','rayon','longueur','dimensions du carton','origine du carton', 'nombre de couches' ]
  cles_entrees = list(arrangement.keys())
  for cle in cles_entrees:
    if cle not in liste_des_cles:
      print(cle, 'n\'est pas un champ standard')
      choix ='feur'
      while choix != 'oui' and choix != 'non' :
        choix = input('Voulez vous le conserver sous ce nom (il ne sera pas utilisé par les programmes) (oui/non)  ')
      if choix == 'non':
        cle_propre = chaine_la_plus_proche(cle,liste_des_cles)
        arrangement[cle_propre] = arrangement[cle]
        arrangement.pop(cle)
  for cle in liste_des_cles : 
    if cle not in arrangement and cle != 'nombre':
      print('Vous n \' avez pas renseigné le champ ', cle)
      arrangement[cle] = eval(input('Que voulez vous y mettre ?  '))

  arrangement['nombre'] = len(arrangement['centres'])

  # Convertit NumPy arrays en listes avant de les encoder (JSON ne reconnaît pas les arrays)
  for key in arrangement:
    if isinstance(arrangement[key], np.ndarray):
      arrangement[key] = arrangement[key].tolist()

  #on réorganise l'ordre des clés dans le dico pour faire propre
  liste_des_cles =['nom','nombre','nombre de couches', 'rayon','longueur','axe','dimensions du carton' ,'origine du carton' ,'centres' ]
  arr_orga = {}
  for cle in liste_des_cles : 
    if cle in arrangement:
      arr_orga[cle] = arrangement[cle]
  for cle in arrangement:
    if cle not in liste_des_cles :
      arr_orga[cle] = arrangement[cle]
  arrangement = arr_orga

  #ce morceau sert à vérifier qu'il n'y a pas déjà un arrangement du même nom
  import json
  with open(nom_fichier, 'r') as f:
    arrangements = json.load(f)

  #On se donne une fonction qui est capable de checker si il y a y autre arrangement du même nom
  def appartenance(nom_d_arrangement):
    for arrangt in arrangements:
      if arrangt['nom'] == nom_d_arrangement:
        ancien = arrangt
        appartenance.ancien_arrg = ancien
        return True
    return False

  flag= False
  if appartenance(arrangement['nom']) == False:
    arrangements.append(arrangement)
  else:
    flag = True
    ancien_arrg = appartenance.ancien_arrg

  #si il existe déjà un arrangement du même nom, on demande à l'utilisateur ce qu'il veut faire
  #Quand le problème est réglé, le nouvel arrangement est ajouté
  while flag:
    choix = '0'
    while choix != '1' and choix != '2' and choix != '3':
      print('Cet arrangement existe déjà')
      print(ancien_arrg)
      print('vous desirez : ')
      print('l\'ajouter sous un autre nom : tapez 1')
      print('l\' ajouter sous ce nom : tapez 2')
      print('ne pas l\'ajouter : tapez 3')
      choix = input('faites votre choix : ')


      if choix == '1':
        nouveau_nom_propose = input('Comment voulez vous nommer ce nouvel arrangement')
        if appartenance(nouveau_nom_propose) == False:
          arrangement['nom'] = nouveau_nom_propose
          arrangements.append(arrangement)
          flag = False
        else:
          print('Cet arrangement existe déjà')


      if choix == '2':
        chx = 'feur'
        while chx != 'oui' and chx != 'non' :
          chx = input('voulez vous renommer l\'arrangement actuel ? oui ou non (si non, il sera supprimé)')
          if chx == 'oui':
            nouveau_nom = input('Comment voulez vous renommer l\'ancien arrangement')
            ind_arg = arrangements.index(ancien_arrg)
            ancien_arrg = arrangements.pop(ind_arg) #on sort l'arrangement de la liste pour pouvoir vérifier qu'il n'en existe pas un autre avec le même nom
            if appartenance(nouveau_nom) == False:
              ancien_arrg['nom'] = nouveau_nom #on modifie le nom
              arrangements.append(ancien_arrg) #on le remet dans la liste des arrangements
              flag = False
            else:
              print('Cet arrangement existe déjà')
              print('Rien n\'a été modifié')
              arrangements.append(ancien_arrg)
              flag = False
          if chx == 'non':
            ind_arg = arrangements.index(ancien_arrg)
            arrangements.pop(ind_arg)
            arrangements.append(arrangement)
            flag = False

      if choix == '3':
        print('Rien n\'a été modifié')
        flag = False  

  #on enregistre la modification dans le fichier
  with open(nom_fichier, 'w') as f:
    json.dump(arrangements, f)

def ouvrir_arrangements(nom_arrangement= None, nom_fichier='Base de données des arrangements.json' ):
  import json
  with open(nom_fichier, 'r') as f:
    arrangements = json.load(f)

  if nom_arrangement == None:
    return arrangements
  else:
    noms_des_arrangements  = [arrangement['nom'] for arrangement in arrangements]
    nom_arrangement = chaine_la_plus_proche(nom_arrangement, noms_des_arrangements)
    for arrangement in arrangements:
      if arrangement['nom'] == nom_arrangement:
        return arrangement

def retirer_arrangement(nom_arrangement = None, nom_fichier='Base de données des arrangements.json'):
  '''
  Lorsque l'on souhaite retirer un arrangement d'une base de donnée, cette fonction la supprime
   - nom_arrangement : nom de l'arrangement que l'on souhaite retirer
   - nom_fichier : nom du fichier contenant la base de données. Par défaut, celui de la base de données des arrangements déjà créé, qui s'appelle 'Base de données des arrangements.json'
  '''
  if nom_arrangement == None:
    nom_arrangement = input('Quel arrangement voulez vous retirer ? ')

  flag = True
  import json
  with open(nom_fichier, 'r') as f:
    arrangements = json.load(f)
  for arrangement in arrangements:
    if arrangement['nom'] == nom_arrangement:
      arrangements.remove(arrangement)
      flag = False
  if flag :
    print('Attention : aucun fichier n\'a été retiré')
    print('vérifiez que vous avez entré le bon nom')

  else :
    print('fichier retiré avec succès')
  print(arrangements)
  with open(nom_fichier, 'w') as f: 
    json.dump(arrangements, f)

def modifier_arrangement(nom_arrangement = None, nom_cible =None,axe=None ,nombre = None ,centres=None,rayon = None, longueur=None, dimensions_carton = None , origine_carton = None, n_couches=None ,nom_fichier='Base de données des arrangements.json'):
  nom_arrangement = input('Quel arrangement voulez vous modifier ? ')
  arrangement= ouvrir_arrangements(nom_arrangement)
  nom_arrangement = arrangement['nom']
  proceed = True 
  arrangement_sortie = arrangement.copy()
  liste_cle = ['nom','axe' ,'nombre' ,'centres','rayon','longueur','dimensions du carton','origine du carton', 'nombre de couches']
  def conv(cle): #convertis une clé en la variable equivalente
    if cle == 'nom ':
      return nom_cible
    if cle == 'axe':
      return axe
    if cle == 'nombre':
      return nombre
    if cle == 'centres':
      return centres
    if cle == 'rayon':
      return rayon
    if cle == 'longueur':
      return longueur
    if cle == 'dimensions du carton':
      return dimensions_carton
    if cle == 'origine du carton':
      return origine_carton
    if cle == 'nombre de couches':
      return n_couches
    else :
      return None

  while proceed:
    
    choix = 'dgbijfnv'
    while choix not in arrangement :
      choix = input('Que voulez vous modifier (tapez le nom du champ) ?  ')
      for cle in arrangement :
        if choix == cle :
          print('vous avez choisi le champ', cle)
          print('voici la valeur actuelle : ', arrangement[cle])
          if conv(cle) == None :
            val = eval(input('Quelle valeur voulez vous lui donner ?  '))
            arrangement_sortie[cle] = val
          else:
            print('Vous avez proposé la valeur', conv(cle))
            c = input('voulez vous lui donner cette valeur ? (oui/non)  ')
            c = chaine_la_plus_proche(c, ['oui', 'non'])
            if c == 'oui':
              arrangement_sortie['nom'] = conv(cle)
    
    continuer = input('Voulez vous continuer à modifier cet arrangement (oui/non)  ')
    if continuer == 'oui':
      proceed = True
    if continuer == 'non':
      proceed = False


  sauver = 'feur'
  while sauver != 'oui' and sauver != 'non' :
    sauver = input('voulez vous sauver les modifications (oui/non)  ')
    if sauver == 'oui':
      retirer_arrangement(nom_arrangement, nom_fichier)
      ajouter_arrangement(arrangement_sortie, nom_fichier)

def dilater_arrangement(arrangement_source, dimensions_carton_cible, origine_carton_cible, interaction = True, nom_fichier='Base de données des arrangements.json' ):
  proportions = np.zeros(3)
  dimensions_carton_source = np.array(arrangement_source['dimensions du carton'])
  origine_carton_source = np.array(arrangement_source['origine du carton'])
  centres_source = np.array(arrangement_source['centres'])
  rayon_source = (arrangement_source['rayon'])
  longueur_source = (arrangement_source['longueur'])
  axe_source  = (arrangement_source['axe'])

  #on extrait les vecteurs qui portent le plan, ainsi que leurs indices 
  axes = np.eye(3)
  axes_du_plan= np.zeros((2,3))
  ind= 0
  indices_direction_plan = np.zeros(2, np.int16)
  indice_direction_axe = np.dot([0,1,2],axe_source)
  for i in range(3):
    if  not np.array_equal(axes[i], axe_source):
      axes_du_plan[ind]=axes[i]
      indices_direction_plan[ind]=i
      ind+=1
  

  
  for i in range(3):

    #on fait les calculs
    proportions[i] = dimensions_carton_cible[i]/dimensions_carton_source[i]
  prop_min = np.min(proportions)
  prop_min_rayon = np.min(axes_du_plan@proportions)
  prop_longueur = axe_source@proportions
  dilat= np.eye(3)
  dilat[indices_direction_plan[0], indices_direction_plan[0]] = prop_min_rayon
  dilat[indices_direction_plan[1], indices_direction_plan[1]] = prop_min_rayon
  dilat[indice_direction_axe, indice_direction_axe] = prop_longueur
  centres_source_centres = np.array([centre_source - origine_carton_source for centre_source in centres_source])
  centres_sortie_centres = np.array([dilat@centre_source_centre for centre_source_centre in centres_source_centres])
  centres_sortie = np.array([centre_sortie_centre + origine_carton_cible for centre_sortie_centre in centres_sortie_centres])
  rayon_sortie = rayon_source * prop_min_rayon
  longueur_sortie = longueur_source * prop_longueur
  axe_sortie = np.array(arrangement_source['axe'])
  nombre_sortie = arrangement_source['nombre']
  dimensions_carton  = arrangement_source['dimensions du carton']
  dimensions_carton_sortie = list(dilat@np.array(dimensions_carton))
  origine_carton = arrangement_source['origine du carton']


  #On demande un nom à l'utilisateur
  if interaction :
    nom = input('Comment voulez vous nommer ce nouvel arrangement  ')

    #on crée le dictionnaire de sortie
    arrangement_sortie = arrangement_source.copy()
    arrangement_sortie.update({'nom': nom, 'axe' : arrangement_source['axe'], 'nombre' : arrangement_source['nombre'], 'centres' : centres_sortie , 'dimensions du carton' :   dimensions_carton_sortie, 'origine du carton' : origine_carton_cible, 'rayon': rayon_sortie, 'longueur' : longueur_sortie})

    #on propose de dessiner l'arrangement
    choix = 'feur'
    while choix != 'oui' and choix != 'non' :
      choix = input('voulez vous dessiner l\'arrangement (oui/non)  ')
      if choix == 'oui':
        print('attention, pour continuer vous devez avoir le fichier "outils_dessin.py')
        feur = input('Continuer ?')
        if feur == 'non':
          break
        else :
          from outils_dessin import tracer_arrangement_2d
          tracer_arrangement_2d(arrangement_sortie)
      if choix == 'non':
        print('Non dessiné')


    choix = 'feurfeur'
    while choix != 'oui' and choix != 'non' :
      choix = input('voulez vous l\'ajouter à la base de données ? (oui/non)  ')
      if choix == 'oui':
        ajouter_arrangement(arrangement_sortie, nom_fichier)
        print('arrangement "', nom, '" ajouté à "', nom_fichier,'"' )
      if choix == 'non':
        print('Non ajouté')

  else :
    arrangement_sortie = arrangement_source.copy()
    arrangement_sortie.update( {'axe' : arrangement_source['axe'], 'nombre' : arrangement_source['nombre'], 'centres' : centres_sortie , 'dimensions du carton' :   dimensions_carton_sortie, 'origine du carton' : origine_carton_cible, 'rayon': rayon_sortie, 'longueur' : longueur_sortie})
  return(arrangement_sortie)
      
def etendre_arrangement_2d(arrangement_source=None,  n_Couches=None, longueur_carton_cible = None, nombre_de_couches_initiales = 1, nom_fichier='Base de données des arrangements.json' ):
  
 #on interroge l'utilisateur sur qui il veut etendre sur combien de couches 
  if arrangement_source == None:
    nom_arrg= input('Quel arrangement voulez vous étendre ?  ')
    arrangement_source = ouvrir_arrangements(nom_arrg)

  if n_Couches == None:
    n_Couches = int(input('En combien de couches voulez vous l\'etendre ?  '))

  if longueur_carton_cible == None:
    longueur_carton_cible = float(input('Quelle longueur voulez vous pour le nouveau carton ?  '))

  #on extrait les données de l'arrangement 
  centres_source = np.array(arrangement_source['centres'])
  rayon_source = (arrangement_source['rayon'])
  longueur_source = (arrangement_source['longueur'])
  axe_source = np.array(arrangement_source['axe'])/np.linalg.norm(np.array(arrangement_source['axe']))
  dimensions_carton_source = np.array(arrangement_source['dimensions du carton'])
  origine_carton_source = np.array(arrangement_source['origine du carton'])
  nombre_source = np.shape(centres_source)[0]


  #on construit les différentes couches en translatant les cylindres d'une longueur
  centres_cibles = np.zeros((nombre_source * n_Couches , 3))
  for i in range(n_Couches) :
    for j in range(nombre_source):
      centres_cibles[i*nombre_source + j] = centres_source[j] + i*axe_source*longueur_source*nombre_de_couches_initiales
  
  dimensions_carton_cible = dimensions_carton_source.copy()
  ind_longueur = int(np.dot([0,1,2],axe_source)) #pour trouver l'indice correspondant à l'axe du cylindre
  dimensions_carton_cible[ind_longueur] = longueur_carton_cible

  rayon_cible = rayon_source
  longueur_cible = longueur_source
  axe_cible = axe_source
  nombre_cible = nombre_source * n_Couches
  origine_carton_cible = origine_carton_source

  arrangement_sortie = {
         'axe' : axe_cible,
         'nombre' : nombre_cible,
         'centres' : centres_cibles,
         'rayon' : rayon_cible, 'longueur' : longueur_cible,
         'dimensions du carton' : dimensions_carton_cible,
         'origine du carton' : origine_carton_cible,
         'nombre de couches' : n_Couches
  }
  ajouter_arrangement(arrangement_sortie, nom_fichier)

  return(arrangement_sortie)
  
def tasser(arrangement, interaction = True):

  if arrangement['nombre de couches'] == 1 :
    sur = 'oui'
  if arrangement['nombre de couches'] != 1:
    sur ='feur'
    while sur != 'oui' and sur != 'non':
      sur = input('Êtes vous sûr.e de vouloir le tasser (oui/non) ?  ')
  if sur == 'oui' or interaction == False:
    centres_tasses = np.array(arrangement['centres'])
    axe = arrangement['axe']
    rayon = arrangement['rayon']
    longueur = arrangement['longueur']
    dimensions_carton = arrangement['dimensions du carton']
    origine_carton = arrangement['origine du carton']
    nombre = arrangement['nombre']
    for i in range(len(centres_tasses)) :
      ind_dir_axe = np.dot([0,1,2],axe)
      centres_tasses[i][ind_dir_axe] = origine_carton[ind_dir_axe] + longueur/2

    arrangement['centres'] = centres_tasses
    if interaction :
      ajouter_arrangement(arrangement)

    return(arrangement)

def paver(arrangement_source= None, dimensions_carton_cible= None, nom_fichier='Base de données des arrangements.json'):
  if arrangement_source == None:
    nom_arrg= input('Quel arrangement voulez vous comme pavé ?  ')
    arrangement_source = ouvrir_arrangements(nom_arrg)
  if dimensions_carton_cible == None:
    dimensions_carton_cible = eval(input('Quelles sont les dimensions du carton à paver ?  '))

  proportions = np.zeros(3, np.int64)
  for i in range(3):
    proportions[i] = int(dimensions_carton_cible[i]//arrangement_source['dimensions du carton'][i])
    
  centres_originaux = np.array(arrangement_source['centres'])
  nouveaux_centres = np.zeros((int(arrangement_source['nombre']*proportions[0]*proportions[1]*proportions[2]), 3))
  nombre = np.shape(centres_originaux)[0]
  ind= 0 #pour ne pas se traîner de formules trop dégeues
  for i in range(proportions[0]):
    for j in range(proportions[1]):
      for k in range(proportions[2]):
        for l in range(nombre):
          nouveaux_centres[ind] = centres_originaux[l]+ np.array([i,j,k])*np.array(arrangement_source['dimensions du carton'])
          ind += 1
  nouvel_arrangement = arrangement_source.copy()
  nouvel_arrangement['centres'] = nouveaux_centres
  nouvel_arrangement['dimensions du carton'] = dimensions_carton_cible

  choix='feur'
  while choix != 'oui' and choix != 'non' :
    choix = input('voulez vous l\'ajouter à la base de données ? (oui/non)  ')
    if choix == 'oui':
      ajouter_arrangement(nouvel_arrangement, nom_fichier)
      
  return(nouvel_arrangement)
  
def dessiner_arrangement(nom_arrangement = None, nom_fichier = 'Base de données des arrangements.json'):
  from outils_dessin import tracer_arrangement_2d
  if nom_arrangement == None:
    nom_arrangement = input('Quel arrangement voulez vous dessiner ?  ')
  arrangement = ouvrir_arrangements(nom_arrangement)
  tracer_arrangement_2d(arrangement)

def optimiser_rayon(arrangement_entree):
  """
  Optimise le rayon d'un arrangement en fonction de la distance minimale entre les cylindres.

  Args:
    arrangement: Un dictionnaire représentant l'arrangement.

  Returns:
    L'arrangement avec le rayon optimisé.
  """

  # Vérifier la validité de l'arrangement
  arrangement = arrangement_entree.copy()
  if not est_legitime(arrangement):
    return ('erreur, l\'arragement initial n\'est pas légitime')  # Retourne l'arrangement inchangé si non légitime

  marges = est_legitime.marges





  # Extraire les informations nécessaires de l'arrangement
  nombre = arrangement['nombre']
  centres = arrangement['centres']
  rayon = arrangement['rayon']
  longueur = arrangement['longueur']
  axe = arrangement['axe']
  dimensions_carton = arrangement['dimensions du carton']
  origine_carton = arrangement['origine du carton']

  #on extrait les vecteurs qui portent le plan, ainsi que leurs indices
  axes = np.eye(3)
  axes_du_plan= np.zeros((2,3))
  ind= 0
  indices_direction_plan = np.zeros(2, np.int16)
  indice_direction_axe = np.dot([0,1,2],axe)
  for i in range(3):
    if  not np.array_equal(axes[i], axe):
      axes_du_plan[ind]=axes[i]
      indices_direction_plan[ind]=i
      ind+=1

  distance_min = float('inf')
  # Calculer la distance minimale entre deux cylindres
  for i in range(nombre):
    for j in range(i + 1, nombre):
      distance = distance_deux_cylindres(centres[i][0],centres[i][1],centres[i][2], centres[j][0], centres[j][1], centres[j][2], rayon, longueur, axe)
      distance_min = min(distance_min, distance)
  
  #calcule la marge minimale dans la direction pas de l'axe
  marge_min = float('inf')
  for i in range (2):
    for j in range(2):
      print( marges[indices_direction_plan[i]][j])
      marge_min = min(marge_min, marges[indices_direction_plan[i]][j])
  print('marge minimale', marge_min)
  print('distance minimale', distance_min)

  #on met à jour le rayon dans l'arrangement 
  arrangement['rayon'] = min(distance_min/2, marge_min) + rayon
  print('rayon', arrangement['rayon'])
  # Retourner l'arrangement optimisé
  return arrangement

def optimiser_laize(arrangement_entree):
  arrangement = arrangement_entree.copy()
  longueur =  arrangement['longueur']
  n_couches = arrangement['nombre de couches']
  dimensions_carton = arrangement['dimensions du carton']
  axe = arrangement['axe']
   
  long_opti = np.dot(axe, dimensions_carton)/n_couches
  if long_opti < longueur :
    return('il n\'y a pas moyen de ranger les rouleux en ', n_couches, ' couches avec une longueur minimale de', longueur, ' et une profondeur de carton de ', np.dot(axe, dimensions_carton) )
  
  else : 
    arrangement['longueur'] = long_opti
    return(arrangement)
  