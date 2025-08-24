import time
import copy

import arrangement_legitimite
import outils_courants
import outils_dessin
import pertes_2d
import perturbations
import calcul_marges
import gestion_arrangements
import penalisation_pertes
import outils_pertes
import final_monos
import minimiseur
import bibos
import minimiseur_bibos
import reglages_bibos
import gestion_nn
import gestion_reglages
import minimiseur_bibos_nouvelle_generation
import pack_unpack
import NN_fonctions_a_evaluer 
import NN_ancienne_version 
import NN_construction_entrainement
import NN_tester_reseau
from collections import deque
from concurrent.futures import ProcessPoolExecutor
import threading



# === Imports étoilés ===
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
from minimiseur_bibos import *
from gestion_nn import *
from gestion_reglages import *
from minimiseur_bibos_nouvelle_generation import *
from pack_unpack import *
from NN_fonctions_a_evaluer import *
from NN_ancienne_version import *
from NN_construction_entrainement import *
from NN_tester_reseau import *


def entrainer_reseau_v2(nom=None, nombre_it= None, sauvegarde = False, conf = False, reinit_reglages = True):
    if sauvegarde :
        sauvegarde_nn()
    if reinit_reglages :
        reglages.dico_reglages ={}

  ### CHARGEMENT/CREATION DU RESEAU
    ancien_ws = os.getcwd()
    chemin_base = path_nn()
    dossier_base = os.path.dirname(chemin_base)
    os.chdir(dossier_base)

   # Chargement base
    with open(chemin_base, 'r') as f:
        base = json.load(f)

   # Choix du nom
    if nom is None:
        noms = [d['Nom'] for d in base]
        print("Réseaux disponibles :")
        for n in noms:
            print("-", n)
        nom = input("Quel réseau voulez-vous entraîner ? ")

   # Recherche dans la base
    idx = next((i for i, d in enumerate(base) if d['Nom'] == nom), None)
    reseau_existant = idx is not None

   # CAS 1 : LE RESEAU EXISTE
    if reseau_existant:
        dico = base[idx]
        try:
            reseau = charger_nn(nom)
        except Exception as e:
            print(f"Erreur lors du chargement du réseau {nom} : {e}")
            os.chdir(ancien_ws)
            return
          
       # On récupère le lien vers les réglages
        lien_reglages = dico['lien reglages']
        lien_param = dico['lien parametres']
        version = len(dico.get('historique', [])) + 1

       # Autosave des anciens paramètres
        if os.path.exists(lien_param):
            timestamp = time.strftime("%Y%m%d-%H%M%S")
            autosave_path = f"{nom}_autosave_v{version}_{timestamp}.pt"
            torch.save(torch.load(lien_param), autosave_path)

            historique = dico.get("historique", [])
            historique.append({
                'version': version,
                'date': timestamp,
                'lien autosave': autosave_path,
                'Entrainement': dico.get('Entrainement', 0),
                'nouveau score': None
            })
            dico['historique'] = historique

   # CAS 2 : LE RESEAU N'EXISTE PAS, ON PROPOSE DE LE CREER
    else:
        print(f"Aucun réseau nommé {nom} trouvé.")
        if not conf:
            choix = input("Voulez-vous en créer un avec les réglages actuels ? (o/n) ")
            if choix.lower() != 'o':
                os.chdir(ancien_ws)
                return

        reseau = creer_reseau(nom=nom)

        with open(chemin_base, 'r') as f:
            base = json.load(f)
        idx = next(i for i, d in enumerate(base) if d['Nom'] == nom)
        dico = base[idx]
        lien_reglages = dico['lien reglages']
        lien_param = dico['lien parametres']

  ### ENTRAÎNEMENT
    
   # INITIALISATION DES OBJETS
    n= reglages_internes('nombre de rouleaux', lien_reglages)
    k= reglages_internes('nombre explorateurs')
    q= reglages_internes('echantillons par entree')
    reg = reglages_internes('reglages minimisation')
    taille_batch = reglages_internes('taille des batchs')
    n_tours = reglages_internes('n_tours')
    valeurs_intermediaires = []
    entrees_test = []
    for i in range(30):
        e = initialisation(0).tolist()
        entrees_test.append(e)

    # Optimiseur
    optimiseur = torch.optim.Adam(reseau.parameters())

    # stratégie
    strategie = Strategie(n,k,taille_batch,reseau)

    #arguments variance
    arguments_variance = ArgumentsVariance()
    arguments_variance.ajouter_argument(nom='variance', argument=torch.ones((taille_batch,k,3*n)))
    arguments_variance.ajouter_argument(nom='iteration', argument=0)
    arguments_variance.ajouter_argument(nom='nombre_total_iterations', argument = n_tours)

    # resultats evaluation
    resultats_evaluation =  ResultatsEvaluation(strategie=strategie, evaluations_par_entree=q, fonction_evaluation= {'version simple' : wrapper_evalue_simple, 'version normale' : wrapper_evalue}[reglages_internes('fonction a evaluer')] , reglages_min = reg)

    # gestion recompense
    gestion_recompense = GestionRecompenses(resultats_evaluation=resultats_evaluation)

   # BOUCLE D'ENTRAINEMENT

    for it in tqdm(range(n_tours), desc="Entraînement"): 

     ### On échantillonne des entrées
      ech_entrees = echantillonner_entrees(n)

     ### On donne les entrées et les arguments pour variance à manger à la stratégie, et on la fait digérer
      strategie = Strategie(n,k,taille_batch,reseau)
      strategie.clean() # Supprime les stratégies individuelles correspondant aux anciennes entrées
      strategie.ajouter_echantillon_entrees(ech_entrees)
      strategie.ajouter_arguments_variance(args_variance=arguments_variance)
      strategie.digerer()

     ### On met à jour resultats_evaluation
      fonction_evaluation = {'version simple' : wrapper_evalue_simple, 'version normale' : wrapper_evalue}[reglages_internes('fonction a evaluer')]
      resultats_evaluation =  ResultatsEvaluation(strategie=strategie, evaluations_par_entree=q, fonction_evaluation=fonction_evaluation , reglages_min = reg)
      resultats_evaluation.clean()
      resultats_evaluation.creer_eval_entrees()
      resultats_evaluation.executer_evaluations()
    
     ### Gestion récompense
      gestion_recompense = GestionRecompenses(resultats_evaluation=resultats_evaluation)
      gestion_recompense.calcul_loss()
    
     ### update du réseau       
      gestion_recompense.loss.backward()
      optimiseur.step()
      optimiseur.zero_grad()

     ### maj arguments_variance
      arguments_variance.ajouter_argument(nom='variance', argument=arguments_variance.variance.clone().detach())
      arguments_variance.ajouter_argument(nom='iteration', argument=arguments_variance.iteration +1)

     ### Sauvegarde des infos
      if sauvegarde :
           nktbqreg=n, k, 30, 1, reg
           tr = TesterReseau(reseau, fonction_evaluation, nktbqreg=nktbqreg)
           e, sortie = tr.evaluer(entrees_test)
           rec = []
           for rue in gestion_recompense.RUEs :
               for recomp in rue.recompenses:
                s = recomp.clone().detach().mean()
                rec.append(s)
           sauvegarde_nn.indicateurs.append({
               'echantillon entrees': ech_entrees.clone(),
               'moyennes': strategie.moyennes.clone().detach(),
               'recompenses' : rec, 
               'echantillon test': {'entrees': e, 'scores': sortie}


           })
        
  ### SAUVEGARDE 
    torch.save(reseau.state_dict(), lien_param)
    dico['lien parametres'] = lien_param
    dico['Date dernier update'] = time.strftime("%Y-%m-%d %H:%M:%S")
    dico['Entrainement'] = dico.get('Entrainement', 0) + n_tours * taille_batch
    dico['Score'] = None

    # Mise à jour base
    base[idx] = dico
    with open(chemin_base, 'w') as f:
        json.dump(base, f, indent=2)

    os.chdir(ancien_ws)
    entrainer_reseau_v2.valeurs_intermediaires = valeurs_intermediaires
    return reseau

def sauvegarde_nn():
    setattr(sauvegarde_nn, 'indicateurs', [] )
