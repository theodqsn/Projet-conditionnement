import torch  
import torch.nn as nn  
import torch.nn.functional as f 
import torch.optim as optim  
from torch.distributions import Normal 
from tqdm import tqdm  # barre de progression
import matplotlib.pyplot as plt  
import random
import numpy as np 
import torch
import os
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



def entrainer_reseau(nom=None, nombre_it= None):
    sauvegarde_nn()
    reglages.dico_reglages ={}

  #CHARGEMENT/CREATION DU RESEAU
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
                'Entraînement': dico.get('Entraînement', 0),
                'nouveau score': None
            })
            dico['historique'] = historique

   # CAS 2 : LE RESEAU N'EXISTE PAS, ON PROPOSE DE LE CREER
    else:
        print(f"Aucun réseau nommé {nom} trouvé.")
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

  # ==== ENTRAÎNEMENT ====

   # Préparation
    path_reglages = dico['lien reglages']
    if nombre_it is None:
        n_tours = reglages_internes('n_tours', path_reglages)
    else :
        n_tours = nombre_it
    taille_batch = reglages_internes('taille des batchs', path_reglages)
    d_total = 3 * reglages_internes('nombre de rouleaux')
    k = reglages_internes('nombre explorateurs')

   #Initialisation
    optimiseur = torch.optim.Adam(reseau.parameters())
    variances = torch.full((k, d_total), reglages_internes('variance initiale'))
    variance_intra_batch = torch.full((k,),  reglages_internes('variance initiale'))
    proposition_learning_rate = reglages_internes('learning rate maximal')
    entrainer_reseau.reseau_initial = copy.deepcopy(reseau)
    entrainer_reseau.reseau_precedent = copy.deepcopy(reseau)

   #Boucle d'entraînement
    for it in tqdm(range(n_tours), desc="Entraînement"):

      # Test de la stratégie
        echantillons_entrees = echantillonner_entrees(reglages_internes('nombre de rouleaux'))
        strategies, variances = construction_strategie(reseau, echantillons_entrees,variance_intra_batch, variances, it, n_tours)
        recompenses, echantillons_sortie = jouer_parties(strategies, echantillons_entrees, taille_batch, nb_workers=None)
        recompenses_baselines = calculer_recompense(recompenses, echantillons_sortie, strategies)

      # Mise à jour du réseau et des paramètres d'apprentissage
        proposition_learning_rate = gestion_lr(proposition_learning_rate, recompenses)
        variance_intra_batch = calculer_VIB_total(recompenses, variance_intra_batch)
        reseau = update(reseau, strategies, echantillons_sortie, recompenses, optimiseur, proposition_learning_rate)

      # Gestion des indicateurs
        sauvegarde_nn.indicateurs.append({ 'recompense': recompenses, 'indices modes': echantillonner_mixture.indices_modes, 'strategie' : strategies, 'variances': copy.deepcopy(variances), 'vib': variance_intra_batch, 'learning rate propose' : proposition_learning_rate })
        if reglages_internes('calculer deviation strategie'):
            strategies_reseau_actuel, _ = construction_strategie(reseau,echantillons_entrees,variance_intra_batch, variances, it, n_tours)
            strategies_reseau_initial, _ = construction_strategie(entrainer_reseau.reseau_initial,echantillons_entrees,variance_intra_batch, variances, it, n_tours)
            '''_,moyennes_actuelles,_ = strategies_reseau_actuel
            _,moyennes_initiales,_ = strategies_reseau_initial'''
            sauvegarde_nn.indicateurs[-1]['deviation_strategie'] = torch.norm(strategies_reseau_actuel - strategies_reseau_initial)
        if reglages_internes('calculer vitesse strategie' ):
            strategies_reseau_actuel, _ = construction_strategie(reseau,echantillons_entrees,variance_intra_batch, variances, it, n_tours)
            strategies_reseau_precedent, _ = construction_strategie(entrainer_reseau.reseau_precedent,echantillons_entrees,variance_intra_batch, variances, it, n_tours)
            '''_,moyennes_actuelles,_ = strategies_reseau_actuel
            _,moyennes_precedentes,_ = strategies_reseau_precedent'''
            sauvegarde_nn.indicateurs[-1]['vitesse_strategie'] = torch.norm(strategies_reseau_actuel - strategies_reseau_precedent) 

  # ==== SAUVEGARDE ====
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
    return reseau

## Définition du réseau de neurones pour paramétrer le mélange de gaussiennes
class ReseauMelangeGaussiennes(nn.Module):
    def __init__(self, nombre_de_rouleaux, nombre_de_couches, taille_sortie, taille_intermediaire= 64 ):
        ## Initialisation du module parent
        super().__init__()  # constructeur de nn.Module
        
        ## Construction des couches cachées
        self.couches_cachees = nn.Sequential(
            nn.Linear(3, taille_intermediaire),  # entrée  : dimensions + rayon mandrin
            nn.ReLU(),  # activation ReLU
            *[layer for _ in range(nombre_de_couches) for layer in (nn.Linear(taille_intermediaire, taille_intermediaire), nn.ReLU())]  # 4 blocs (Linear+ReLU)
        )
        
        ## Définition de la couche de sortie
        self.pre_couche_sortie = nn.Linear(taille_intermediaire, taille_sortie) 
        self.couche_sortie = nn.Sigmoid()
        #Sortie : moyennes et probas de tirer pour la mixture gaussienne 

    def forward(self, x):
        ## Passage dans les couches cachées
        x = self.couches_cachees(x) 

        ## Calcul de la sortie brute
        x  = self.pre_couche_sortie(x)
        return self.couche_sortie(x) 
      
def indicateurs(liste):
    if not hasattr(sauvegarde_nn, 'indicateurs'):
        raise Exception("pas d'indicateurs sauvegardés")

    liste_a_plot = []
    noms_a_plot = []
    n = len(sauvegarde_nn.indicateurs)

    def deviation_strategie():
        res = np.array([np.array([sauvegarde_nn.indicateurs[i]['deviation_strategie'].detach().numpy() for i in range(n)])])
        liste_a_plot.append(res)
        noms_a_plot.append("Déviation des stratégies")

    def vitesse_strategie():
        k = reglages_internes('nombre explorateurs')
        res = np.zeros((k, n))
        res = np.log(np.array([np.array([sauvegarde_nn.indicateurs[i]['vitesse_strategie'].detach().numpy() for i in range(n)])]))
        liste_a_plot.append(res)
        noms_a_plot.append("Log-Vitesse des stratégies")
        
    def recompenses():
        rec =np.array([(np.mean(np.array([(sauvegarde_nn.indicateurs[i]['recompense'].detach().numpy().flatten()) for i in range(len(sauvegarde_nn.indicateurs))]), axis=1))])
        liste_a_plot.append(rec)
        noms_a_plot.append("Évolution des récompenses")

    def var_tempo_recompenses():
        n = len(sauvegarde_nn.indicateurs)
        r =np.array([(sauvegarde_nn.indicateurs[i]['recompense'].detach().numpy().flatten()) for i in range(len(sauvegarde_nn.indicateurs))])
        r = np.mean(r, axis = 1)

        mem  = reglages_internes('memoire variation recompenses')
        rec = np.zeros((1,n))
        for i in range(n):
            if i >= mem :
                rec[0][i] = np.mean(np.abs(np.diff(r[i-mem:i])))
        liste_a_plot.append(rec)
        noms_a_plot.append("Variation temporelle des récompenses")

    def log_vib():
        rec=np.log(np.array([(np.array([(sauvegarde_nn.indicateurs[i]['vib']) for i in range(len(sauvegarde_nn.indicateurs))]))]))
        liste_a_plot.append(rec)
        noms_a_plot.append("Évolution de la log-variance intra-batch des récompenses")

    def var_exploration():
        k = len(sauvegarde_nn.indicateurs[0]['variances'].detach().numpy())
        n = len(sauvegarde_nn.indicateurs)
        rec = np.zeros((k, n))
        for j in range(k):
            for i in range(n):
                rec[j][i] = sauvegarde_nn.indicateurs[i]['variances'][j].clone().detach().numpy().mean()
        rec = rec[:,1:]
        liste_a_plot.append(rec)
        noms_a_plot.append("Évolution de la variance d\'exploration")

    def learning_rate():
        k = 1
        n = len(sauvegarde_nn.indicateurs)
        rec = np.zeros((k, n))
        for j in range(k):
            for i in range(n):
                rec[j][i] = sauvegarde_nn.indicateurs[i]['learning rate propose']
        liste_a_plot.append(rec)
        noms_a_plot.append("Évolution du learning rate proposé")

  # Dictionnaire des fonctions disponibles
    dict_fonction = {
        'deviation_strategie': deviation_strategie,
        'vitesse_strategie': vitesse_strategie,
        'recompenses': recompenses,
        'variance_temporelle' : var_tempo_recompenses,
        'learning rate': learning_rate,
        'variance_exploration' : var_exploration,
        'vib' : log_vib
        
        
    }
    if liste == 'Tout':
        liste = list(dict_fonction.keys())
        if not reglages_internes('calculer deviation strategie'):
            liste.remove('deviation_strategie')
        if not reglages_internes('calculer vitesse strategie'):
            liste.remove('vitesse_strategie')

    for nom in liste:
        if nom not in dict_fonction:
            raise ValueError(f"Indicateur inconnu : {nom}")
        dict_fonction[nom]()  # exécute la fonction correspondante

  # Tracer joliment tous les éléments de liste_a_plot
    for i, data in enumerate(liste_a_plot):
        plt.figure(figsize=(10, 4))
        for j in range(data.shape[0]):
            plt.plot(data[j], label=f"dim {j}", c='orange')
        plt.title(noms_a_plot[i])
        plt.xlabel("Itérations")
        plt.ylabel("Valeur")
        plt.legend()
        plt.grid(True)
        plt.show()

def initialisation(nombre_de_rouleaux):
    # Déterminer les bornes admissibles pour x
    borne_inf = 0.5   # x >= 0.5 car 2 - x <= 3x ⇒ 2 <= 4x
    borne_sup = 1.5   # x <= 1.5 car 1/3 x <= 2 - x ⇒ 4x <= 6

    # Tirage de x, puis calcul de y = 2 - x
    x = random.uniform(borne_inf, borne_sup)
    y = 2 - x

    # Calcul du rayon maximal autorisé
    rayon_max = min(x, y)/3 

    # Tirage du petit rayon
    petit_rayon = random.uniform(0, rayon_max)

    # Assemblage sous forme de tenseur torch
    return torch.tensor([x, y, petit_rayon], dtype=torch.float32)

def echantillonner_entrees(nombre_de_rouleaux): 
  t_b = reglages_internes('taille des batchs')
  return( torch.stack([initialisation(nombre_de_rouleaux) for _ in range(t_b)]))

def construction_strategie(reseau, entrees,variance_intra_batch, variance, iteration, n_it):
  k = reglages_internes('nombre explorateurs')
  strats = []
  for j in range(len(entrees)):
    probas, moyennes = decode_nn(reseau(entrees[j])) # (k,) , (k,3d)
    variances = gestion_variance(variance_intra_batch, variance, iteration, n_it) #  (k,3d)
    probas_f = torch.flatten(probas)
    moyennes_f = torch.flatten(moyennes)
    variances_f = torch.flatten(variances)
    strategie = torch.concatenate([probas_f, moyennes_f, variances_f])
    strats.append(strategie)
  strategies = torch.stack(strats)
  return(strategies, variances)


"""
def jouer_parties(strategies, entrees, taille_batch, nb_workers=None):
    k = reglages_internes('nombre explorateurs')
    reg = reglages_internes('reglages minimisation')
    troisd = 3*reglages_internes('nombre de rouleaux')
    echant_par_entree = reglages_internes('echantillons par entree')
    fonction_evaluation = {'version simple' : wrapper_evalue_simple, 'version normale' : wrapper_evalue}[reglages_internes('fonction a evaluer')]
    reglages_min = copy.deepcopy(reg)
    echantillons_sorties = echantillonner_sorties(strategies, echant_par_entree,k, troisd)

    # Raffinage de l'échantillonnage et conditionnement pour la parallélisation
    args = []
    for i,strategie in enumerate(strategies) : 
        x, y, petit_rayon = entrees[i]
        petit_rayon = float(petit_rayon.numpy())
        dims = (x.item(), y.item())
        echantillon_sortie = echantillons_sorties[i]
        echantillon_raffine = recentrer(echantillon_sortie, dims, petit_rayon)

        # Préparer les arguments pour chaque évaluation
        args_i = [(elem.detach(), dims, petit_rayon, k, reglages_min) for elem in echantillon_raffine]
        args += args_i

    # Parallélisation des évaluations
    reglages.dico_reglages['iterations batch'] = 0
    with ProcessPoolExecutor(max_workers=nb_workers) as executor:
        recompenses = list(executor.map(fonction_evaluation, args))

    recompenses_tensor = torch.tensor(recompenses, dtype=torch.float32)
    recompenses_tensor = torch.stack(list(torch.split(recompenses_tensor, echant_par_entree) ))

    return recompenses_tensor, echantillons_sortie
    """

def echantillonner_sorties(strategies, echant_par_entree, k,troisd):
  sorties = []
  indices_modes = []
  for i,strat in enumerate(strategies) : 
        p,m,v = torch.split(strat, [k,k*troisd,k*troisd], dim=0)
        probas= torch.split(p,k)[0] # k= 2
        moyennes = torch.stack(list(torch.split(m, troisd ))) #3d= 3
        variances= torch.stack(list(torch.split(v, troisd)))
        strategie = (probas, moyennes, variances)
        sorties_i = echantillonner_mixture(strategie, echant_par_entree) # Pour chaque entrée, on echantillonne qques sorties
        sorties.append(sorties_i)
        indices_modes.append(echantillonner_mixture.indices_modes)
  echantillonner_sorties.indices_modes = torch.stack(indices_modes)
  return torch.stack(sorties)

def echantillonner_mixture(strategie, n_echant):
  poids, moyennes, variances = strategie  # tailles : (k,), (k, 3d), (k, 3d)
  k, d_total = moyennes.shape              # d_total = 3 * d

  # Tirage des indices de mode selon la loi catégorielle
  indices_modes = torch.multinomial(poids, n_echant, replacement=True)

  # Sélection des paramètres du mode pour chaque échantillon
  moyennes_sel = moyennes[indices_modes]      # (n_echant, d_total)
  variances_sel = variances[indices_modes]    # (n_echant, d_total)

  # Tirage dans chaque dimension selon la N(m, sigma)
  bruit = torch.randn(n_echant, d_total)
  echantillon = moyennes_sel + bruit * variances_sel.sqrt()

  ''' # Pour les dimensions [2d:] (i.e. les angles), on réduit modulo 2pi . Pas malin, ça va faire mal au calcul de logprob
  d = d_total // 3
  echantillon[:, 2*d:] = echantillon[:, 2*d:] % (2 * torch.pi)'''
  echantillonner_mixture.indices_modes = indices_modes
  return echantillon

def calculer_recompense(recompenses, echantillons, strategies):
    portee = reglages_internes('portee explorateurs')

     #ajout d'une baseline (une par état)
    rec_baselines = torch.zeros_like(recompenses)
    for i,rec_etat in enumerate(recompenses) : 
        baseline = torch.mean(rec_etat)
        rec_baselines[i] = rec_etat - baseline
    return rec_baselines  # (B,)

def update(reseau, strategies, echantillons, recompenses, optimiseur, learning_rate):
    """
    Met à jour les paramètres du réseau selon REINFORCE.

    Args:
        reseau: réseau de neurones (nn.Module)
        strategie: tuple (poids, moyennes, variances)
        echantillon: (B, d) échantillons tirés de la stratégie
        recompense: (B,) récompenses individuelles (torch.tensor)
        optimiseur: instance torch.optim (ex: Adam)
        learning_rate: float, taux d’apprentissage déjà adapté

    Returns:
        reseau mis à jour
    """
    if reglages_internes('Interventionnisme learning rate'):
        # Met à jour le learning rate de l’optimiseur
        for group in optimiseur.param_groups:
            group['lr'] = learning_rate

    optimiseur.zero_grad()
    k =  reglages_internes('nombre explorateurs')
    troisd = 3* reglages_internes('nombre de rouleaux')
    liste_loss = []
    for i, echantillon in enumerate(echantillons):
        recompense = recompenses[i] #déjà baselinée
        strat = strategies[i]
        p,m,v = torch.split(strat, [k,k*troisd,k*troisd], dim=0)
        probas= torch.split(p,k)[0] # k= 2
        moyennes = torch.stack(list(torch.split(m, troisd ))) 
        variances= torch.stack(list(torch.split(v, troisd)))
        strategie = (probas, moyennes, variances)
        log_probs = calcul_log_prob(strategie, echantillon)  # (B,)
        liste_loss.append( - (recompense * log_probs).mean() ) 

    loss= torch.stack(liste_loss).mean()
    loss.backward()
    entrainer_reseau.reseau_precedent = copy.deepcopy(reseau)
    optimiseur.step()

    return reseau

def calcul_log_prob(strategie, echantillon):
    """
    Calcule les log-probabilités d’un batch d’échantillons sous une mixture gaussienne.

    Args:
        strategie: tuple (poids, moyennes, variances)
            - poids: (k,)
            - moyennes: (k, d)
            - variances: (k, d)
        echantillon: (B, d), batch de B vecteurs à évaluer

    Returns:
        log_probs: (B,), log densité de chaque échantillon sous la mixture
    """
    poids, moyennes, variances = strategie  # (k,), (k, d), (k, d)
    k = reglages_internes('nombre explorateurs')
    n = reglages_internes('nombre de rouleaux')
    d = 3*n

    # (B, 1, d) - (1, k, d) => (B, k, d)
    diff = echantillon.unsqueeze(1) - moyennes.unsqueeze(0)  # (B, k, d)
    exponent = -0.5 * (diff**2 / variances.unsqueeze(0)).sum(dim=2)  # (B, k)
    log_det = 0.5 * torch.log(2 * torch.pi * variances).sum(dim=1)  # (k,)
    
    # log(p_i * N_i(x)) = log(p_i) - log_det + exponent
    log_component = exponent - log_det.unsqueeze(0)  # (B, k)
    log_probs = torch.log(poids + 1e-8).unsqueeze(0) + log_component  # (B, k)

    # Log-sum-exp over components
    log_mix = torch.logsumexp(log_probs, dim=1)  # (B,)

    return log_mix

def interpretation(sortie_reseau_brute, dims,n_explo):
    centres_orientations_bruts = sortie_reseau_brute.detach().cpu().numpy()
    k= n_explo
    n = len(centres_orientations_bruts) // k
    n= (n-1) // 3
    x,y = dims
    for i in range(n):
        centres_orientations_bruts[2*i] *= x
        centres_orientations_bruts[2*i+1] *= y
        centres_orientations_bruts[2*n+i] *= 2*np.pi
    return centres_orientations_bruts

def gestion_variance(args):

  # On récupère les arguments necessaires à la gestion de la variance
    if hasattr(args, 'variance_intra_batch'):
        variance_intra_batch = args.variance_intra_batch
    if hasattr(args, 'variance'):
        variance = args.variance
    if hasattr(args, 'iteration'):
        iteration = args.iteration
    if hasattr(args, 'nombre_total_iterations'):
        n_it = args.nombre_total_iterations
    
    variances_actuelles = variance 

  # 1ère manière de modifier la variance : contrôle de la variance intra-batch (VIB)
    if reglages_internes('Controlabilite VIB'):
        for ind_mode in range(reglages_internes('nombre explorateurs')):
            vib = variance_intra_batch[ind_mode]
            alpha = reglages_internes('reactivite prudence')
            seuil = reglages_internes('seuil prudence')
            coeff = np.clip((seuil/vib)**alpha, 0, reglages_internes('max acceleration variance exploration'))
            t = iteration / n_it
            for i in range(len(variances_actuelles[ind_mode])):
                variances_actuelles[ind_mode][i] = torch.clamp(coeff * variances_actuelles[ind_mode][i], min=(1-t)*reglages_internes('min variance exploration autorisee'), max= reglages_internes('max variance exploration autorisee'))

  # Manière par défaut : On fait de l'annealing lineaire 
    else :
        t = iteration / n_it
        for a in range(len(variances_actuelles)):
            for ind_mode in range(reglages_internes('nombre explorateurs')):
                for i in range(len(variances_actuelles[a][ind_mode])):
                    variances_actuelles[a][ind_mode][i] = (1-t)*reglages_internes('variance annealing debut') + t*reglages_internes('variance annealing fin')

    return variances_actuelles

def decode_nn(sortie):
    n = reglages_internes('nombre de rouleaux')
    k = reglages_internes('nombre explorateurs')
    moyennes_1d, probas_1d = torch.split(sortie, [3 * k * n, k])
    moyennes = moyennes_1d.reshape(k, 3 * n)

    # Softmax classique
    probas = torch.softmax(probas_1d, dim=0)

    # Clipping bas (valeurs minimales)
    seuil = reglages_internes('seuil exploration minimal')/k
    min_total = seuil * k
    if min_total <= 1:
        probas = torch.clamp(probas, min=seuil)
        exces = probas.sum() - 1.0
        if exces > 0:
            # Réduction proportionnelle sur les éléments > seuil
            masque = probas > seuil
            surplus = probas[masque] - seuil
            somme_surplus = surplus.sum()
            if somme_surplus > 0:
                reduction = exces * (surplus / somme_surplus)
                probas[masque] -= reduction
            else:
                # Cas limite : tous les probas sont en dessous de seuil
                probas = torch.full_like(probas, 1.0 / k)

    return (probas, moyennes)

def recentrer_une_entree(c_o_0, dims, r, k):
    x, y = dims
    c_o = c_o_0.clone()
    n = len(c_o)//3

    for j in range(n):
            idx_x = 2 * j
            idx_y = 2 * j + 1
            idx_theta = 2*n+j

            t_x = c_o[idx_x].item() / x
            t_y = c_o[idx_y].item() / y

            val_x = torch.sigmoid(torch.tensor(4*t_x-2)) * x
            val_y = torch.sigmoid(torch.tensor(4*t_y-2)) * y

            c_o[ idx_x] = val_x
            c_o [idx_y] = val_y
            c_o[idx_theta] = c_o[idx_theta]% (2*pi)
    return(c_o)

def recentrer(centres, dims,r):
    x, y = dims
    k = reglages_internes('nombre explorateurs')
    n = reglages_internes('nombre de rouleaux')
    T = reglages_internes('echantillons par entree')

    def g(t):
        return 4*t -2

    for i in range(T):
        for j in range(n):
            idx_x = 2 * j
            idx_y = 2 * j + 1
            idx_theta = 2*n+j

            t_x = centres[i, idx_x].item() / x
            t_y = centres[i, idx_y].item() / y

            val_x = torch.sigmoid(torch.tensor(g(t_x))) * x
            val_y = torch.sigmoid(torch.tensor(g(t_y))) * y

            centres[i, idx_x] = val_x
            centres[i, idx_y] = val_y
            centres[i, idx_theta] = centres[i, idx_theta]% (2*pi)

    return centres

def charger_nn(nom=None):
    # Enregistrement du workspace courant
    workspace_actuel = os.getcwd()

    try:
        # Accès à la base de données
        chemin_base = path_nn()
        dossier_base = os.path.dirname(chemin_base)
        os.chdir(dossier_base)

        # Chargement de la base de données
        with open(chemin_base, 'r') as f:
            base = json.load(f)

        # Si nom est None, afficher les réseaux disponibles
        if nom is None:
            print("Réseaux disponibles :")
            for r in base:
                print(f"• {r.get('Nom', '(sans nom)')}")
            nom = input("Quel réseau voulez-vous modifier ?  ").strip()

        # Recherche du réseau
        reseau_dict = next((r for r in base if r.get("Nom") == nom), None)
        if reseau_dict is None:
            raise ValueError(f"Réseau nommé '{nom}' introuvable dans la base de données.")

        # Récupération des paramètres
        n_rouleaux = reseau_dict["Nombre de rouleaux"]
        n_couches = reseau_dict["Nombre de couches"]
        taille_inter = reseau_dict["Taille des couches internes"]
        lien_param = reseau_dict["lien parametres"]
        k = reseau_dict["Nombre d\'explorateurs"]
        lien_reglages = reseau_dict["lien reglages"]
        charger_nn.lien_reglages = lien_reglages

        # Calcul de la taille de sortie 
        taille_sortie = (3*n_rouleaux + 1)*k

        # Création du réseau
        modele = ReseauMelangeGaussiennes(
            nombre_de_rouleaux=n_rouleaux,
            nombre_de_couches=n_couches,
            taille_sortie=taille_sortie,
            taille_intermediaire=taille_inter
        )

        # Chargement des paramètres
        modele.load_state_dict(torch.load(lien_param, map_location=torch.device('cpu')))
        modele.eval()

        return modele

    finally:
        # Retour dans le workspace initial
        os.chdir(workspace_actuel)

def creer_reseau(nombre_de_rouleaux=None, nombre_de_couches=None, taille_sortie=None, taille_couches_internes=None, nombre_explorateurs=None, path=None, nom=None):
    import gestion_reglages
    path_reglages_or = gestion_reglages.path_vers_reglages_bibos + '/reglages_bibos.py'
    # Récupération des paramètres depuis les réglages si nécessaires
    if nombre_de_rouleaux is None:
        nombre_de_rouleaux = reglages("nombre de rouleaux")
    if nombre_de_couches is None:
        nombre_de_couches = reglages("nombre de couches")
    if taille_couches_internes is None:
        taille_couches_internes = reglages("taille couches intermediaires")
    if nombre_explorateurs is None:
        nombre_explorateurs = reglages("nombre explorateurs")
    if path is None:
        path = reglages("path parametres defaut")

    # Détermination du nom
    if nom is None:
        nom = input("Quel nom voulez vous donner au réseau ? \n  ")

    # Construction du chemin complet avec nom
    path_complet = f"{path}_{nom}_parametres"
    if not path_complet.endswith('.pt'):
        path_complet += '.pt'

    path_reglages = f"{path}_{nom}_reglages"
    if not path_reglages.endswith('.py'):
      path_reglages += '.py'

    if not os.path.exists(path_reglages_or):
        raise FileNotFoundError(f"Fichier de réglages introuvable : {chemin_source_reglages}")
    shutil.copy(path_reglages_or, path_reglages)

    # Création du répertoire cible si nécessaire
    dossier = os.path.dirname(path_complet)
    if dossier != '' and not os.path.exists(dossier):
        os.makedirs(dossier)

    # Refus d’écraser un fichier existant
    if os.path.exists(path_complet):
        raise FileExistsError(f"Le fichier '{path_complet}' existe déjà. Choisis un autre nom ou supprime le fichier.")

    # Calcul de la taille de sortie
    taille_sortie = (3 * nombre_de_rouleaux + 1) * nombre_explorateurs

    # Création du réseau
    reseau = ReseauMelangeGaussiennes(
        nombre_de_rouleaux=nombre_de_rouleaux,
        nombre_de_couches=nombre_de_couches,
        taille_sortie=taille_sortie,
        taille_intermediaire=taille_couches_internes
    )

    # Sauvegarde des paramètres
    torch.save(reseau.state_dict(), path_complet)

    # Accès à la base de données
    workspace_actuel = os.getcwd()
    try:
        chemin_base = path_nn()
        dossier_base = os.path.dirname(chemin_base)
        os.chdir(dossier_base)

        # Chargement de la base de données
        with open(chemin_base, 'r') as f:
            base = json.load(f)

        # Vérification de l'unicité du nom
        if any(r.get("Nom") == nom for r in base):
            raise ValueError(f"Un réseau nommé '{nom}' existe déjà dans la base de données.")

        # Création de l'entrée
        entree = {
            "Nom": nom,
            "Nombre de rouleaux": nombre_de_rouleaux,
            "Score": None,
            "Entrainement": 0,
            "Taille des couches internes": taille_couches_internes,
            "Nombre de couches": nombre_de_couches,
            "Nombre d'explorateurs": nombre_explorateurs,
            "Date creation": datetime.now().strftime("%d-%m-%Y, %H:%M:%S"),
            "historique modification": [],
            "lien parametres": path_complet,
            "lien reglages" :  path_reglages,
            "Commentaires": []
        }

        base.append(entree)

        # Sauvegarde de la base
        with open(chemin_base, 'w') as f:
            json.dump(base, f, indent=2)

    finally:
        # Retour dans le workspace initial
        os.chdir(workspace_actuel)

    return reseau

def calcul_vib(recompenses, ancien_vib, indices):
    vibs = []
    rec = recompenses.clone().detach()
    rec = rec.cpu().numpy()
    indices = indices.detach().cpu().numpy()
    k = reglages_internes('nombre explorateurs')
    vib  = torch.zeros(k)
    for i in range(k):
        rec_i = rec[indices == i]
        if len(rec_i) > 1:
            vib[i] =float( np.var(rec_i, axis=0))
        else :
            vib[i] = ancien_vib[i] # Pas forcement optim
        vibs.append(vib)

    return  vibs 

def calculer_VIB_total(recompenses, ancien_vib):
    '''
    Sortie:  vib ; tensor de taille k. Chaque élément est la variance des récompense du ième explorateur, moyenné sur les différentes entrées
    '''
    tableau_vibs = []
    for ind, recompenses_de_l_entree in enumerate(recompenses) :
        indices = echantillonner_sorties.indices_modes[ind]
        vibs_de_l_entree = calcul_vib(recompenses_de_l_entree, ancien_vib, indices) #liste de floats
        tableau_vibs.append(vibs_de_l_entree)

    tableau_vibs = np.array(tableau_vibs)
    nouveau_vib = np.mean(tableau_vibs,axis=0)
    return nouveau_vib
        
def gestion_lr(lr,recompense):
    rec = recompense.clone().detach().cpu().numpy()
    rec_moy = rec.mean()
    

    #gestion de l'historique des recompenses récentes
    if not hasattr(gestion_lr, 'histo_rec'):
        gestion_lr.histo_rec = []
    if len(gestion_lr.histo_rec) > reglages_internes('memoire variation recompenses'):
        gestion_lr.histo_rec.pop(0)
    gestion_lr.histo_rec.append(rec_moy)

    # Adaptation du lr par commandabilité foireuse
    if len(gestion_lr.histo_rec) >1:
        var = np.abs(np.diff(gestion_lr.histo_rec)).mean()
        v_seuil = reglages_internes('variance recompense acceptee')
        coeff = np.clip(v_seuil/var, 0, a_max = reglages_internes('acceleration max learning rate') )**0.5
        lr  = np.clip(lr * coeff, reglages_internes('learning rate minimal'), reglages_internes('learning rate maximal'))
    return(lr)

def arrangemement_from_reseau(nom, dimensions, rayon_mandrin):
  reseau = charger_nn(nom)
  x,y = dimensions
  X = torch.tensor([x,y,rayon_mandrin])

  probas, moyennes = decode_nn(reseau(X))
  print(moyennes)
  probas = probas.detach().numpy()
  moyennes = moyennes.detach().numpy()
  i = np.argmax(probas)
  m = moyennes[i]
  centres_orientations = m
  centres, orientations = t_2_d_bibos(injection_bibos(centres_orientations, np.array([0,0,1]), 0.5))
  dims = np.array([x,y,1])
  arrangement = {
    'nom': 'gouloug',
    'dimensions du carton': dims,
    'petit rayon': rayon_mandrin,
    'centres': centres,
    'orientations': orientations,
    'grand rayon' : 0,
    'axe': np.array([0,0,1]),
    'longueur' : 1

  }
  arrangement = inch_allah_bibos(arrangement)
  return arrangement

def log_prob_mixture_gaussienne(X, pi, mu, sigma2):
    """
    X : (d,)          # vecteur
    pi : (k,)         # probabilités de mélange (sommant à 1)
    mu : (k, d)       # moyennes
    sigma2 : (k, d)   # variances diagonales
    """
    k, d = mu.shape

    # (k, d) - (d,) => (k, d)
    diff = X.unsqueeze(0) - mu  # broadcasting

    # log densité de la gaussienne diagonale pour chaque composante
    log_det = torch.sum(torch.log(sigma2), dim=1)  # (k,)
    mahalanobis = torch.sum(diff**2 / sigma2, dim=1)  # (k,)

    log_prob_components = -0.5 * (log_det + mahalanobis + d * torch.log(torch.tensor(2 * torch.pi)))  # (k,)

    # ajout des log(pi)
    log_pi = torch.log(pi + 1e-12)  # on évite log(0)
    log_weighted = log_pi + log_prob_components  # (k,)

    # somme des densités pondérées (en log), via log-sum-exp
    log_prob = torch.logsumexp(log_weighted, dim=0)  # scalaire

    return log_prob