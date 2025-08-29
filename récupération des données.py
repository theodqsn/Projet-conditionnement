# Databricks notebook source
import numpy as np
import pandas as pd
import os
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
!pip install openpyxl
import matplotlib.dates as mdates
from pyspark.sql import functions as F
import matplotlib.pyplot as plt
from datetime import timedelta
from scipy.stats import ks_2samp
from datetime import datetime
!pip install python-Levenshtein
import Levenshtein
from outils_cf import *
import json

# COMMAND ----------

df = pd.read_excel(
   '/Workspace/Users/theo.duquesne@armor-iimak.com/analyse des donnees/incidents_PBI.xlsx',
    skiprows=2
)
df.head()

# COMMAND ----------


casse_film = df[df['Groupe d\'evenement'].apply(lambda x: Levenshtein.distance(x, 'Casse-Film') <= 2)]
casse_film.shape
df.to_excel("historique_casse_film.xlsx", index=False)
casse_film_juin = casse_film[casse_film['Date'].apply(lambda x: x.month == 6 and x.year == 2025)]
print('casse film juin', casse_film_juin.shape )
df.to_excel("historique_casse_film_juin_2025.xlsx", index=False)


# COMMAND ----------

print( casse_film.columns)

# COMMAND ----------

# Pour chaque casse_film, on récupère les données deux heures avant du même OF. On le range dans une classe et on va enregistrer le tout dans un .zip bien rangé


# COMMAND ----------

class CasseFilm:

    def __init__(self, date, of, equipement,lien_casse_film = '/Workspace/Users/theo.duquesne@armor-iimak.com/analyse des donnees/dossier_casse_film' ):
        self.date_reportee = date
        self.date_reelle = None
        self.lien_donnees = None
        self.vrai_casse_film = None
        self.lien_casse_film = lien_casse_film # au bout de ce lien on enregistrera le casse film
        self.donnees = None 
        self.date_debut = date - timedelta(hours=2)
        self.date_fin = date + timedelta(minutes=15)
        self.lien_donnees_propres = None
        self.OF = of
        self.equipement = equipement

    def extraire_donnees(self, lien_vers_xlsx=None, df=None, sql = None, nom_colonnes= None):

      if sql is None :
        # On dépose dans self.donnes la base de données liée au casse film
        if self.lien_donnees_propres is None:
            if lien_vers_xlsx is None and df is None:
                raise Exception('Pas de données fournies')
            elif df is not None:
                self.donnees = df
            else:
                self.donnees = pd.read_excel(lien_vers_xlsx)
        else:
            self.donnees = pd.read_excel(self.lien_donnees_propres)
            self.donnees = self.donnees.sort_values(by='date', ascending=True)
    
      else : 
          if nom_colonnes is None:
              nom_colonnes = spark.table(sql).columns

          self.donnees = extraire_table(nom_colonnes, self.date_debut, self.date_fin, sql)
          if self.donnees is None:
              raise Exception(f'Pas de données fournies : vérifiez bien que des données existent pour cet équipement {self.equipement} sur la periode fournie (entre {self.date_debut} et {self.date_fin})')

    def nettoyer_donnees(self):
        if self.lien_donnees_propres is None:
            self.donnees = self.donnees.dropna(subset=['date'])
            self.donnees = renommer_colonne(self.donnees)
            self.donnees = self.donnees.dropna(subset=['date'])
            self.donnees = self.donnees.sort_values(by='date', ascending=True)
            self.donnees = extraction_date(self.date_debut, self.date_fin, self.donnees)
            
            lien = os.path.join(
                self.lien_casse_film,
                f"{self.OF}_{self.equipement}_{self.date_reportee.strftime('%Y%m%d_%H%M%S')}_donnees_propres.xlsx"
            )
            self.donnees.to_excel(lien, index=False)
            self.lien_donnees_propres = lien

        else : 
            print('Données déjà nettoyées')

    def determiner_date_reelle(self):
        self.date_reelle = detecter_casse_film(self.donnees, date_officielle = self.date_reportee)

    def enregistrer(self):
        """
        Extrait, nettoie et enregistre les données d'un casse-film,
        puis sauvegarde les métadonnées dans un fichier JSON.
        """
        # Création du dossier de sortie s'il n'existe pas
        os.makedirs(self.lien_casse_film, exist_ok=True)

        # Pipeline de traitement
        self.extraire_donnees()
        self.nettoyer_donnees()
        self.determiner_date_reelle()

        # Fichier JSON d'enregistrement
        lien = os.path.join(
            self.lien_casse_film,
            f"{self.OF}_{self.equipement}_{self.date_reportee.strftime('%Y%m%d_%H%M%S')}_infos_cf.json"
        )

        # On choisit ce qu'on sauvegarde (évite de sérialiser le DataFrame entier)
        contenu = {
            "date_reportee": self.date_reportee.strftime("%Y-%m-%d %H:%M:%S") if self.date_reportee else None,
            "date_reelle": self.date_reelle.strftime("%Y-%m-%d %H:%M:%S") if self.date_reelle else None,
            "lien_donnees": self.lien_donnees,
            "lien_casse_film": self.lien_casse_film,
            "lien_donnees_propres": self.lien_donnees_propres,
            "OF": self.OF,
            "equipement": self.equipement,
            "date_debut": self.date_debut.strftime("%Y-%m-%d %H:%M:%S") if self.date_debut else None,
            "date_fin": self.date_fin.strftime("%Y-%m-%d %H:%M:%S") if self.date_fin else None
        }

        # Sauvegarde JSON
        with open(lien, "w", encoding="utf-8") as f:
            json.dump(contenu, f, indent=4, ensure_ascii=False)

        return lien


    def charger(self, lien):
        """
        Charge un fichier JSON et met à jour l'objet CasseFilm avec son contenu.
        """
        if not os.path.exists(lien):
            raise FileNotFoundError(f"Le fichier {lien} est introuvable.")

        with open(lien, "r", encoding="utf-8") as f:
            contenu = json.load(f)

        # Mise à jour des attributs
        self.date_reportee = datetime.strptime(contenu["date_reportee"], "%Y-%m-%d %H:%M:%S") if contenu["date_reportee"] else None
        self.date_reelle = datetime.strptime(contenu["date_reelle"], "%Y-%m-%d %H:%M:%S") if contenu["date_reelle"] else None
        self.lien_donnees = contenu.get("lien_donnees")
        self.lien_casse_film = contenu.get("lien_casse_film", self.lien_casse_film)
        self.lien_donnees_propres = contenu.get("lien_donnees_propres")
        self.OF = contenu.get("OF")
        self.equipement = contenu.get("equipement")
        self.date_debut = datetime.strptime(contenu["date_debut"], "%Y-%m-%d %H:%M:%S") if contenu["date_debut"] else None
        self.date_fin = datetime.strptime(contenu["date_fin"], "%Y-%m-%d %H:%M:%S") if contenu["date_fin"] else None

        return self


    def distribution_variance(self, colonne='Derouleur_DCM42_G2_Mes_Tension', fenetre=10):
        # Extraire la colonne et calculer la variance glissante
        self.nettoyer_donnees()
        var_glissante = self.donnees[colonne].rolling(window=fenetre).var().dropna()
        var_glissante_pos = var_glissante[var_glissante > 0]  # valeurs strictement positives
        log_var = np.log(var_glissante_pos)

        # KDE
        kde = gaussian_kde(log_var)
        x_grid = np.linspace(log_var.min(), log_var.max(), 1000)
        densite = kde(x_grid)

        # Dérivée numérique
        deriv = np.gradient(densite, x_grid)

        # x0 : maximum de densité
        idx_max = np.argmax(densite)
        x0 = x_grid[idx_max]

        # x1 : dernier zéro avant x0
        mask_avant = np.where(x_grid < x0)[0]
        x_1 = None
        if len(mask_avant) > 1:
            signe = np.sign(deriv[mask_avant])
            zero_crossings = np.where(np.diff(signe) != 0)[0]
            if len(zero_crossings) > 0:
                x_1 = x_grid[mask_avant][zero_crossings[-1]]

        # x2 : premier zéro après x0
        mask_apres = np.where(x_grid > x0)[0]
        x_2 = None
        if len(mask_apres) > 1:
            signe = np.sign(deriv[mask_apres])
            zero_crossings = np.where(np.diff(signe) != 0)[0]
            if len(zero_crossings) > 0:
                x_2 = x_grid[mask_apres][zero_crossings[0]]

        print(f"x0 = {x0}, x_1 = {x_1}, x_2 = {x_2}")

        # Tracé
        plt.figure(figsize=(8,5))
        plt.plot(x_grid, densite, label="KDE (log variance)")
        plt.axvline(x0, color='red', linestyle='--', label=f"x0 = {x0:.3f}")
        if x_1 is not None:
            plt.axvline(x_1, color='green', linestyle='--', label=f"x1 = {x_1:.3f}")
        if x_2 is not None:
            plt.axvline(x_2, color='blue', linestyle='--', label=f"x2 = {x_2:.3f}")
        plt.title("Densité (KDE) et points caractéristiques - log variance")
        plt.xlabel("Log(variance glissante)")
        plt.ylabel("Densité")
        plt.legend()
        plt.show()


# COMMAND ----------

liste_casse_film_juin = []
for index, row in casse_film_juin.iterrows():
    casse_film = CasseFilm(row['Date'], row['OF'], row['Equipement'])
    liste_casse_film_juin.append(casse_film)


# COMMAND ----------

print(liste_casse_film_juin)

# COMMAND ----------

print(liste_casse_film_juin[-5].date_debut)
print(liste_casse_film_juin[-5].date_fin)

# COMMAND ----------


import json
sql = "hive_metastore.agg_hist.dcm42_all"
nom_colonnes=['Derouleur_DCM42_G2_Mes_Tension', 'Derouleur_DCM42_G2_Cons_Tension', 'GroupePilote_DCM42_G2_Mes_Vitesse']
for cf in liste_casse_film_juin :
    cf.extraire_donnees(sql = sql, nom_colonnes=nom_colonnes)
    cf.nettoyer_donnees()
    cf.determiner_date_reelle()
    cf.enregistrer()

# COMMAND ----------

for cf in liste_casse_film_juin : 
  cf.nettoyer_donnees()
  plt.figure(figsize=(15,5))
  plt.scatter(cf.donnees['date'], cf.donnees['Derouleur_DCM42_G2_Mes_Tension'])
  plt.xticks(rotation=45)


# COMMAND ----------

if False :
  cf.distribution_variance()

# COMMAND ----------

import numpy as np
from scipy.stats import gaussian_kde

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde, norm

def normalite_variance(df, colonne, fenetre=10, dessiner=False):
    """
    Analyse la distribution de la variance glissante d'une colonne et renvoie
    le centre, la borne inférieure et la borne supérieure selon une estimation
    de type gaussienne.
    """
    
    # ================================
    # Bloc 1 : Variance glissante
    # ================================
    var_glissante = df[colonne].rolling(window=fenetre).var().dropna()
    var_glissante_pos = var_glissante[var_glissante > 0]  # valeurs strictement positives
    log_var = np.log(var_glissante_pos)
    
    # ================================
    # Bloc 2 : KDE sur le log de la variance
    # ================================
    kde = gaussian_kde(log_var)
    x_grid = np.linspace(log_var.min(), log_var.max(), 1000)
    densite = kde(x_grid)
    
    # ================================
    # Bloc 3 : dérivées première et seconde
    # ================================
    deriv1 = np.gradient(densite, x_grid)
    deriv2 = np.gradient(deriv1, x_grid)
    
    # ================================
    # Bloc 4 : maximum de densité
    # ================================
    idx_max = np.argmax(densite)
    x0 = x_grid[idx_max]
    ex0 = np.exp(x0)
    
    # ================================
    # Bloc 5 : demi-hauteur pour sigma
    # ================================
    densite_max = densite[idx_max]
    demi_hauteur = densite_max / 2
    mask_demi = np.where(densite >= demi_hauteur)[0]
    a, b = x_grid[mask_demi[0]], x_grid[mask_demi[-1]]
    sigma = (b - a) / 2.335  # conversion demi-hauteur -> sigma pour gaussienne
    
    # ================================
    # Bloc 6 : quantiles extrêmes
    # ================================
    x_inf_et = x0 + norm.ppf(0.001) * sigma
    x_sup_et = x0 + norm.ppf(0.999) * sigma
    
    # ================================
    # Bloc 7 : zéros de dérivées
    # ================================
    # Zéros avant x0
    mask_avant = np.where(x_grid < x0)[0]
    deriv1_avant = deriv1[mask_avant]
    deriv2_avant = deriv2[mask_avant]
    
    zero_crossings1_avant = np.where(np.diff(np.sign(deriv1_avant)) != 0)[0]
    zero_crossings2_avant = np.where(np.diff(np.sign(deriv2_avant)) != 0)[0]
    
    x_inf_d = x_grid[mask_avant][zero_crossings1_avant[-1]] if len(zero_crossings1_avant) > 0 else None
    x_inf_dd = x_grid[mask_avant][zero_crossings2_avant[-2]] if len(zero_crossings2_avant) > 1 else None
    
    # Zéros après x0
    mask_apres = np.where(x_grid > x0)[0]
    deriv1_apres = deriv1[mask_apres]
    deriv2_apres = deriv2[mask_apres]
    
    zero_crossings1_apres = np.where(np.diff(np.sign(deriv1_apres)) != 0)[0]
    zero_crossings2_apres = np.where(np.diff(np.sign(deriv2_apres)) != 0)[0]
    
    x_sup_d = x_grid[mask_apres][zero_crossings1_apres[0]] if len(zero_crossings1_apres) > 0 else None
    x_sup_dd = x_grid[mask_apres][zero_crossings2_apres[1]] if len(zero_crossings2_apres) > 1 else None


    # ================================
    # Bloc 8 : gestion des absences
    # ================================
    x_inf_der = x_inf_d if x_inf_d is not None else x_inf_dd
    x_sup_der = x_sup_d if x_sup_d is not None else x_sup_dd
    
    # ================================
    # Bloc 9 : bornes finales
    # ================================
    if x_inf_der is None:
        x_inf = x_inf_et
    else:
        x_inf = max(x_inf_et, x_inf_der)
    
    if x_sup_der is None:
        x_sup = x_sup_et
    else:
        x_sup = min(x_sup_et, x_sup_der)
    
    centre, bas, haut = np.exp(x0), np.exp(x_inf), np.exp(x_sup)
    
    # ================================
    # Bloc 10 : tracé si demandé
    # ================================
    if dessiner:
        plt.figure(figsize=(8,5))
        plt.plot(x_grid, densite, label="KDE log-variance")
        if x_inf_et is not None:
            plt.axvline(x_inf_et, color='red', linestyle='--', label='quantile 0.001')
        if x_sup_et is not None :
            plt.axvline(x_sup_et, color='red', linestyle='--', label='quantile 0.999')
        if x_inf_d is not None:
            plt.axvline(x_inf_d, color='purple', linestyle='--', label='dérivée 1')
        if x_sup_d is not None:
            plt.axvline(x_sup_d, color='purple', linestyle='--', label='dérivée 1')
        if x_inf_dd is not None:
            plt.axvline(x_inf_dd, color='green', linestyle='--', label='dérivée 2')
        if x_sup_dd is not None:
            plt.axvline(x_sup_dd, color='green', linestyle='--', label='dérivée 2')
        plt.title("KDE de la log-variance")
        plt.legend()
        plt.show()
    
    return centre, bas, haut


# COMMAND ----------

import pandas as pd
import numpy as np

def classer_points(df, colonne, fenetre=10):

    # Calcul des seuils via normalite_variance
    x0, x1, x2 = normalite_variance(df, colonne, fenetre=fenetre)

    # Variance glissante
    var_glissante = df[colonne].rolling(window=fenetre).var()

    # Classification
    def label_variance(v):
        if pd.isna(v):
            return np.nan
        elif v < x1:
            return 'anormalement bas'
        elif v > x2:
            return 'anormalement haut'
        else:
            return 'normal'

    labels = var_glissante.apply(label_variance)
    return labels


# COMMAND ----------

def detecter_casse_film(df, debut=None, fin=None, date_officielle=None,
                        colonne='Derouleur_DCM42_G2_Mes_Tension',
                        fenetre=10, dessiner=False):

    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    temps_normalite = pd.Timedelta(minutes=3)
    n_anomalies= 3
    

    # Définir bornes temporelles
    if debut is None:
        debut = df['date'].min()
    if fin is None:
        fin = df['date'].max()

    # Extraction de la plage temporelle et tri
    dft = extraction_date(debut, fin, df).sort_values(by='date', ascending=True)

    # Classer les points en variance normale, variance anormale basse, variance anormale haute
    labels = classer_points(dft, colonne, fenetre=fenetre)

    dft = dft.copy()
    dft['classe'] = labels
    if date_officielle is  None:
        date_officielle = fin
    dft.dropna(inplace=True)
    dft.reset_index(inplace=True)

    normal_anormal = []
    tete_exploration  = None
    marqueurs_anomalies = []
    tete_lecture = 0
    while tete_lecture  <  dft.index[-1]-1:
        label = labels.iloc[tete_lecture]
        if label != 'normal':
            normal_anormal.append('anormal')
            tete_lecture += 1
        if label == 'normal':
            #on regarde la prochaine minute avec une tete d'exploration. Si on a déjà une tête d'exploration existante, on s'en sert
              if True :  #tete_exploration is None:
                tete_exploration = tete_lecture
                marqueurs_anomalies = []
              legit = True
              compteur_anomalies =0 
              date_actuelle = dft['date'].iloc[tete_lecture]

              while dft['date'].iloc[tete_exploration] - date_actuelle <temps_normalite and legit and tete_exploration < dft.index[-1]-1 :
                  tete_exploration += 1
                  if labels.iloc[tete_exploration] != 'normal':
                      marqueurs_anomalies.append(tete_exploration)
                      compteur_anomalies += 1
                  if compteur_anomalies >=n_anomalies:
                    legit = False

            # Option 1: en sortie de la boucle, on est legit. On a au moins une minute d'affilé de légitimité
              if legit : 
                tete_balai = tete_exploration
                while  tete_lecture < tete_exploration : 
                    normal_anormal.append('normal')
                    tete_lecture += 1
                  # on va ensuite avancer accumulant les anomalies par l'avant (tête lecture) et en les lachant par l'arrière (tête balai)
                tete_exploration = None
                continuer_normal = True
                while continuer_normal and tete_lecture <  dft.index[-1]-1:
                    
                    
                    if labels.iloc[tete_lecture] != 'normal':
                      compteur_anomalies += 1

                    while dft['date'].iloc[tete_lecture] - dft['date'].iloc[tete_balai] > temps_normalite :
                        if labels.iloc[tete_balai] != 'normal': 
                          compteur_anomalies -= 1
                        tete_balai += 1

                    if compteur_anomalies >= n_anomalies:
                      continuer_normal = False
                      normal_anormal.append('anormal')
                      

                    else:
                      normal_anormal.append('normal')

                    tete_lecture += 1

              # Option 2 : en sortie de la boucle, on est pas legit. On va alors mettre tous ceux avant la première anomalie en anormal puis on va recommencer l'exploration
              else: 
                  marqueur_premiere_anomalie  = marqueurs_anomalies.pop(0)
                  while tete_lecture < marqueur_premiere_anomalie:
                      normal_anormal.append('anormal')
                      tete_lecture += 1
                  compteur_anomalies -= 1 




    # Trouver la dernière zone anormale avant la date officielle
    anormal = False
    dates_anomalies = []
    dates_anomalies_apres = []
    for i, l in enumerate(normal_anormal):
      if dft['date'].iloc[i] < date_officielle:
        if l == 'anormal':
          if not anormal :
            dates_anomalies.append(dft['date'].iloc[i])
            anormal = True
            
        elif l == 'normal':
            anormal = False
      else: 
        if l == 'anormal':
          if not anormal :
            dates_anomalies_apres.append(dft['date'].iloc[i])
            anormal = True
            
        elif l == 'normal':
            anormal = False
    
 
    derniere_anomalies = dates_anomalies[-1]

    # Chercher les gros sauts
    valeurs = dft[colonne].to_numpy()
    der = np.diff(valeurs)
    max_der = np.max(np.abs(der))
    ind_extremes = np.where(np.abs(der) >= max_der / 5)[0]
    gros_sauts_bas = [i for i in ind_extremes if der[i] < 0]

    # Trouver le gros saut juste avant la dernière zone anormale
    print(F"dates anomalies {dates_anomalies}")
    coupure = None
    candidat_coupure_bas = None
    candidat_coupure_haut = None
    for i in gros_sauts_bas:
        if dft['date'].iloc[i] <= derniere_anomalies:
            candidat_coupure_bas = i
        if dft['date'].iloc[i] >= derniere_anomalies and candidat_coupure_haut is None:
            candidat_coupure_haut = i

    
    date_coupure_bas = dft['date'].iloc[candidat_coupure_bas] if candidat_coupure_bas is not None else None
    date_coupure_haut = dft['date'].iloc[candidat_coupure_haut] if candidat_coupure_haut is not None else None
    diff_haut = date_coupure_haut - derniere_anomalies if date_coupure_haut is not None else pd.Timedelta(days =10000 )
    diff_bas = derniere_anomalies - date_coupure_bas if date_coupure_bas is not None else pd.Timedelta(days = 10000 )

    date_coupure = None
    if diff_haut ==  pd.Timedelta(days =10000 ) and pd.Timedelta(days =10000 ):
        print("pas de coupure détectée")
    else :
        if diff_haut < diff_bas:
            coupure = date_coupure_haut
        else:
            coupure = date_coupure_bas

    # Dessin optionnel
    if dessiner:
        couleurs = {
            'anormalement bas': 'blue',
            'normal': 'grey',
            'anormalement haut': 'orange'
        }
        dft['classe'] = dft['classe'].fillna('normal')
        dft['couleur_plot'] = dft['classe'].map(couleurs).fillna('grey')

        plt.figure(figsize=(20, 5))
        plt.scatter(dft['date'], dft[colonne],
                    c=dft['couleur_plot'], s=10, label='Mesures')

        # Barres verticales roses = gros sauts
        for i in ind_extremes:
            plt.axvline(dft['date'].iloc[i], color='pink', linestyle='--', alpha=0.2)

        # Barres vertes = début de zones anormales
        dates = dates_anomalies + dates_anomalies_apres
        for date in dates:
            plt.axvline(date, color='green', linestyle='--', alpha=0.7)
          
        # Création d'une colonne de couleurs selon normal_anormal
        couleurs_etat = {'normal': 'green', 'anormal': 'red'}
        diff = dft.index[-1]- len(normal_anormal) +1
        for i in range(diff) :
            normal_anormal.append( 'normal')
        dft['couleur_etat'] = [couleurs_etat.get(x, 'grey') for x in normal_anormal]

        print(f"dft : {dft}")
        print(f"diff{diff}")
        dft['normal_anormal'] = normal_anormal
        dft['couleur_status'] = dft['normal_anormal'].map({'normal': 'green', 'anormal': 'red'})
        plt.scatter(dft['date'], [150] * len(dft), c=dft['couleur_status'], s=20, label='Normal/Anormal')
        
        plt.title(f"Détection casse-film ({colonne})")
        plt.ylabel(colonne)
        plt.xlabel("Date")
        plt.legend()
        plt.show()
        detecter_casse_film.colonne_choisie = colonne
        detecter_casse_film.dft = dft

    return coupure


# COMMAND ----------

liens = [
  '/Workspace/Users/theo.duquesne@armor-iimak.com/analyse des donnees/dossier_casse_film/4764983_PPDEC179 - DEC179_20250620_163411_donnees_propres.xlsx',
 '/Workspace/Users/theo.duquesne@armor-iimak.com/analyse des donnees/dossier_casse_film/4770931_PPDEC163 - DEC163_20250624_080502_donnees_propres.xlsx',
  '/Workspace/Users/theo.duquesne@armor-iimak.com/analyse des donnees/dossier_casse_film/4768861_PPDEC163 - DEC163_20250617_222516_donnees_propres.xlsx',
  '/Workspace/Users/theo.duquesne@armor-iimak.com/analyse des donnees/dossier_casse_film/4775159_PPDEC209 - DEC209_20250618_201417_donnees_propres.xlsx'
  ]

# COMMAND ----------

for l in liens:
  df = pd.read_excel(l)
  normalite_variance(df, colonne='Derouleur_DCM42_G2_Mes_Tension', dessiner=True)


# COMMAND ----------

for l in liens:
  df = pd.read_excel(l)

  dtae = detecter_casse_film(df, debut=None, fin=None, date_officielle=None,
                          colonne='Derouleur_DCM42_G2_Mes_Tension',
                          fenetre=10, dessiner=True)
  print(f"date du casse-film : {dtae}")

# COMMAND ----------

dft = detecter_casse_film.dft

# COMMAND ----------

print( to_date(heures=14, minutes=50, secondes=0, jour=20, mois=6, annee=2025))

# COMMAND ----------

dft3 = extraction_date( debut=to_date(heures=14, minutes=50, secondes=0, jour=20, mois=6, annee=2025), fin=to_date(heures=15, minutes=50, secondes=0, jour=20, mois=6, annee=2025), data_frame=dft)
plt.figure(figsize=(20, 5))
plt.scatter(dft3['date'], [150] * len(dft3), c=dft3['couleur_status'], s=20, label='Normal/Anormal')
plt.scatter(dft3['date'], dft3['Derouleur_DCM42_G2_Mes_Tension'],
                    c=dft3['couleur_plot'], s=10, label='Mesures')