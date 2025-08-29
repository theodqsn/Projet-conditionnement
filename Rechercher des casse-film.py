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

lien_vers_incidents_PBI = '/Workspace/Users/theo.duquesne@armor-iimak.com/analyse des donnees/incidents_PBI.xlsx'
casse_film = df[df['Groupe d\'evenement'].apply(lambda x: Levenshtein.distance(x, 'Casse-Film') <= 2)]
casse_film.shape
df.to_excel("historique_casse_film.xlsx", index=False)
casse_film_juin = casse_film[casse_film['Date'].apply(lambda x: x.month == 6 and x.year == 2025)]
print('casse film juin', casse_film_juin.shape )
df.to_excel("historique_casse_film_juin_2025.xlsx", index=False)
liste_casse_film_juin = []
for index, row in casse_film_juin.iterrows():
    casse_film = CasseFilm(row['Date'], row['OF'], row['Equipement'])
    liste_casse_film_juin.append(casse_film)

import json
sql = "hive_metastore.agg_hist.dcm42_all"
nom_colonnes=['Derouleur_DCM42_G2_Mes_Tension', 'Derouleur_DCM42_G2_Cons_Tension', 'GroupePilote_DCM42_G2_Mes_Vitesse']

for cf in liste_casse_film_juin :
    cf.extraire_donnees(sql = sql, nom_colonnes=nom_colonnes)
    cf.nettoyer_donnees()
    cf.determiner_date_reelle()
    cf.enregistrer()


# COMMAND ----------

import pandas as pd
import ipywidgets as widgets
from IPython.display import display
from datetime import datetime

def interface()
  # 1️⃣ Widget pour sélectionner le fichier Excel
  fichier_widget = widgets.FileUpload(
      accept='.xlsx',
      multiple=False,
      description='Fichier incidents'
  )

  # 2️⃣ Widgets pour la sélection des dates
  date_debut_widget = widgets.DatePicker(
      description='Date début',
      value=datetime(2025, 6, 1)
  )
  date_fin_widget = widgets.DatePicker(
      description='Date fin',
      value=datetime(2025, 6, 30)
  )

  # 3️⃣ Widget pour la table Hive
  table_widget = widgets.Text(
      value='hive_metastore.agg_hist.dcm42_all',
      description='Table Hive',
      layout=widgets.Layout(width='400px')
  )

  # 4️⃣ Widget pour les colonnes (séparées par des virgules)
  colonnes_widget = widgets.Text(
      value='Derouleur_DCM42_G2_Mes_Tension,Derouleur_DCM42_G2_Cons_Tension,GroupePilote_DCM42_G2_Mes_Vitesse',
      description='Colonnes',
      layout=widgets.Layout(width='600px')
  )

  # 5️⃣ Bouton pour lancer le traitement
  bouton_lancer = widgets.Button(
      description="Lancer le traitement"
  )

  # 6️⃣ Fonction qui s'exécute au clic
  def lancer_traitement(b):
      if len(fichier_widget.value) == 0:
          print("Veuillez sélectionner un fichier d'incidents")
          return
      
      # Récupération du fichier uploadé
      uploaded_filename = list(fichier_widget.value.keys())[0]
      content = fichier_widget.value[uploaded_filename]['content']
      df = pd.read_excel(content)

      date_debut = date_debut_widget.value
      date_fin = date_fin_widget.value
      table_hive = table_widget.value
      colonnes = [c.strip() for c in colonnes_widget.value.split(',')]
      
      if date_debut is None or date_fin is None:
          print("Veuillez remplir les deux dates")
          return
      
      # Filtrage par date
      incidents_dates = df[(df['Date'] >= pd.Timestamp(date_debut)) & (df['Date'] <= pd.Timestamp(date_fin))]
      print(f"Nombre d'incidents sélectionnés : {incidents_dates.shape[0]}")
      
      # Création des objets CasseFilm
      liste_incidents = []
      for _, row in incidents_dates.iterrows():
          incident = CasseFilm(row['Date'], row['OF'], row['Equipement'])
          liste_incidents.append(incident)
      
      # Extraction et traitement des données
      for inc in liste_incidents:
          inc.extraire_donnees(sql=table_hive, nom_colonnes=colonnes)
          inc.nettoyer_donnees()
          inc.determiner_date_reelle()
          inc.enregistrer()
      
      print("Traitement terminé.")

  # 7️⃣ Connexion du bouton à la fonction
  bouton_lancer.on_click(lancer_traitement)

  # 8️⃣ Affichage des widgets
  display(fichier_widget, date_debut_widget, date_fin_widget, table_widget, colonnes_widget, bouton_lancer)


# COMMAND ----------

interface()