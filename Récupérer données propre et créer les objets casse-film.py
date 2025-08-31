# Databricks notebook source
# MAGIC %md
# MAGIC Ce notebook sert, à partir d'une extraction des incidents powerBI ainsi que d'une base de donnée des capteurs (en excel ou esql via databricks), de créer des objets "CasseFilm" qui contiennent l'information suur chacun des casses-film. Ils sont alors enregistrés au bout du lien fourni, sous la forme OF_equipement_date_reportee_infos_cf.json, ainsi qu'un fichier excel qui est une extraction de la base de données des capteurs correspondant aux enregidtrement 2h avant et 15 minutes après la déclaration de chaque casse-film.

# COMMAND ----------

# MAGIC %md
# MAGIC ## Import des modules

# COMMAND ----------

!pip install python-Levenshtein
!pip install openpyxl


# COMMAND ----------

import numpy as np
import pandas as pd
import os
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import matplotlib.dates as mdates
from pyspark.sql import functions as F
import matplotlib.pyplot as plt
from datetime import timedelta
from scipy.stats import ks_2samp
from datetime import datetime
import Levenshtein
from outils_cf import *
from classe_casse-film import *
import json

# COMMAND ----------

# MAGIC %md
# MAGIC ## Import de la base de données

# COMMAND ----------

# Pour pouvoir cliquer sur Run All sans exécuter des choses non prévues
raise Exception('Stop')

# COMMAND ----------

# MAGIC %md
# MAGIC #### Aller chercher les données des capteurs
# MAGIC Remplir les champs puis exécuter la cellule

# COMMAND ----------

aller_chercher_donnees_sql = True

if aller_chercher_donnees_sql :
    print("vérifiez que les champs sont corrects")
    sql = "hive_metastore.agg_hist.dcm42_all"
    nom_colonnes=['Derouleur_DCM42_G2_Mes_Tension', 'Derouleur_DCM42_G2_Cons_Tension', 'GroupePilote_DCM42_G2_Mes_Vitesse']

else :
    print("entrez ci-dessous le lien vers le fichier excel contenant les enregistrements des capteurs")
    lien_excel = "historique_casse_film.xlsx"
  

# COMMAND ----------

# MAGIC %md
# MAGIC #### Aller chercher les données des incidents

# COMMAND ----------

# MAGIC %md
# MAGIC vérifier que les données sont bien importées

# COMMAND ----------

# va afficher le début de la table des incidents. Si des lignes sont sautées, où si il y a des lignes vides, ajuster skiprows   
df_incidents = pd.read_excel(
   '/Workspace/Users/theo.duquesne@armor-iimak.com/analyse des donnees/incidents_PBI.xlsx',
    skiprows=2
)
df_incidents.head()

# COMMAND ----------

# MAGIC %md
# MAGIC extraire les casses films et enregistrer la base de données

# COMMAND ----------

casses_film = df_incidents[df_incidents['Groupe d\'evenement'].apply(lambda x: Levenshtein.distance(x, 'Casse-Film') <= 2)]
casses_film.to_excel('casses_film.xlsx', index  = False)

# COMMAND ----------

# MAGIC %md
# MAGIC Création des objets casse-film à partir des infos de la base de données
# MAGIC
# MAGIC Ils sont alors tous rangés dans une liste

# COMMAND ----------

# vérifier la correction du lien vers le dossier contenant les fichiers des casse de film
liste_casse_film = []
for index, row in casse_film.iterrows():
    casse_film_i = CasseFilm(row['Date'], row['OF'], row['Equipement'], lien_casse_film = '/Workspace/Users/theo.duquesne@armor-iimak.com/analyse des donnees/dossier_casse_film')
    liste_casse_film.append(casse_film)


# COMMAND ----------

# MAGIC %md
# MAGIC Pour chacun des casses-film, on exécute une pipeline de traitements au nom explicite.

# COMMAND ----------

if aller_chercher_donnees_sql :
    for cf in liste_casse_film_juin :
        cf.extraire_donnees(sql = sql, nom_colonnes=nom_colonnes)
        cf.nettoyer_donnees()
        cf.determiner_date_reelle()
        cf.enregistrer()

else :
    for cf in liste_casse_film :
        cf.extraire_donnees(lien_vers_xlsx = lien_excel)
        cf.nettoyer_donnees()
        cf.determiner_date_reelle()
        cf.enregistrer()