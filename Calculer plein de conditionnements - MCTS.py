# Databricks notebook source
# MAGIC %md
# MAGIC ## À cacher

# COMMAND ----------

# MAGIC %pip install openpyxl

# COMMAND ----------


import os
path =dbutils.notebook.entry_point.getDbutils().notebook().getContext().notebookPath().get()
print(path)
parent = os.path.dirname(path)
path2 = '/Workspace'+ os.path.dirname(parent)
print(path2)
os.chdir(path2)
import Tout
from Tout import *
import ipywidgets as widgets
from IPython.display import display, clear_output
import glob



# COMMAND ----------

# MAGIC %md
# MAGIC #### À la pêche aux réseaux de neurones

# COMMAND ----------

import uuid
import pickle

def wrapper_evalue_arrg(args):
    entree, dims, petit_rayon, k , reg = args
    evalu = evalue_arrg(entree, dims, petit_rayon, k, reg)
    arrg = evalue_arrg.arrg
    # identifiant unique lisible
    identifiant = uuid.uuid4().hex  
    fichier = path2 + f'/fichier_temporaire_peche_arrangements_{identifiant}.pkl'
    with open(fichier, 'wb') as f:
        pickle.dump(arrg, f)
    return evalu


# COMMAND ----------

# MAGIC %md
# MAGIC #### Extraction des données

# COMMAND ----------

dbutils.widgets.removeAll()

dbutils.widgets.dropdown(
    "afficher arbre",
    "Non",
    ["Oui", "Non"]
)

dbutils.widgets.dropdown(
    "calculer indicateurs",
    "Non",
    ["Non", "Oui"]
)

dbutils.widgets.text(
    "nombre iterations",
    "200"
)


# COMMAND ----------

# MAGIC %md
# MAGIC #### Conversion des parametres en une entree

# COMMAND ----------

# MAGIC %md
# MAGIC #### Évaluation via l'algo de MCTS

# COMMAND ----------

# MAGIC %md
# MAGIC #### Extraction des données depuis le .xlsx

# COMMAND ----------

import pandas as pd
import ipywidgets as widgets
from IPython.display import display

noms_par_defaut = ["Nb rlx", "Ø MRE (mm)", "Longueur int (mm)", 
                   "Largeur int (mm)", "Hauteur int (mm)"]

def extraction():
    boite1 = widgets.Text(
        description="Lien vers le xlsx",
        layout=widgets.Layout(width="400px"), 
        style={"description_width": "130px"}
    )
    bouton_valider = widgets.Button(description="Valider", button_style="info")
    out = widgets.Output()
    zone_dynamique = widgets.VBox([])

    def on_valider(_):
        with out:
            out.clear_output()
            url = boite1.value.strip()
            try:
                df = pd.read_excel(url)
                print("✅ Fichier chargé avec succès")

                colonnes = df.columns.tolist()
                mapping_widgets = {}

                # Si colonnes différentes → on demande un mapping
                if colonnes != noms_par_defaut:
                    print("⚠️ Les noms des colonnes diffèrent des noms par défaut.")
                    print("Colonnes détectées :", colonnes)

                    corrections = []
                    for attendu in noms_par_defaut:
                        dd = widgets.Dropdown(
                            options=colonnes,
                            description=f"{attendu}",
                            layout=widgets.Layout(width="400px"),
                            style={"description_width": "180px"}
                        )
                        mapping_widgets[attendu] = dd
                        corrections.append(dd)

                    bouton_confirmer = widgets.Button(description="Confirmer mapping", button_style="success")

                    def on_confirmer(_):
                        with out:
                            out.clear_output()
                            mapping = {attendu: w.value for attendu, w in mapping_widgets.items()}
                            print("✅ Mapping choisi :", mapping)

                            # On renomme le dataframe
                            df_renomme = df.rename(columns={v: k for k, v in mapping.items()})

                            traiter_dataframe(df_renomme)

                    bouton_confirmer.on_click(on_confirmer)
                    ui_mapping = widgets.VBox(corrections + [bouton_confirmer])
                    display(ui_mapping)
                else:
                    traiter_dataframe(df)

            except Exception as e:
                print(f"❌ Erreur lors du chargement : {e}")

    def traiter_dataframe(df):
        # Vérification mandrin
        if "mandrin_debordant" not in df.columns:
            df["mandrin_debordant"] = False

        liste_dico = []
        for _, row in df.iterrows():
            nb_rouleaux = int(row["Nb rlx"])
            diam_mandrin = float(row["Ø MRE (mm)"])
            longueur = float(row["Longueur int (mm)"])
            largeur = float(row["Largeur int (mm)"])
            hauteur = float(row["Hauteur int (mm)"])
            debord = str(row["mandrin_debordant"]).strip()
            mandrin_debordant = debord.lower() in ["oui", "o", "true"]

            # Orientation X
            dic_x = {
                "Reference": f"{nb_rouleaux}_{diam_mandrin}_{largeur}_{hauteur}",
                "nombre_rouleaux": nb_rouleaux,
                "rayon_mandrin": diam_mandrin / 2,
                "dim_x": largeur,
                "dim_y": hauteur
            }
            liste_dico.append(dic_x)

            # Orientation Y
            dic_y = {
                "Reference": f"{nb_rouleaux}_{diam_mandrin}_{longueur}_{hauteur}",
                "nombre_rouleaux": nb_rouleaux,
                "rayon_mandrin": diam_mandrin / 2,
                "dim_x": longueur,
                "dim_y": hauteur
            }
            liste_dico.append(dic_y)

            # Orientation Z (si pas débordant)
            if not mandrin_debordant:
                dic_z = {
                    "Reference": f"{nb_rouleaux}_{diam_mandrin}_{longueur}_{largeur}",
                    "nombre_rouleaux": nb_rouleaux,
                    "rayon_mandrin": diam_mandrin / 2,
                    "dim_x": longueur,
                    "dim_y": largeur
                }
                liste_dico.append(dic_z)

        print("✅ Extraction terminée")
        print(f"{len(liste_dico)} configurations créées")
        extraction.liste_dico = liste_dico

    bouton_valider.on_click(on_valider)
    ui = widgets.VBox([boite1, bouton_valider, zone_dynamique, out])
    display(ui)


# COMMAND ----------


def exe():
  liste_arrg = []
  choix = dbutils.widgets.get( "calculer indicateurs")=="Oui"
  choix_arbre = dbutils.widgets.get( "afficher arbre") == "Oui"
  n_it = int(dbutils.widgets.get("nombre iterations"))

  for params in extraction.liste_dico:
    rayon_mandrin = params["rayon_mandrin"]
    n_rouleaux = params["nombre_rouleaux"]
    fichier_dot = params["Reference"]+'_arbre'
    lo= params["dim_x"]
    la = params["dim_y"]
    
    reinit_reglages_MCTS()
    coeff = (lo + la)/2
    rayon_mandrinp= rayon_mandrin/coeff
    lop = lo/coeff
    lap = la/coeff
    etat_init= init_etat_initial(n_rouleaux, [lop,lap,1] ,rayon_mandrin/coeff)

    if choix : 
      init_indicateurs_mtcs()

    best_noeud  = best_leaf_mcts(etat_init, n_it)

    if choix :
      indic= get_indicateurs_mtcs()

    if choix_arbre :
      path = path2 + "/dessins_arbres/" + fichier_dot + '.dot'
      if os.path.exists(path):
          os.remove(path)
          print(f"Fichier {path} supprimé.")
      else:
          print(f"Fichier {path} absent.")
      arbre = best_leaf_mcts.arbre
      dessiner_arbre(arbre, path2 + "/dessins_arbres/" + fichier_dot)

    arrg = best_noeud.etat
    liste_arrg.append(arrg)

  for arrg in liste_arrg:
    arrg_sortie = evaluer_MCTS(arrg)
    print(f"🐼 Référence : {resultat['Reference']} 🐼")
    print(f"Le diamètre maximal obtenu est {2*evaluer_MCTS.arrangement['grand rayon']*coeff}")
    dessiner_gradient(evaluer_MCTS.arrangement, reglages={'fonction de perte inch': perte_inch_lisse_v2}, fleches = False)
  

# COMMAND ----------

# MAGIC %md
# MAGIC ## Interface

# COMMAND ----------

raise Exception ("🛑Stop (c'est normal) 🛑")

# COMMAND ----------

# MAGIC %md
# MAGIC #### Extraire les données

# COMMAND ----------

extraction()

# COMMAND ----------

# MAGIC %md
# MAGIC #### Exécuter

# COMMAND ----------

exe()