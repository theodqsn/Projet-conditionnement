# Databricks notebook source
# MAGIC %md
# MAGIC ## À cacher

# COMMAND ----------

# MAGIC %md
# MAGIC ### Import des modules

# COMMAND ----------

import os
path =dbutils.notebook.entry_point.getDbutils().notebook().getContext().notebookPath().get()
print(path)
parent = os.path.dirname(path)
path2 = '/Workspace'+ os.path.dirname(parent)
print(path2)
os.chdir(path2)
from Tout import *
import ipywidgets as widgets
from IPython.display import display, clear_output

# COMMAND ----------

# MAGIC %md
# MAGIC ### Construction de l'interface

# COMMAND ----------

import ipywidgets as widgets
from IPython.display import display, clear_output

def interface():
    # --- Champs texte ---
    boite1 = widgets.Text(
        description="Nombre de rouleaux :",
        placeholder="ex: 10",
        layout=widgets.Layout(width="400px"),
        style={"description_width": "200"}
    )
    boite2 = widgets.Text(
        description="Longueur du carton :",
        placeholder="ex: 1.0",
        layout=widgets.Layout(width="400px"),
        style={"description_width": "200"}
    )
    boite3 = widgets.Text(
        description="Largeur du carton :",
        placeholder="ex: 1.0",
        layout=widgets.Layout(width="400px"),
        style={"description_width": "200"}
    )
    boite4 = widgets.Text(
        description="Diamètre mandrin :",
        placeholder="ex: 0.1",
        layout=widgets.Layout(width="400px"),
        style={"description_width": "200"}
    )
    boite5 = widgets.Text(
        description="Nombre d'itérations :",
        placeholder="ex: 100",
        layout=widgets.Layout(width="400px"),
        style={"description_width": "200"}
    )
    boite6 = widgets.Text(
        description="Nom du fichier arbre :",
        placeholder="ex: arbre",
        layout=widgets.Layout(width="400px"),
        style={"description_width": "200"}
    )
    
    # --- Menus déroulants ---
    menu1 = widgets.Dropdown(
        options=["oui", "non"],
        value="non",
        description="Afficher indicateurs :",
        style={"description_width": "200"}
    )
    menu2 = widgets.Dropdown(
        options=["oui", "non"],
        value="non",
        description="Dessiner arbre :",
        style={"description_width": "200"}
    )


    # --- Bouton ---
    bouton = widgets.Button(
        description="Exécuter",
        button_style="success"
    )

    # --- Zone de sortie ---
    sortie = widgets.Output()

    # --- Callback ---
    def on_button_click(b):
        with sortie:
            clear_output()
            try:
                params = {
                    "n_rouleaux": int(boite1.value),
                    "longueur_carton": float(boite2.value),
                    "largeur_carton": float(boite3.value),
                    "rayon_mandrin": 0.5*float(boite4.value),
                    "n_it": int(boite5.value),
                    "choix_indicateurs": (menu1.value == "oui"),
                    "choix_arbre": (menu2.value == "oui"),
                    "fichier_dot": boite6.value.strip(),
                   
                }
                # On vérifie que tout est rempli
                if any(v == "" for v in [boite1.value, boite2.value, boite3.value, boite4.value, boite5.value]):
                    print("⚠️ Merci de remplir toutes les cases.")
                else:
                    interface.parametres = params
                    print("✅ Vous pouvez désormais cliquer sur la cellule ci-dessous")
            except Exception as e:
                print(f"Erreur : {e}")

    bouton.on_click(on_button_click)

    # --- Affichage ---
    ui = widgets.VBox([boite1, boite2, boite3, boite4, boite5, menu1, menu2, boite6, bouton, sortie])
    display(ui)


# COMMAND ----------

def exe():
  params = interface.parametres
  lo = params["longueur_carton"]
  la = params["largeur_carton"]
  rayon_mandrin = params["rayon_mandrin"]
  n_rouleaux = params["n_rouleaux"]
  n_it = params["n_it"]
  choix = params["choix_indicateurs"]
  choix_arbre = params["choix_arbre"]
  fichier_dot = params["fichier_dot"]
  
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
  arrg_sortie = evaluer_MCTS(arrg)
  dessiner_gradient(evaluer_MCTS.arrangement, reglages={'fonction de perte inch': perte_inch_lisse_v2}, fleches = False)
  exe.etat_init = etat_init
  exe.best_noeud = best_noeud
  exe.arrg_sortie = arrg_sortie
  print(f"Le rayon maximal obtenu est {evaluer_MCTS.arrangement['grand rayon']*coeff}")

# COMMAND ----------

def indic():
  if interface.parametres["choix_indicateurs"]:
    print("Vous avez choisi d'afficher les indicateurs")
    for cle in indic[0].keys() :
      plt.plot([indic[i][cle] for i in range(len(indic))], '-o')
      plt.title(cle)
      plt.show()


# COMMAND ----------

# MAGIC %md
# MAGIC ## Interface

# COMMAND ----------

raise Exception ("🛑Stop (c'est normal) 🛑")

# COMMAND ----------

interface()

# COMMAND ----------

exe()