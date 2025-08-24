# Databricks notebook source
# MAGIC %md
# MAGIC ## À cacher

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

def exe():
  params = interface.parametres
  nom_fichier = params['lien indicateurs']
  if os.path.getsize(nom_fichier) == 0:
        raise ValueError(f"⚠️ Le fichier {nom_fichier} est vide, impossible de le charger.")

  with open(nom_fichier, "rb") as f:
        liste_de_dicts_recup = pickle.load(f)

  for k in liste_de_dicts[0].keys():
    a_plot = [liste_de_dicts[i][k] for i in range(len(liste_de_dicts))]
    plt.figure(figsize=(20, 5))
    plt.plot(a_plot)
    plt.title(k)
plt.show()


# COMMAND ----------

import ipywidgets as widgets
from IPython.display import display, clear_output

def interface():
    # --- Champs texte ---
    boite1 = widgets.Text(
        description="Lien vers le fichier des indicateurs :",
        layout=widgets.Layout(width="1000px"),
        style={"description_width": "500px"} 
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
                    "lien indicateurs": boite1.value
                }
                # Vérification que la case est remplie
                if boite1.value.strip() == "":
                    print("⚠️ Merci de remplir toutes les cases.")
                else:
                    interface.parametres = params
                    print("✅ Vous pouvez désormais cliquer sur la cellule ci-dessous")
            except Exception as e:
                print(f"Erreur : {e}")

    bouton.on_click(on_button_click)

    # --- Affichage ---
    ui = widgets.VBox([boite1, bouton, sortie])
    display(ui)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Interface

# COMMAND ----------

raise Exception ("🛑Stop (c'est normal) 🛑")

# COMMAND ----------

interface()

# COMMAND ----------

exe()