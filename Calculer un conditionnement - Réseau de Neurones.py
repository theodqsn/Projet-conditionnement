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
import glob


# COMMAND ----------

# MAGIC %md
# MAGIC #### À la pêche aux réseaux de neurones

# COMMAND ----------

dbutils.widgets.text('path vers la base de données des réseaux de neurones (faire "Copy URL/path -> full path)', "None")
path  = dbutils.widgets.get('path vers la base de données des réseaux de neurones (faire "Copy URL/path -> full path)')
if path in ("None", "", None):
    path = None
# On accède aux réseaux
acceder_reseaux(path)

# COMMAND ----------

# Exemple de fonction générant la liste (à remplacer par la tienne)
def choix_reseaux(n):
    chemin_base = path_nn()
    with open(chemin_base, 'r') as f:
        base = json.load(f)
    liste = []
    for reseau in base:
      if reseau["Nombre de rouleaux"]==n:
        liste.append(reseau["Nom"])
    return liste

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
# MAGIC #### Conversion des parametres en une entree

# COMMAND ----------

import ipywidgets as widgets
from IPython.display import display

def interface():
    # --- Boîtes de texte ---
    boite1 = widgets.Text(
        description="Nombre de rouleaux",
        layout=widgets.Layout(width="400px"),   # largeur de la case
        style={"description_width": "130px"}  # le texte de description prend toute la place
    )

    boite2 = widgets.Text(
        description="Diamètre mandrin",
        layout=widgets.Layout(width="400px"),
        style={"description_width": "130px"}
    )

    boite3 = widgets.Text(
        description="Longueur boîte",
        layout=widgets.Layout(width="400px"),
        style={"description_width": "130px"}
    )

    boite4 = widgets.Text(
        description="Largeur boîte",
        layout=widgets.Layout(width="400px"),
        style={"description_width": "130px"}
    )

    bouton_valider = widgets.Button(description="Valider", button_style="info")

    # Zone où l’on insèrera le menu + bouton Exécuter après validation
    zone_dynamique = widgets.VBox([])

    # Sortie pour messages
    out = widgets.Output()

    # Dictionnaire de résultats (il sera rempli au clic sur Exécuter)
    resultats = {}
    interface.resultats = resultats   # accessible depuis la cellule suivante

    def on_valider(_):
        with out:
            out.clear_output()
            # Lire et vérifier n
            try:
                n = int(boite1.value)
            except ValueError:
                print("⚠️ Entrez un entier valide pour le nombre de rouleaux.")
                return

            # Construire la liste via TA fonction (doit exister dans ton namespace)
            options = choix_reseaux(n)
            if not options:
                print("⚠️ Aucun réseau trouvé pour n =", n)
                zone_dynamique.children = []
                return

            # Créer un nouveau menu + bouton Exécuter (ne réutilise pas l’ancien pour éviter les handlers multiples)
            menu = widgets.Dropdown(options=options, value=options[0], description="Réseau:")
            bouton_exec = widgets.Button(description="Exécuter", button_style="success")

            def on_exec(__):
                with out:
                    out.clear_output()
                    try:
                        resultats["nb_rouleaux"]   = int(boite1.value)
                        resultats["rayon_mandrin"] = 0.5*float(boite2.value)
                        resultats["longueur_boite"] = float(boite3.value)
                        resultats["largeur_boite"]  = float(boite4.value)
                        resultats["choix_menu"] = menu.value
                        print("✅ Vous pouvez désormais exécuter la cellule suivante")
                        interface.parametres = resultats
                    except ValueError:
                        print("⚠️ Merci de remplir correctement toutes les cases (Zahlen).")

            bouton_exec.on_click(on_exec)
            zone_dynamique.children = [menu, bouton_exec]
            print(f"✅ {len(options)} option { '' if len(options)==1 else 's'} chargée{ '' if len(options)==1 else 's'}. Choisissez puis cliquez sur Exécuter.")

    bouton_valider.on_click(on_valider)

    # Affichage initial
    ui = widgets.VBox([boite1, boite2, boite3, boite4, bouton_valider, zone_dynamique, out])
    display(ui)


# COMMAND ----------

# MAGIC %md
# MAGIC #### Évaluation via la class TesterReseau

# COMMAND ----------

import glob
import os
import pickle

def exe():
    if not hasattr(interface, 'parametres'):
        raise Exception("Vous devez exécuter la cellule précédente et remplir les cases")
    resultats = interface.parametres
    nb_rouleaux = resultats["nb_rouleaux"]
    rayon_mandrin = resultats["rayon_mandrin"]
    longueur_boite = resultats["longueur_boite"]
    largeur_boite = resultats["largeur_boite"]
    norm = (longueur_boite + largeur_boite)/2
    nom_reseau = resultats["choix_menu"]
    tr = TesterReseau(nom_reseau, fonction_evaluation=wrapper_evalue_arrg)
    entrees = torch.tensor([longueur_boite/norm, largeur_boite/norm, rayon_mandrin/norm])
    entrees_propres, resultat = tr.evaluer(entrees)
    
    # --- Chargement des arrangements depuis les fichiers temporaires ---
    liste_arrangements = []
    motif = os.path.join(path2, "fichier_temporaire_peche_arrangements*")

    for fichier in glob.glob(motif):
        with open(fichier, "rb") as f:
          
            data = pickle.load(f)
            liste_arrangements.append(data)
        os.remove(fichier)

    ar = liste_arrangements[0]

    print(f"Diamètre rouleau trouvé : {2*norm*resultat[0]} ")
    dessiner_gradient(ar, reglages=None, beta=0.1, fleches=False)


# COMMAND ----------

# MAGIC %md
# MAGIC ## Interface

# COMMAND ----------

raise Exception ("🛑Stop (c'est normal) 🛑")

# COMMAND ----------

# MAGIC %md
# MAGIC #### Remplir les paramètres

# COMMAND ----------

interface()

# COMMAND ----------

# MAGIC %md
# MAGIC #### Exécuter

# COMMAND ----------

peche_arrangement()
exe()