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
# MAGIC ### Préparation de l'interface

# COMMAND ----------

dbutils.widgets.removeAll()
dbutils.widgets.dropdown(
    "enregistrer_indicateurs",   # identifiant interne (sans espace)
    "non",                       # valeur par défaut
    ["oui", "non"],              # liste des valeurs
    "Enregistrer indicateurs ?"  # label visible dans l'UI
)
lien =  'Entrainement_NN'+str(datetime.now()).replace(' ', '_').replace(':', '_').replace('.', '_').replace('-', '_')
dbutils.widgets.text( 'Lien d\'enregistrement des indicateurs', lien)


# COMMAND ----------

def charger_reglages(path_py):
    spec = importlib.util.spec_from_file_location("module_reglages", path_py)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module, module.reglages_nn, module.reglages_par_defaut

# --------------------------------------------------------
# Sauvegarder reglages_nn dans le fichier .py
# --------------------------------------------------------
def sauvegarder_reglages(path_py, reglages_nn):
    with open(path_py, "r") as f:
        contenu = f.read()

    nouveau_dict = f"reglages_nn = {repr(reglages_nn)}"
    contenu_modifie = []
    inside_nn = False
    for line in contenu.splitlines():
        if line.strip().startswith("reglages_nn"):
            contenu_modifie.append(nouveau_dict)
            inside_nn = True
        elif inside_nn and line.strip().startswith("reglages_par_defaut"):
            contenu_modifie.append(line)
            inside_nn = False
        elif not inside_nn:
            contenu_modifie.append(line)

    with open(path_py, "w") as f:
        f.write("\n".join(contenu_modifie))


# COMMAND ----------

dbutils.widgets.text('path vers la base de données des réseaux de neurones (faire "Copy URL/path -> full path)', "None")
path  = dbutils.widgets.get('path vers la base de données des réseaux de neurones (faire "Copy URL/path -> full path)')
if path in ("None", "", None):
    path = None
# On accède aux réseaux
acceder_reseaux(path)

# COMMAND ----------

import random
import time
from IPython.display import display, clear_output, HTML
import ipywidgets as widgets
from IPython.display import display, clear_output
import importlib.util
import os
import ast

# Liste d'emoji animaux
emojis_animaux = ["🐶","🐱","🐭","🐹","🐰","🦊","🐻","🐼","🐨","🐯",
                  "🦁","🐮","🐷","🐸","🐵","🐔","🐧","🐦","🐤","🦆"]

import ipywidgets as widgets
from IPython.display import display, clear_output
import random



def interface_reglages(path_py):
    # Chargement des réglages
    module, reglages_nn, reglages_par_defaut = charger_reglages(path_py)

    # Container principal
    container = widgets.VBox()
    display(container)

    out = widgets.Output()
    
    # Bouton global Annuler
    bouton_annuler = widgets.Button(description="Annuler")
    display(bouton_annuler)
    def on_annuler(_):
        with out:
            print("Opération annulée.")
            container.children = []
    bouton_annuler.on_click(on_annuler)

    def afficher_feu(couleur):
        with out:
            if couleur == "rouge":
                print('🔴')
            elif couleur == "vert":
                print('🟢')

    def etape1():
        nombre_de_rouleaux = reglages_nn.get("nombre de rouleaux")
        n_tours = reglages_nn.get("n_tours")

        w_rouleaux = widgets.Text(value=str(nombre_de_rouleaux), description="Nombre de rouleaux")
        w_rouleaux.style = {"description_width": "auto"}
        w_tours = widgets.Text(value=str(n_tours), description="Nombre d'itérations")
        w_tours.style = {"description_width": "auto"}
        checkbox_etape2 = widgets.Checkbox(description="Modifier d'autres réglages", value=True)
        bouton_valider1 = widgets.Button(description="Valider")
        bouton_suivant1 = widgets.Button(description="Suivant")

        def valider1(_):
            try:
                reglages_nn["nombre de rouleaux"] = int(w_rouleaux.value)
                reglages_nn["n_tours"] = int(w_tours.value)
                sauvegarder_reglages(path_py, reglages_nn)
                afficher_feu("vert")
            except Exception as e:
                with out:
                    print("Erreur lors de la conversion :", e)
                    afficher_feu("rouge")

        def suivant1(_):
            if checkbox_etape2.value:
                etape2()
            else:
                etape3()

        bouton_valider1.on_click(valider1)
        bouton_suivant1.on_click(suivant1)

        container.children = [w_rouleaux, w_tours, checkbox_etape2, bouton_valider1, bouton_suivant1, out]

    def etape2():
        out2 = widgets.Output()
        bouton_suivant2 = widgets.Button(description="Suivant")

        # Liste des réglages modifiables (hors nombre de rouleaux et n_tours)
        reglages_modifiables = list(reglages_par_defaut.keys())
        
        menu = widgets.Dropdown(options=reglages_modifiables, description="Choisir réglage")
        val_widget = widgets.Text(description="Nouvelle valeur", layout=widgets.Layout(width="50%"))
        bouton_valider2 = widgets.Button(description="Valider")
        bouton_reset = widgets.Button(description="Réinitialiser réglages")

        def afficher_reglages():
            with out2:
                clear_output(wait=True)
                for cle in reglages_modifiables:
                    valeur = reglages_nn.get(cle, reglages_par_defaut[cle])
                    print(f"{cle}: {valeur}")

        def valider2(_):
            cle = menu.value
            valeur = val_widget.value
            if valeur.strip().lower() == "supprimer":
                if cle in reglages_nn:
                    reglages_nn.pop(cle)
            else:
                try:
                    type_def = type(reglages_par_defaut[cle])
                    reglages_nn[cle] = type_def(valeur)
                except Exception as e:
                    with out2:
                        print(f"Erreur de conversion pour {cle} :", e)
                        return
            sauvegarder_reglages(path_py, reglages_nn)
            afficher_reglages()
            print("🟢")  # petit feu vert

        def reset2(_):
            for k in reglages_modifiables:
                if k in reglages_nn:
                    reglages_nn.pop(k)
            sauvegarder_reglages(path_py, reglages_nn)
            afficher_reglages()
            print("🟢 Réglages réinitialisés !")

        bouton_valider2.on_click(valider2)
        bouton_reset.on_click(reset2)
        bouton_suivant2.on_click(lambda _: etape3())

        container.children = [menu, val_widget, bouton_valider2, bouton_reset, bouton_suivant2, out2]
    
        afficher_reglages()

    def etape3():
        # Vider complètement l'interface widget
        container.children = []
        clear_output(wait=True)  # <-- important : nettoie tout l'affichage
        
        # Affichage des réglages finaux
        print("Réglages finaux :")
        for k, v in reglages_nn.items():
            print(f"{k}: {v}")

        # Lancement de l'entraînement
        print("Lancement de entrainer_reseau_v2()...")
    etape1()


# COMMAND ----------

# MAGIC %md
# MAGIC ## Interface

# COMMAND ----------

raise Exception ("🛑Stop (c'est normal) 🛑")

# COMMAND ----------

# MAGIC %md
# MAGIC #### Gestion des réglages
# MAGIC
# MAGIC -Si vous souhaitez créer un nouveau réseau de neurones, exécutez d'abord la cellule ci dessous pour modifier éventuellement les réglages (comme le nombre de rouleaux, le nombre d'itérations...)
# MAGIC Vous pouvez trouver une description détaillée de chacun des réglages et modifier les réglages par défaut dans le fichier reglages_bibos.py
# MAGIC
# MAGIC -Si vous souhaitez entraîner un réseau existant, ce n'est pas la peine

# COMMAND ----------

"""
Exécuter cette cellule pour gérer les réglages
"""

pathpy = gestion_reglages.path_vers_reglages_bibos + '/reglages_bibos.py'
flag = False
interface_reglages(pathpy)


# COMMAND ----------

# MAGIC %md
# MAGIC #### Entraînement du réseau
# MAGIC
# MAGIC -Si vous souhaitez créer un nouveau réseau, exécutez d'abord la cellule ci dessus afin de définir les réglages, puis la cellule ci dessous
# MAGIC
# MAGIC -Si vous souhaitez entraîner un réseau existant, vous pouvez directement exécuter la cellule suivante (assurez vous que dans les réglages le nombre d'itérations soit le bon)

# COMMAND ----------

"""
Exécuter cette cellule pour entraîner un réseau
"""
reinit_reglages_nn()
choix_indicateurs = {"oui": True, "non":False}[dbutils.widgets.get('enregistrer_indicateurs')]
entrainer_reseau_v2(sauvegarde = choix_indicateurs)
lien_entrainement = path2 + '/indicateurs_entrainement_reseaux_neurones/' + dbutils.widgets.get('Lien d\'enregistrement des indicateurs') + '.pkl'
if choix_indicateurs:
    with open(lien_entrainement, "wb") as f:
        pickle.dump(sauvegarde_nn.indicateurs, f)
