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
# MAGIC #### Extraction des données

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

    liste_dico = selection_reseaux.liste_dico
    liste_resultats = []

    for dico in liste_dico :
        reinit_reglages_nn()
        longueur_boite = dico["dim_x"]
        largeur_boite = dico["dim_y"]
        norm = (longueur_boite+largeur_boite)/2
        rayon_mandrin = dico["rayon_mandrin"]
        entrees = torch.tensor([longueur_boite/norm, largeur_boite/norm, rayon_mandrin/norm])
        nom_reseau = dico["nom reseau"]
        print(f" 🌞 On utilise le réseau {nom_reseau} pour l'arrangement {dico['Reference']}  🌞")
        tr = TesterReseau(nom_reseau, fonction_evaluation=wrapper_evalue_arrg)
        tr.recuperer_reseau()
        print(tr.reglages)
        entrees_propres, resultat = tr.evaluer(entrees)
    

        motif = os.path.join(path2, "fichier_temporaire_peche_arrangements*")

        liste_arrangements = []
        for fichier in glob.glob(motif):
            with open(fichier, "rb") as f:
            
                data = pickle.load(f)
                liste_arrangements.append(data)
            os.remove(fichier)

        ar = liste_arrangements[0]

        liste_resultats.append(
            {
                'Reference' : dico["Reference"], 
                'Norme' : norm, 
                'arrangement' : ar, 
                'Rayon maximal' : ar['grand rayon'] * norm
            }
        )
    for resultat in liste_resultats:
        print(f"🐼 Référence : {resultat['Reference']} 🐼")
        print(f" Diamètre maximal : {2*resultat['Rayon maximal']}")
        dessiner_gradient(resultat['arrangement'], reglages=None, beta=0.1, fleches=False)


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

import ipywidgets as widgets
from IPython.display import display

def selection_reseaux():
    # Récupère la liste construite par extraction()
    liste_dico = getattr(extraction, "liste_dico", None)
    if liste_dico is None:
        print("❌ Aucun 'liste_dico' trouvé dans extraction. Avez-vous exécuté extraction() ?")
        return
    
    # Récupère les nombres de rouleaux distincts
    nb_rouleaux_uniques = sorted(set(d["nombre_rouleaux"] for d in liste_dico))
    
    # Crée un dropdown par nombre de rouleaux
    dropdowns = {}
    for n in nb_rouleaux_uniques:
        options = choix_reseaux(n)  # liste de noms compatibles
        if not options:
            options = ["(aucun disponible)"]
        dd = widgets.Dropdown(
            options=options,
            description=f"{n} rouleaux",
            layout=widgets.Layout(width="400px"),
            style={"description_width": "120px"}
        )
        dropdowns[n] = dd
    
    bouton_valider = widgets.Button(description="Valider", button_style="success")
    out = widgets.Output()

    def on_valider(_):
        with out:
            out.clear_output()
            for n, dd in dropdowns.items():
                choix = dd.value
                # On met à jour tous les dicos correspondant à ce nombre de rouleaux
                for d in liste_dico:
                    if d["nombre_rouleaux"] == n:
                        d["nom reseau"] = choix
            print("✅ Mise à jour terminée : 'nom reseau' ajouté dans chaque dictionnaire")

    bouton_valider.on_click(on_valider)
    selection_reseaux.liste_dico = liste_dico
    ui = widgets.VBox(list(dropdowns.values()) + [bouton_valider, out])
    display(ui)


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

selection_reseaux()

# COMMAND ----------

# MAGIC %md
# MAGIC #### Remplir les paramètres

# COMMAND ----------

# MAGIC %md
# MAGIC #### Exécuter

# COMMAND ----------

peche_arrangement()
exe()