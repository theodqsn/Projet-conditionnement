# Databricks notebook source
# MAGIC %md
# MAGIC ## À cacher

# COMMAND ----------


import os
path =dbutils.notebook.entry_point.getDbutils().notebook().getContext().notebookPath().get()
parent = os.path.dirname(path)
path2 =  '/Workspace' + os.path.dirname(parent) + '/'
print(path2)
#ancien = '/Workspace/Users/theo.duquesne@armor-iimak.com/test deplacement'
actuel =  path2

# COMMAND ----------



# COMMAND ----------

import os
import importlib.util

def charger_ancien():
    """
    Charge la variable 'ancien' depuis le fichier ancien.py
    situé dans le même dossier que le notebook courant.
    """
    notebook_dir = os.path.dirname(path)  # 'path' = chemin du notebook
    ancien_file = os.path.join('/Workspace' + notebook_dir, "ancien.py")

    spec = importlib.util.spec_from_file_location("ancien", ancien_file)
    ancien_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(ancien_module)
    return  ancien_module.ancien


def identifier():
    """
    Parcourt récursivement les .py sous path2.
    Affiche les fichiers et lignes contenant 'ancien'.
    """
    ancien = charger_ancien()

    for root, dirs, files in os.walk(path2):
        for file in files:
            if file.endswith(".py") or file.endswith(".json")or file.endswith(".JSON"):
                filepath = os.path.join(root, file)
                try:
                    with open(filepath, "r", encoding="utf-8") as f:
                        lines = f.readlines()
                    
                    lignes_trouvees = []
                    for num, line in enumerate(lines, start=1):
                        if ancien in line:
                            if line.strip().startswith(ancien) or f"'{ancien}" in line or f'"{ancien}' in line:
                                lignes_trouvees.append((num, line.strip()))
                    
                    if lignes_trouvees:
                        print(f"🐳 on est dans le fichier {file}, {filepath}")
                        for num, content in lignes_trouvees:
                            print(f"    🐠 ligne {num}: {content}")
                except Exception as e:
                    print(f"Impossible de lire {filepath}: {e}")


import os
import ast

def remplacer(grand_nettoyage=False):
    """
    Parcourt récursivement les .py et .json sous path2,
    et remplace les occurrences de `ancien` par `actuel`.

    Améliorations :
    - Ne modifie jamais la liste `ancetres` dans ancien.py
    - Ajoute `ancien` à la liste `ancetres` avant de commencer
    - Si grand_nettoyage=True : remplace aussi tous les liens présents dans `ancetres`
    """

    # Charger ancien et actuel
    ancien = charger_ancien()

    # Charger la liste des ancêtres dans ancien.py
    try:
        with open(path2 + "Notebook executables/ancien.py", "r", encoding="utf-8") as f:
            source = f.read()
        tree = ast.parse(source, filename="ancien.py")

        ancetres = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Assign):
                for target in node.targets:
                    if isinstance(target, ast.Name) and target.id == "ancetres":
                        if isinstance(node.value, (ast.List, ast.Tuple)):
                            ancetres = [elt.s for elt in node.value.elts if isinstance(elt, ast.Constant)]
        # Ajouter ancien à la liste
        if ancien not in ancetres:
            ancetres.append(ancien)
    except Exception as e:
        print(f"⚠️ Impossible de charger la liste ancetres dans ancien.py : {e}")
        ancetres = [ancien]

    # Confirmation
    if grand_nettoyage:
        confirmation = input(f"👉 Confirmer le GRAND NETTOYAGE (remplacement des chemins {ancetres}) ? (o/n) : ").strip().lower()
    else:
        confirmation = input(f"👉 Confirmer le remplacement de tous les chemins commençant par '{ancien}' ? (o/n) : ").strip().lower()

    if confirmation != "o":
        print("❌ Remplacement annulé par l’utilisateur.")
        return

    # Parcours des fichiers
    for root, dirs, files in os.walk(path2):
        for file in files:
            if file.endswith(".py") or file.endswith(".json") or file.endswith(".JSON"):
                filepath = os.path.join(root, file)

                try:
                    with open(filepath, "r", encoding="utf-8") as f:
                        lines = f.readlines()

                    new_lines = []
                    modified = False
                    lignes_modifiees = []

                    for num, line in enumerate(lines, start=1):

                        # Ne pas toucher à la liste "ancetres"
                        if file == "ancien.py" and "ancetres" in line:
                            new_lines.append(line)
                            continue

                        # Définir la liste des candidats à remplacer
                        cibles = ancetres if grand_nettoyage else [ancien]

                        # Remplacements
                        replaced_line = line
                        for cible in cibles:
                            if cible in replaced_line:
                                if replaced_line.strip().startswith(cible) or f"'{cible}" in replaced_line or f'"{cible}' in replaced_line:
                                    replaced_line = replaced_line.replace(cible, actuel)
                                    modified = True

                        if replaced_line != line:
                            lignes_modifiees.append((num, line.strip(), replaced_line.strip()))

                        new_lines.append(replaced_line)

                    # Écriture si modifié
                    if modified:
                        with open(filepath, "w", encoding="utf-8") as f:
                            f.writelines(new_lines)
                        print(f"\n🐳 on est dans le fichier {file}, {filepath}")
                        for num, avant, apres in lignes_modifiees:
                            print(f"    🐠 ligne {num}:")
                            print(f"       avant: {avant}")
                            print(f"       après: {apres}")

                except Exception as e:
                    print(f"Impossible de traiter {filepath}: {e}")


# COMMAND ----------

# MAGIC %md
# MAGIC ## Programmes

# COMMAND ----------

identifier()

# COMMAND ----------

remplacer(grand_nettoyage=False)