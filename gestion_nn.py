import os
import json
import torch
from datetime import datetime
from gestion_reglages import reglages
import importlib

def acceder_reseaux(path=None):
    if path is None:
        path = reglages('path vers la base de donnees des reseaux')
    
    acceder_reseaux.path_nn = path
    print(f"vous allez désormais chercher des réseaux de neurones dans la base de données à l’emplacement : \n {path}")

    if not os.path.isfile(path):
        print(f"Le fichier '{path}' n'existe pas.")
        creer = input("Souhaitez-vous le créer ? (o/n) : ").strip().lower()
        if creer == 'o':
            with open(path, 'w') as f:
                json.dump([], f, indent=2)
            print(f"Fichier créé à l’emplacement : {path}")
        else:
            print("Aucun fichier n’a été créé.")

def path_nn():
    try:
        path = acceder_reseaux.path_nn
        if not path:
            print("ERREUR : aucun chemin vers la base de données de réseaux de neurones n’a été défini.")
            print('\n')
            print('svp, appelez la fonction 🌸🌸 acceder_reseaux(path) 🌸🌸 pour définir un chemin. \n Notez path entre guillements')
            print('\n')
            print('Vous pouvez aussi choisir d\'appeler la fonction 🌸🌸 acceder_reseaux() 🌸🌸 vous allez être dirigé vers la base de données')
            raise ValueError("🌼")
        return path
    except AttributeError:
            print("ERREUR : aucun chemin vers la base de données de réseaux de neurones n’a été défini.")
            print('\n')
            print('svp, appelez la fonction 🌸🌸 acceder_reseaux(path) 🌸🌸 pour définir un chemin. \n Notez path entre guillements')
            print('\n')
            print('Vous pouvez aussi choisir d\'appeler la fonction 🌸🌸 acceder_reseaux() 🌸🌸 vous allez être dirigé vers la base de données')
            raise ValueError("🌼")

def afficher_reseaux(liste=None):
    import shutil  # pour détecter la largeur du terminal

    # Enregistrement du workspace courant
    workspace_actuel = os.getcwd()

    try:
        # Aller dans le dossier contenant la base de données
        chemin_base = path_nn()
        dossier_base = os.path.dirname(chemin_base)
        os.chdir(dossier_base)

        # Chargement de la base de données
        with open(chemin_base, 'r') as f:
            base = json.load(f)

        # Filtrage selon le type de `liste`
        if liste is None:
            reseaux_a_afficher = base
        elif isinstance(liste, str):
            reseaux_a_afficher = [r for r in base if r.get("Nom") == liste]
        elif isinstance(liste, int):
            reseaux_a_afficher = [r for r in base if r.get("Nombre de rouleaux") == liste]
        elif isinstance(liste, list):
            reseaux_a_afficher = [r for r in base if r.get("Nom") in liste]
        else:
            reseaux_a_afficher = []  # Cas inattendu

        # Définir les largeurs de colonnes
        largeur_terminal = shutil.get_terminal_size().columns
        col_nom = 25
        col_rouleaux = 12
        col_score = 10
        col_autre = largeur_terminal - col_nom - col_rouleaux - col_score - 6  # pour marges et séparateurs

        # En-tête
        print("-" * largeur_terminal)
        print(f"{'Nom':<{col_nom}} | {'Rouleaux':^{col_rouleaux}} | {'Score':^{col_score}} | {'Entraînement':<{col_autre}}")
        print("-" * largeur_terminal)

        # Contenu
        for r in reseaux_a_afficher:
            nom = r.get("Nom", "(sans nom)")[:col_nom]
            rouleaux = str(r.get("Nombre de rouleaux", "?"))
            score = str(r.get("Score", "—"))
            entrainement = str(r.get("Entrainement", "?"))
            print(f"{nom:<{col_nom}} | {rouleaux:^{col_rouleaux}} | {score:^{col_score}} | {entrainement:<{col_autre}}")

        print("-" * largeur_terminal)

    finally:
        os.chdir(workspace_actuel)

def afficher_reseaux_technique(liste=None):
    # Enregistrement du workspace courant
    workspace_actuel = os.getcwd()

    try:
        # Aller dans le dossier contenant la base de données
        chemin_base = path_nn()
        dossier_base = os.path.dirname(chemin_base)
        os.chdir(dossier_base)

        # Chargement de la base de données
        with open(chemin_base, 'r') as f:
            base = json.load(f)

        # Filtrage selon le type de `liste`
        if liste is None:
            reseaux_a_afficher = base
        elif isinstance(liste, str):
            reseaux_a_afficher = [r for r in base if r.get("Nom") == liste]
        elif isinstance(liste, int):
            reseaux_a_afficher = [r for r in base if r.get("Nombre de rouleaux") == liste]
        elif isinstance(liste, list):
            reseaux_a_afficher = [r for r in base if r.get("Nom") in liste]
        else:
            reseaux_a_afficher = []  # Cas inattendu

        # Affichage technique
        for r in reseaux_a_afficher:
            print('🌻🌻🌻')
            print(f"{r.get('Nom', '(sans nom)')} | Rouleaux : {r.get('Nombre de rouleaux', '?')} | Score : {r.get('Score', None)}")
            print(f"  Date création         : {r.get('Date creation', 'inconnue')}")
            print(f"  Entraînement          : {r.get('Entrainement', '?')}")
            print(f"  Nombre de couches     : {r.get('Nombre de couches', '?')}")
            k = r.get('Nombre d\'explorateurs', '?')
            print(f"  Nombre d'explorateurs : {k}")
            print(f"  Taille couches internes : {r.get('Taille des couches internes', '?')}")
            print(f"  Commentaires       : {r.get('Commentaires', 'aucun')}")
            


    finally:
        # Retour dans le workspace initial
        os.chdir(workspace_actuel)

def supprimer_reseau(nom=None, conf= False):
    workspace_actuel = os.getcwd()
    try:
        chemin_base = path_nn()
        dossier_base = os.path.dirname(chemin_base)
        os.chdir(dossier_base)

        # Chargement de la base
        with open(chemin_base, 'r') as f:
            base = json.load(f)

        # Si nom est None, afficher les réseaux disponibles
        if nom is None:
            print("Réseaux disponibles :")
            for r in base:
                print(f"• {r.get('Nom', '(sans nom)')}")
            nom = input("Quel réseau voulez-vous modifier ?  ").strip()

        # Recherche du réseau à supprimer
        a_supprimer = next((r for r in base if r.get("Nom") == nom), None)

        if a_supprimer is None:
            print(f"Aucun réseau nommé '{nom}' n’a été trouvé dans la base.")
            return

        if not conf :
            # Demande de confirmation
            confirmation = input(f"Êtes-vous sûr de vouloir supprimer 🍫🍫tous les fichiers🍫🍫 associés à '{nom}' ? (o/n) ").strip().lower()
            if confirmation != 'o':
                print("Suppression annulée.")
                return

        # Suppression du fichier de paramètres principal
        chemin_parametres = a_supprimer.get("lien parametres")
        if chemin_parametres and os.path.exists(chemin_parametres):
            try:
                os.remove(chemin_parametres)
                print(f"Fichier principal '{chemin_parametres}' supprimé.")
            except Exception as e:
                print(f"Erreur lors de la suppression du fichier principal : {e}")

                chemin_parametres = a_supprimer.get("lien parametres")

        # Supression du fichier de réglages
        chemin_reglages = a_supprimer.get("lien reglages")
        if chemin_reglages and os.path.exists(chemin_reglages):
            try:
                os.remove(chemin_reglages)
                print(f"Fichier principal '{chemin_reglages}' supprimé.")
            except Exception as e:
                print(f"Erreur lors de la suppression du fichier principal : {e}")

        # Suppression des fichiers listés dans l'historique
        historique = a_supprimer.get("historique", [])
        for version in historique:
            lien_autosave = version.get("lien autosave")
            if lien_autosave and os.path.exists(lien_autosave):
                try:
                    os.remove(lien_autosave)
                    print(f"Fichier historique '{lien_autosave}' supprimé.")
                except Exception as e:
                    print(f"Erreur lors de la suppression de '{lien_autosave}' : {e}")

        # Mise à jour de la base
        nouvelle_base = [r for r in base if r.get("Nom") != nom]
        with open(chemin_base, 'w') as f:
            json.dump(nouvelle_base, f, indent=2)

        print(f"Le réseau '{nom}' et tous les fichiers associés ont été supprimés de la base.")

    finally:
        os.chdir(workspace_actuel)

def afficher_commentaires(nom=None):
    workspace_actuel = os.getcwd()
    try:
        chemin_base = path_nn()
        os.chdir(os.path.dirname(chemin_base))

        with open(chemin_base, 'r') as f:
            base = json.load(f)

        # Si nom est None, afficher les réseaux disponibles
        if nom is None:
            print("Réseaux disponibles :")
            for r in base:
                print(f"• {r.get('Nom', '(sans nom)')}")
            nom = input("Quell réseau voulez-vous modifier ?  ").strip()

        reseau = next((r for r in base if r.get("Nom") == nom), None)

        if reseau is None:
            print(f"Aucun réseau nommé '{nom}' trouvé.")
            return

        commentaires = reseau.get("commentaires", [])
        if not commentaires:
            print(f"Le réseau '{nom}' ne contient aucun commentaire.")
            return

        print(f"Commentaires pour '{nom}' :\n")
        print(f"{'Indice':<8} {'Date':<22} Commentaire")
        print(f"{'-'*8} {'-'*22} {'-'*50}")
        for c in commentaires:
            indice = str(c.get("indice", "—"))
            date = c.get("date", "—")
            commentaire = c.get("commentaire", "")
            print(f"{indice:<8} {date:<22} {commentaire}")

    finally:
        os.chdir(workspace_actuel)

def afficher_historique(nom):
    workspace_actuel = os.getcwd()
    try:
        chemin_base = path_nn()
        os.chdir(os.path.dirname(chemin_base))

        with open(chemin_base, 'r') as f:
            base = json.load(f)

        reseau = next((r for r in base if r.get("Nom") == nom), None)

        if reseau is None:
            print(f"Aucun réseau nommé '{nom}' trouvé.")
            return

        historique = reseau.get("historique modification", [])
        if not historique:
            print(f"Le réseau '{nom}' ne possède aucun historique.")
            return

        print(f"Historique pour '{nom}' :\n")
        print(f"{'Version':<8} {'Date':<22} {'Score':<10} {'Entrainement':<14} Lien autosave")
        print(f"{'-'*8} {'-'*22} {'-'*10} {'-'*14} {'-'*30}")
        for h in historique:
            version = str(h.get("version", "—"))
            date = h.get("date", "—")
            score = h.get("nouveau score", "—")
            entrainement = h.get("Entrainement", "—")
            lien = h.get("lien autosave", "—")
            print(f"{version:<8} {date:<22} {score!s:<10} {entrainement!s:<14} {lien}")

    finally:
        os.chdir(workspace_actuel)

def ajouter_commentaire(nom=None, commentaire=None):
    workspace_actuel = os.getcwd()
    try:
        chemin_base = path_nn()
        os.chdir(os.path.dirname(chemin_base))

        with open(chemin_base, 'r') as f:
            base = json.load(f)

        # Sélection du nom si nécessaire
        if nom is None:
            print("Réseaux disponibles :")
            for reseau in base:
                print(f"• {reseau.get('Nom', '(sans nom)')}")
            nom = input("Nom du réseau auquel ajouter un commentaire : ").strip()

        reseau = next((r for r in base if r.get("Nom") == nom), None)
        if reseau is None:
            print(f"Aucun réseau nommé '{nom}' trouvé.")
            return

        if "commentaires" not in reseau:
            reseau["commentaires"] = []

        commentaires = reseau["commentaires"]

        # Affichage des commentaires existants
        if commentaires:
            print(f"\nCommentaires existants pour le réseau '{nom}' :")
            for c in commentaires:
                print(f"[{c['indice']}] {c['date']} — {c['commentaire']}")
        else:
            print(f"\nAucun commentaire existant pour le réseau '{nom}'.")

        # Demande du nouveau commentaire si nécessaire
        if commentaire is None:
            commentaire = input("\nNouveau commentaire : ").strip()
            if commentaire == '':
                print("Commentaire vide, opération annulée.")
                return

        # Calcul de l'indice
        nouvel_indice = max((c.get('indice', -1) for c in commentaires), default=-1) + 1

        nouveau_commentaire = {
            'date': datetime.now().strftime('%d-%m-%Y, %H:%M:%S'),
            'indice': nouvel_indice,
            'commentaire': commentaire
        }

        commentaires.append(nouveau_commentaire)

        with open(chemin_base, 'w') as f:
            json.dump(base, f, indent=2)

        print(f"\n✅ Commentaire ajouté au réseau '{nom}' avec l’indice {nouvel_indice}.")

    finally:
        os.chdir(workspace_actuel)

def supprimer_commentaire(nom=None, indice=None):
    workspace_actuel = os.getcwd()
    try:
        chemin_base = path_nn()
        os.chdir(os.path.dirname(chemin_base))

        with open(chemin_base, 'r') as f:
            base = json.load(f)

        # Si nom est None, afficher les réseaux disponibles
        if nom is None:
            print("Réseaux disponibles :")
            for r in base:
                print(f"• {r.get('Nom', '(sans nom)')}")
            nom = input("Nom du réseau dont vous voulez supprimer un commentaire : ").strip()

        # Recherche du réseau
        reseau = next((r for r in base if r.get("Nom") == nom), None)
        if reseau is None:
            print(f"Aucun réseau nommé '{nom}' trouvé.")
            return

        commentaires = reseau.get("commentaires", [])
        if not commentaires:
            print(f"Le réseau '{nom}' ne possède aucun commentaire.")
            return

        # Si indice est None, afficher les commentaires et demander à choisir
        if indice is None:
            print(f"\nCommentaires pour '{nom}' :")
            print(f"{'Indice':<8} {'Date':<22} Commentaire")
            print(f"{'-'*8} {'-'*22} {'-'*50}")
            for c in commentaires:
                print(f"{c.get('indice', '—'):<8} {c.get('date', '—'):<22} {c.get('commentaire', '')}")
            try:
                indice = int(input("\nIndice du commentaire à supprimer : "))
            except ValueError:
                print("Indice invalide.")
                return

        # Suppression du commentaire avec l'indice exact
        nouveaux_commentaires = [c for c in commentaires if c.get('indice') != indice]
        if len(nouveaux_commentaires) == len(commentaires):
            print(f"Aucun commentaire avec l’indice {indice} n’a été trouvé.")
            return

        reseau['commentaires'] = nouveaux_commentaires

        with open(chemin_base, 'w') as f:
            json.dump(base, f, indent=2)

        print(f"Commentaire d’indice {indice} supprimé pour le réseau '{nom}'.")

    finally:
        os.chdir(workspace_actuel)
