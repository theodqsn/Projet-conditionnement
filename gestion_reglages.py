import importlib
import reglages_bibos
import os
import importlib.util
import shutil



'🐢'

'👇👇👇👇👇👇👇👇👇👇👇👇'
path_vers_reglages_bibos = '/Workspace/Users/theo.duquesne@armor-iimak.com/empilements des cylindres2/modules'
'☝️☝️☝️☝️☝️☝️☝️☝️☝️☝️☝️☝️☝️☝️'


'👇👇👇👇👇👇👇👇👇👇👇👇'
path_vers_reglages_MCTS = '/Workspace/Users/theo.duquesne@armor-iimak.com/empilements des cylindres2/modules'
'☝️☝️☝️☝️☝️☝️☝️☝️☝️☝️☝️☝️☝️☝️'



def reglages(nom_du_reglage):
    current_wd = os.getcwd()
    try:
        os.chdir(path_vers_reglages_bibos)

        # Création du cache si besoin
        if not hasattr(reglages, 'dico_reglages'):
            reglages.dico_reglages = {}

        # Si déjà en cache
        if nom_du_reglage in reglages.dico_reglages:
            return reglages.dico_reglages[nom_du_reglage]

        # Vérifie l’existence du fichier reglages_bibos.py
        chemin_reglages = os.path.join(path_vers_reglages_bibos, 'reglages_bibos.py')
        if not os.path.exists(chemin_reglages):
            print('🐢🐢🐢')
            print(f"Le fichier 'reglages_bibos.py' est introuvable dans '{path_vers_reglages_bibos}'.")
            print('SVP suivez les indications suivantes :')
            print('  - 🐢 Identifiez l\'endroit où est enregistré le fichier reglages_bibos.py' )
            print('  - 🐢 Copiez le chemin vers le fichier reglages_bibos.py')
            print('  - 🐢 Identifiez l\' endroit où est enregistré le fichier gestion_reglages.py')
            print('  - 🐢 Collez le chemin dans \"path_vers_reglages_bibos\" ')
            raise FileNotFoundError(f"Suivez la sagesse des tortues")

        # Import dynamique du module reglages_bibos
        spec = importlib.util.spec_from_file_location("reglages_bibos", chemin_reglages)
        if spec is None or spec.loader is None:
            raise ImportError("Impossible de charger le module 'reglages_bibos'.")

        reglages_bibos = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(reglages_bibos)

        # Lecture des réglages
        if nom_du_reglage in reglages_bibos.reglages_nn:
            valeur = reglages_bibos.reglages_nn[nom_du_reglage]
        else:
            valeur = reglages_bibos.reglages_par_defaut[nom_du_reglage]

        # Mise en cache
        reglages.dico_reglages[nom_du_reglage] = valeur
        return valeur

    finally:
        os.chdir(current_wd)

def reglages_internes(nom_du_reglage, path_vers_reglages_reseau= None):
    current_wd = os.getcwd()
    if  path_vers_reglages_reseau is None and hasattr(reglages_internes, 'path_regint'):
        path_vers_reglages_reseau = reglages_internes.path_regint
    else : 
        reglages_internes.path_regint = path_vers_reglages_reseau

    try : 

      # Création du cache si besoin
      if not hasattr(reglages_internes, 'dico_reglages'):
              reglages_internes.dico_reglages = {}
    
      # Si déjà en cache
      if nom_du_reglage in reglages_internes.dico_reglages:
          return reglages_internes.dico_reglages[nom_du_reglage]

      # Vérifie l’existence du fichier r
      if not os.path.exists(path_vers_reglages_reseau):
          print('🐢🐢🐢')
          print('File not Found : reglages du rdn')

          raise FileNotFoundError(f"Probablement un rdn trop ancien, avant la maj")

      # Import dynamique du module reglages_bibos
      spec = importlib.util.spec_from_file_location("reglages_int", path_vers_reglages_reseau)
      if spec is None or spec.loader is None:
          raise ImportError("Impossible de charger le module 'reglages_int'.")

      reglages_int = importlib.util.module_from_spec(spec)
      spec.loader.exec_module(reglages_int)

      # Lecture des réglages
      if nom_du_reglage in reglages_int.reglages_nn:
          valeur = reglages_int.reglages_nn[nom_du_reglage]
      else:
          valeur = reglages_int.reglages_par_defaut[nom_du_reglage]

      # Mise en cache
      reglages_internes.dico_reglages[nom_du_reglage] = valeur
      return valeur

    finally:
        os.chdir(current_wd)

def reglages_MCTS(nom_du_reglage):
    current_wd = os.getcwd()
    try:
        os.chdir(path_vers_reglages_MCTS)

        # Création du cache si besoin
        if not hasattr(reglages_MCTS, 'dico_reglages'):
            reglages_MCTS.dico_reglages = {}

        # Si déjà en cache
        if nom_du_reglage in reglages_MCTS.dico_reglages:
            return reglages_MCTS.dico_reglages[nom_du_reglage]

        # Vérifie l’existence du fichier reglages_bibos.py
        chemin_reglages = os.path.join(path_vers_reglages_MCTS, 'reglages_MCTS.py')
        if not os.path.exists(chemin_reglages):
            print('🐢🐢🐢')
            print(f"Le fichier 'reglages_MCTS.py' est introuvable dans '{path_vers_reglages_MCTS}'.")
            print('SVP suivez les indications suivantes :')
            print('  - 🐢 Identifiez l\'endroit où est enregistré le fichier reglages_MCTS.py' )
            print('  - 🐢 Copiez le chemin vers le fichier reglages_MCTS.py')
            print('  - 🐢 Identifiez l\' endroit où est enregistré le fichier gestion_reglages.py')
            print('  - 🐢 Collez le chemin dans \"path_vers_reglages_MCTS\" ')
            raise FileNotFoundError(f"Suivez la sagesse des tortues")

        # Import dynamique du module reglages_MCTS
        spec = importlib.util.spec_from_file_location("reglages_MCTS", chemin_reglages)
        if spec is None or spec.loader is None:
            raise ImportError("Impossible de charger le module 'reglages_MCTS'.")

        module_reglages_MCTS = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module_reglages_MCTS)

        # Lecture des réglages
        if nom_du_reglage in module_reglages_MCTS.reglages_MCTS:
            valeur = module_reglages_MCTS.reglages_MCTS[nom_du_reglage]
        else:
            valeur = module_reglages_MCTS.reglages_par_defaut[nom_du_reglage]

        # Mise en cache
        reglages_MCTS.dico_reglages[nom_du_reglage] = valeur
        return valeur

    finally:
        os.chdir(current_wd)

def reinit_reglages_MCTS():
    if hasattr(reglages_MCTS, 'dico_reglages'):
        delattr(reglages_MCTS, 'dico_reglages')

def reinit_reglages_nn():
    if hasattr(reglages_internes, 'dico_reglages'):
        delattr(reglages_internes, 'dico_reglages')

    if hasattr(reglages, 'dico_reglages'):
        delattr(reglages, 'dico_reglages')
