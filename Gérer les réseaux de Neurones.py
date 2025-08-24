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
# MAGIC ### Préparation des widgets

# COMMAND ----------

dbutils.widgets.text('path vers la base de données des réseaux de neurones (faire "Copy URL/path -> full path)', "")
path  = dbutils.widgets.get('path vers la base de données des réseaux de neurones (faire "Copy URL/path -> full path)')
if path in ("None", "", None):
    path = None
# On accède aux réseaux
acceder_reseaux(path)

# COMMAND ----------

dico_fonctions = {


    "Afficher un réseau": {'fonction': afficher_reseaux, 'arguments': 
      [
      {'nom': 'liste', 'type': list, 'valeur par defaut': None , 'nom_display': 'Liste des réseaux à afficher'}
      ]},
    
    
    "Afficher un réseau (détails techniques)": {'fonction': afficher_reseaux_technique, 'arguments': 
      [
        {'nom': 'liste', 'type': list, 'valeur par defaut': None , 'nom_display': 'Liste des réseaux à afficher'}
        ]},
    

    "Supprimer un réseau": {'fonction': supprimer_reseau , 'arguments': 
      [
        {'nom': 'nom', 'type': str , 'valeur par defaut': None, 'nom_display': 'Nom du réseau à supprimer'},
        {'nom': 'conf', 'type': bool , 'valeur par defaut': False, 'nom_display': 'Confirmation de la suppression ?'}
       ]},
    

    "Afficher les commentaires": {'fonction': afficher_commentaires, 'arguments': 
      [
        {'nom': 'nom', 'type': str, 'valeur par defaut': None, 'nom_display': 'Nom du réseau dont afficher les commentaires'}
       ]},
    

    "Afficher l'historique": {'fonction': afficher_historique , 'arguments':
       [
         {'nom':'nom', 'type': str, 'valeur par defaut': None, 'nom_display': 'Nom du réseau dont afficher l\'historique'}
         ]},
    

    "Ajouter un commentaire": {'fonction': ajouter_commentaire , 'arguments':
      [
        {'nom':'nom', 'type': str, 'valeur par defaut': None, 'nom_display': 'Nom du réseau auquel ajouter un commentaire'},
        {'nom':'commentaire', 'type': str, 'valeur par defaut': None, 'nom_display': 'Commentaire'}
        ]},
    

    "Supprimer un commentaire": {'fonction': supprimer_commentaire, 'arguments': 
      [
        {'nom':'nom', 'type': str, 'valeur par defaut': None, 'nom_display': 'Nom du réseau auquel ajouter un commentaire'},
        {'nom': 'indice', 'type': int , 'valeur par defaut': None , 'nom_display': 'Indice du commentaire à supprimer (⚠️commence à 0)'}
        ]},
}


# COMMAND ----------

  def interface():
    import ipywidgets as widgets
    from IPython.display import display, clear_output

    # Menu déroulant pour sélectionner la fonction
    dropdown = widgets.Dropdown(
        options=list(dico_fonctions.keys()),
        description='Fonction :',
        style={'description_width': 'initial'}
    )

    # Bouton d'exécution
    button = widgets.Button(
        description="Exécuter",
        button_style='success'
    )

    # Zone de sortie
    out = widgets.Output(layout={'width': '100%'})

    # Boîte pour les widgets d'arguments
    args_box = widgets.VBox([])

    # Dictionnaire pour garder les widgets associés aux arguments
    widgets_arguments = {}

    # Fonction pour créer un widget selon le type
    def creer_widget(arg):
        arg_type = arg['type']
        valeur_defaut = arg.get('valeur par defaut')
        label = arg.get('nom_display', arg['nom'])
        
        if arg_type == int:
            w = widgets.IntText(value=valeur_defaut if valeur_defaut is not None else 0,
                                layout=widgets.Layout(width="50%"))
        elif arg_type == float:
            w = widgets.FloatText(value=valeur_defaut if valeur_defaut is not None else 0.0,
                                layout=widgets.Layout(width="50%"))
        elif arg_type == str:
            w = widgets.Text(value=valeur_defaut if valeur_defaut is not None else "",
                            layout=widgets.Layout(width="50%"))
        elif arg_type == bool:
            w = widgets.Checkbox(value=valeur_defaut if valeur_defaut is not None else False,
                                layout=widgets.Layout(width="50%"))
        else:
            w = widgets.Text(value=str(valeur_defaut) if valeur_defaut is not None else "",
                            layout=widgets.Layout(width="50%"))
        
        w.description = label 
        w.style = {'description_width': 'auto'}  # largeur de l’étiquette
        return w

    # Callback pour changer les widgets d'arguments quand la fonction change
    def on_dropdown_change(change):
        clear_output(wait=True)
        nom_fct = change['new']
        fct_info = dico_fonctions[nom_fct]
        args_widgets = []
        widgets_arguments.clear()
        
        for arg in fct_info.get('arguments', []):
            w = creer_widget(arg)
            args_widgets.append(w)
            widgets_arguments[arg['nom']] = w  # on garde toujours 'nom' comme clé pour appeler la fonction
        
        args_box.children = args_widgets

    dropdown.observe(on_dropdown_change, names='value')

    # Callback du bouton pour exécuter la fonction avec les arguments
    def on_button_click(b):
        nom_fct = dropdown.value
        fct_info = dico_fonctions[nom_fct]
        fct = fct_info['fonction']
        
        # Récupérer les valeurs des widgets avec conversion "" -> None
        kwargs = {}
        for nom, w in widgets_arguments.items():
            val = w.value
            if isinstance(w, widgets.Text):   # champ texte vide
                if val == "":
                    val = None
            elif isinstance(w, (widgets.IntText, widgets.FloatText)):
                # Si l'utilisateur a effacé, ipywidgets force à 0 ou 0.0
                # -> on teste la chaîne dans le DOM
                raw_val = w.get_state().get("value", None)
                if raw_val in ("", None):
                    val = None
            # pour Checkbox, on garde True/False
            kwargs[nom] = val
        
        with out:
            clear_output(wait=True)
            fct(**kwargs)

    button.on_click(on_button_click)

    # Affichage final
    display(dropdown, args_box, button, out)

    # Initialiser les widgets pour la première fonction sélectionnée
    on_dropdown_change({'new': dropdown.value})

# COMMAND ----------

# MAGIC %md
# MAGIC ## Interface

# COMMAND ----------

interface()