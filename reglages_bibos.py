from minimiseur_bibos import *

reglages_nn = {'n_tours': 2, 'nombre de rouleaux': 10, 'taille des batchs': 32, 'nombre de couches': 150}
reglages_par_defaut = {
 # Paramètres d'entrée :
  'nombre de rouleaux' : 2,

 # RÉGLAGES INTERNES AU RÉSEAU DE NEURONES : 

  'nombre de couches': 15, 
  # Plus c'est gros, plus on pourra faire avoir des résultats précis mais plus il faudra entraîner

  'n_tours' : 100,

  'nombre explorateurs' : 1, 
  #Plus c'est grand, plus on explore l'espace mais plus on doit faire de calculs

  'taille des batchs' : 32, 
  # Plus c'est grand, meilleure sera l'idée de la direction à explorer mais plus grand sera le temps de calcul

  # 🌼🌼🌼 Nombre d'échantillons par entrées
  'echantillons par entree' : 4,
  # à augmenter si le réseau semble avoir du mal à comprendre dans quelle direction évoluer. Plus c'est grand, plus c'est lent 

  'taille couches intermediaires' : 64, 
  # Plus c'est gros, plus on pourra faire avoir des résultats précis mais plus il faudra entraîner

  'portee explorateurs' : 0.05, 
  #Distance en dessous de laquelle les explorateurs sont considérés trop proches

  'reactivite prudence': 1/3 , 
  #proche de 1 => très réactif, proche de 0 => très lent à réagir pour ajuster la variance exploratoire en fonction de la variance observée des récompenses 

  "seuil prudence" : 0.5, 
  # variation en ordre de grandeur accepté sur la récompense lors du tirage depuis un même mode

  'variance initiale':0.01, 
  # Vitesse d'exploration initiale

  'seuil exploration minimal' :1/3, 
  # On dédie cette proportion des calculs aux explorateurs qui ont encore rien découvert d'interressant 

  # 🌼🌼🌼 Variance d'exploration initiale lorsque l'on fait de l'annealing
  'variance annealing debut' : 0.01,
  # Si la stratégie semble converger trop lentement (ie que la récompense est croissante et ne semble pas stagner après beaucoup d'itérations), on peut tenter d'augmenter. Si n'arrive pas à avoir une récompense qui augmente, on peut tenter de diminuer

  # 🌼🌼🌼 Variance d'exploration finale lorsque l'on fait de l'annealing
  'variance annealing fin' : 0.0001,
  #Si la récompense oscille lorsqu'elle est stabilisée, c'est que la variance est trop grande. On peut alors tenter de diminuer cette valeur. Si la stratégie semble "bloquée" sur la fin (dérivée de la récompense par rapport à la taille du pas est non presque nulle, mais la stratégie bouge peu), alors on peut tenter d'augmenter.

  # 🌼🌼🌼 Modification de la variance d'exploration pour réduire la variance des récompense à l'interieur d'un batch
  'Controlabilite VIB' : False,
  # Si on a un seul explorateur, est pertinent d'utilisation. Sinon, on risque d'avoir des prolèmes. Peut éventuellement être utilisé si le nombre d'explorateur <= 2* echantillons par entree. 

  'max variance exploration autorisee' : 1, 
  #Vitesse d'exploration maximale autorisée

  'max acceleration variance exploration' : 2,
   # Plus c'est grand, plus c'est instable. Plus c'est petit, plus on prendre du temps à converger alors qu'on aurait pû aller plus vite

  'min variance exploration autorisee' : 1e-8, 
  #Vitesse d'exploration minimale autorisée. Sera atteinte lorsque l'on sera dans un optimum local

  'memoire variation recompenses' : 4 ,
   #Nombre des derniers tours que l'on considère pour savoir comment diminuer le learning rate de l'optimiseur. Si trop gros, risque d'être trop lent à réagir. Si trop rapide, sera en mode Dory et oubliera qu'il doit être prudent à l'instant où le danger ne sera plus en vue (mais sera encore là)

  'acceleration max learning rate' : 1.5, 
  #Plus c'est grand, plus on risque d'être instable. Plus c'est petit, plus on prend du temps à converger alors qu'on aurait pu aller plus vite. Le diminuer permet de diminuer la memoire de la variation des recompenses.

  # 🌼🌼🌼  Learning rate modifié en forcing ?
  'Interventionnisme learning rate' : False,
  # Si la variance entre récompense est énorme d'une itération sur l'autre, que les batchs sont de taille énorme et que la récompense n'augmente pas après beaucoup d'itérations, on peut essayer de le mettre en True

  # 🌼🌼🌼  Learning rate minimal accepté
  'learning rate minimal' : 0,


  # 🌼🌼🌼 Learning rate maximal accepté
  'learning rate maximal' : 0.001,
  # Si trop élevée, on risque, les fois où la commandabilité ne suffira pas, de se déplacer très fortement dans le monde des paramètres. Si il est trop faible, on va vraiment avoir du mal à converger
  

 # 🌼🌼🌼 Différence maximale de récompense acceptée entre deux itérations successives
 'variance recompense acceptee' : 1,
 # Plus c'est  grand, plus on ira vite vers un optimum mais plus on risque d'être instable et de ne pas converger. Plus c'est petit, plus on prend du temps à converger alors qu'on aurait pu potentiellement aller plus vite. 
 # Signe qu'il faut l'abaisser : la courbe de récompenses fluctue beaucoup

  'reglages minimisation' : {
    'resol_main' : resol_a_la_main_bibos
                             },
# 🌼🌼🌼 Choisit la fonction à évaluer pour obtenir la récompense
  'fonction a evaluer' : 'version normale',






 #Sauvegarde et chargement des paramètres du reseau entraîné :
  'path vers la base de donnees des reseaux' : '/Workspace/Users/theo.duquesne@armor-iimak.com/empilements des cylindres2/bases de données/Base_Donnees_Reseaux.json',
  'path parametres defaut' : '/Workspace/Users/theo.duquesne@armor-iimak.com/empilements des cylindres2/modules/parametres_rdn/',






  # Paramètres des indicateurs

  # 🌼🌼🌼 Afficher la déviation de la stratégie ?
  'calculer deviation strategie' : False,
  # ça demande un peu de temps de calcul en plus, mais c'est un indicateur intéressant

  # 🌼🌼🌼 Afficher la vitesse de la stratégie ?
  'calculer vitesse strategie' : False
  # ça demande un peu de temps de calcul en plus, mais c'est un indicateur intéressant



}