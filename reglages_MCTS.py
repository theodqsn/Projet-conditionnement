from Tout import *
from math import * 

reglages_MCTS = {
### Début

### Fin
}












reglages_par_defaut = {


### reglages pour la fonction de minimisation locale 
'reglages_inch' :  {'fonction pertes inch': perte_inch_lisse_v2},

### longueur du déphasage
'decalage dephasage' : 'auto',

### liste des types d'actions possibles
'types actions possibles' : ['dephaser vertical', 'dephaser horizontal', 'tourner groupe'],

### liste des types d'orientations possibles
'orientations possibles' : [pi/4, 3*pi/4, 5*pi/4, 7*pi/4]


}
