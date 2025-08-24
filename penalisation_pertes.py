import numpy as np
import math
from math import *
from outils_pertes import *

def opp_log(x, reglages = None) :
  reglages_par_defaut = {'derivee_au_bord': 1, 'scale': 1}
  if reglages is None :
    reglages = reglages_par_defaut
  else :
    reglages = reglages_par_defaut | reglages
  db = reglages['derivee_au_bord']
  s = reglages['scale']
  return( -db*log(x/s+0.01 ))

def carre(x, reglages=None):
  return(x**2)

def inv_exp(x, reglages=None):
  x=x+0.1
  return (1/(1-exp(-1)) * exp(-x**0.5) + 1-(1/(1-exp(-1))))

def opp_gauss(x, reglages=None):
  reglages_par_defaut = {'portee' : 0.1}
  if reglages is None :
    reglages = reglages_par_defaut
  else :
    reglages = reglages_par_defaut | reglages

  portee = reglages['portee']
  return (scipy.stats.norm.pdf(0, 0, sigma)-scipy.stats.norm.pdf(x, 0, sigma))

def rac_trans(x, reglages = None):
  reglages_par_defaut = {'sigma rac_trans' : 0.1}
  if reglages is None :
    reglages = reglages_par_defaut
  else :
    reglages = reglages_par_defaut | reglages
  sigma = reglages['sigma rac_trans']
  return (sqrt(abs(x))/(0.1*sigma+sqrt(abs(x)))-1)

def creux_quadra (x, reglages=None): #si je sais bien calculer, le meilleur creux pour une descente de gradient à pas constant. En plus, c'est Cînfini
  reglages_par_defaut = {'portee' : 0.1}
  if reglages is None :
    reglages = reglages_par_defaut
  else :
    reglages = reglages_par_defaut | reglages

  portee = reglages['portee']

  return((partition_unite(x, -2*portee, -portee, portee, 2*portee)**2)*((x/portee )**2 -1) )

def exponentielle_adaptee ( x, reglages = None):
  reglages_par_defaut = {'scale': 1 ,  'derivee_au_bord': 1}
  if reglages == None :
    reglages = reglages_par_defaut
  else :
    reglages = reglages_par_defaut | reglages
  db = reglages['derivee_au_bord']
  s = reglages['scale']
  return( db*exp(x/s))

def id(x):
  return(x)

def exp_v3(x, reglages = None):
  reglages_par_defaut = {'derivee_au_bord': 1, 'scale': 1}
  if reglages is None :
    reglages = reglages_par_defaut
  else :
    reglages = reglages_par_defaut | reglages
  db = reglages['derivee_au_bord']
  s = reglages['scale']
  x=x/s
  if x <= - 0.5:
    return 0
  if x >5 : 
    return(db*exp(-(1/(5+2))**2 + 5)*e/2 +x)
  else :
    return db*exp(-(1/(x+2))**2 + x)*e/2
  
def exp_v4(x, reglages = None):
  reglages_par_defaut = {'derivee_au_bord': 1, 'scale': 1}
  if reglages is None :
    reglages = reglages_par_defaut
  else :
    reglages = reglages_par_defaut | reglages
  db = reglages['derivee_au_bord']
  s = reglages['scale']
  x=x/s
  if x <= - 0.01:
    return 0
  if x >5 : 
    return(db*exp(-(1/(5+2))**2 + 5)*e/2 +x)
  else :
    return db*exp(-(1/(x+0.01))**2 + x)*e/(1+0.01)

def carre_trans(x, reglages = None):
  reglages_par_defaut = {'scale' :1, 'derivee au bord': 1, 'portee penalite' : 1}
  if reglages is None :
    reglages = reglages_par_defaut
  else :
    reglages = reglages_par_defaut | reglages
  db = reglages['derivee au bord']
  u = x/reglages['scale']
  p = reglages['portee penalite']
  if u< - p :
    return 0
  else :  
    return db*(u+p)^2/(2*p)

def carctan(x, reglages = None):
  reglages_par_defaut = {'scale' :1, 'derivee au bord': 1, 'portee penalite' : 0, 'portee penalite' : 1}
  if reglages is None :
    reglages = reglages_par_defaut
  else :
    reglages = reglages_par_defaut | reglages
  portee = reglages['portee penalite']
  db = 1
  u = (x/reglages['scale'])
  v = u *5/portee
  if u <= 0 :
    return db*(atan(v) )
  else :
    return db*(atan(v)) +v**2

def exp_neg(x, reglages=None):
    reglages_par_defaut = {'derivee_au_bord': 1, 'scale': 1}
    if reglages is None :
        reglages = reglages_par_defaut
    else :
        reglages = reglages_par_defaut | reglages
    ord_or = math.e * reglages['scale'] * reglages['derivee_au_bord']
    u = x / reglages['scale']
    return ord_or * np.exp(-u)

def carre_tronque(x, reglages=None):
  reglages_par_defaut = {'scale' :1, 'derivee au bord': 1}
  if reglages is None :
    reglages = reglages_par_defaut
  else :
    reglages = reglages_par_defaut | reglages
  db = reglages['derivee au bord']
  if x <= 0 :
    return 0
  else :
    return 0.5*db*x**2



























