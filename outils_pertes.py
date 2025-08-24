import numpy as np
import math
from math import *

def partition_unite(x, d=-2, a=-1, b=1, f=2):
  if x>=a and x<=b :
    return(1)
  if x<=d or x>=f :
    return(0)
  if x>d and x<a:
    return( (exp(-  (1/((x-d)**2))  )) * (1-(exp(-  (1/((x-a)**2))  )))*(exp(-  (1/((a-d)**2))  ))  )
  if x>b and x<f:
    return( (exp(-  (1/((x-f)**2))  )) * (1-(exp(-  (1/((x-b)**2))  )))*(exp(-  (1/((b-f)**2))  ))  )
  
def rec_noyaux(X, poids, reglages = None): #on effectue une reconstruction à noyeaux de la distribution des angles, pondérée par e. On pourra ensuite normaliser, effectuer une convolution avec un pic et utiliser une norme p avec p>= 1. A priori, plus p est élevé, plus le fait que les pics soient hauts sera favorisé
  n= np.shape(X)[0]
  poids = poids/np.sum(poids)
  reglages_defaut = {'portee': 0.05, 'nombre de points abscisse' : 1000, 'nombre de tours' : 1}
  if reglages is not None:
    reglages = reglages_defaut | reglages
  else:
    reglages = reglages_defaut
  portee = reglages['portee']
  n_abs = reglages['nombre de points abscisse']
  nombre_tours = reglages['nombre de tours'] #ça servira jamais, mais permet de savoir à quel point on pousse la précision pour la reconstruction à noyaux. Par defaut 1
  #abs = np.linspace(0, 2*pi, n_abs) #on a l'air cons, ça va être facile de faire de la reconstruction par noyaux sur [0, 2pi]/(0~2pi) tiens. On s'en sortira hein
  Y = np.zeros(n_abs)
  for i in range(n):
    or_i = int(X[i]/(2*np.pi) * n_abs) #le centre de la gaussienne générée par le i_eme point
    for j in range(-n_abs*nombre_tours//2,n_abs*nombre_tours//2 ):
      abs = X[i] + j/n_abs*2*pi # l'abscisse du j-ème point dans \mathbb(R)
      abs_red = (or_i +j)%n_abs # la position dans  np.linspace(0, 2*pi, n_abs) du j ème point
      valeur = 1/n * scipy.stats.norm.pdf(abs, loc = X[i], scale = portee )* poids[i]

      Y[abs_red] = Y[abs_red] + valeur
  return(Y)

def convol(Y, reglages=None):
  reglages_defaut={'bande passante' : 0.1, 'p' : 5} #plus p est grand, plus la fonction sera élevée lorsque les points sont concentrés
  if reglages is not None:
    reglages = reglages_defaut | reglages
  else:
    reglages = reglages_defaut

  n_abs = np.shape(Y)[0]
  portee = int(reglages['bande passante']*n_abs)
  p= reglages['p']
  hauteur = 1/(portee * n_abs)*2*np.pi
  s= 0
  for i in range(n_abs):
    s_i= 0
    for j in range(-portee//2+i, i+portee//2):
      s_i += Y[j % n_abs]*hauteur
    s+= (s_i)**p
  return(s**(1/p))

def recollement_continu (x, a=1, h= 0.001) :
  p=0.01
  if x<=a:
    return(0)
  elif x>=a+h :
    return(1)
  else :
    return( exp(- 1/(x-a)**p)*exp(1/h**p))

def prol(marge, a=1, h=0.001):
  t_moins = 0
  t_plus = 0
  if marge <= 1+h:
    t_moins = exp(-marge)*exp(1)
  if marge >= 1 - h :
    t_plus = (2/(2+h))
  if marge >= 1 +h :
    t_plus = 2/(1+marge)
  
  prol.t_plus = t_plus
  prol.t_moins = t_moins
  total = recollement_continu(marge, a, h) * t_plus + (1-recollement_continu(marge, a, h)) * t_moins
  return(total)

def projeter_sur_plan(point, axe):
    axes_du_plan = []

    iterateur  = np.eye(3)
    for e in iterateur :
        if np.array_equal(e, axe) == False:
            axes_du_plan.append(e)

    projete = np.zeros(2)
    projete[0] = np.dot(point, axes_du_plan[0])
    projete[1] = np.dot(point, axes_du_plan[1])
    return projete

def calculs_spatiaux(centre_i, centre_j, axe):
  centre_i_projete = projeter_sur_plan(centre_i, axe)
  centre_j_projete = projeter_sur_plan(centre_j, axe)
  y = np.abs(centre_j_projete[1] - centre_i_projete[1])
  x = np.abs(centre_j_projete[0] - centre_i_projete[0])
  angle = atan(y/x)
  distance = np.sqrt(y**2 + x**2)
  return distance, angle

def ajout_angle(angles_ponderations, angle, pond):
    if angle in angles_ponderations:
        angles_ponderations[angle][1] += pond
    else:
        angles_ponderations.append([angle, pond])
    return angles_ponderations

def distance_modulo(p1, p2):
    """Calcule la distance entre p1 et p2 modulo π/2."""
    mod = math.pi / 2
    diff = abs(p1 - p2) % mod
    return min(diff, mod - diff)

def distance_modulo_pi_4(a, b):
    """Distance angulaire modulo pi/4."""
    pi_4 = np.pi / 4
    diff = abs(a - b) % pi_4
    return min(diff, pi_4 - diff)

def mesure_dispersion(angles_ponderations, reglages = None):
    pi_4 = np.pi / 4
    # Extraction des angles modulo pi/4
    angles = [a % pi_4 for a, _ in angles_ponderations]
    poids = [p for _,p in angles_ponderations]
    # Tri
    angles.sort()
    n = len(angles)
    s = 0
    for i in range(n):
        a1 = angles[i]
        a2 = angles[(i+1) % n]  # suivant, avec boucle circulaire
        dist = distance_modulo_pi_4(a1, a2)
        a = 0
        if dist <= 0.05 :
          a = poids[i]*dist/0.05 #(dist/0.05) ** 0.5
        else :
          a = poids[i]
        s += a
    return(s)






