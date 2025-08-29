import numpy as np
import pandas as pd
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
from datetime import timedelta
from scipy.stats import ks_2samp
from datetime import datetime
from pyspark.sql import functions as F
from pyspark.sql import SparkSession
spark = SparkSession.builder.getOrCreate()

lien_casse_film = '/Workspace/Users/theo.duquesne@armor-iimak.com/analyse des donnees/dossier_casse_film'

def conversion_datetime(date_mauvais_format):
  nouvelle_date = ''
  date_mauvais_format = str(date_mauvais_format)
  for c in date_mauvais_format:
    if c==' ':
      nouvelle_date += 'T'
    else:
      nouvelle_date += c
  return(nouvelle_date)

def extraction_date(debut, fin, data_frame):
  return(data_frame[(data_frame['date']>=debut) & (data_frame['date']<=fin)])

def to_date( heures , minutes, secondes= 0, jour = 17, mois = 7, annee = 2025):
  return(pd.to_datetime(pd.Timestamp(year=annee, month=mois, day=jour, hour=heures, minute=minutes, second=secondes)))

def renommer_colonne(df, printer = True):
  flag = False
  if 'date' not in df.columns:
    for col in df.columns:
        if pd.api.types.is_datetime64_any_dtype(df[col]):
            if col != 'date':
                if printer:
                  print(f"⚠️  Colonne '{col}' de type timestamp renommée en 'date'.")
                  flag = True
                df.rename(columns={col: 'date'}, inplace=True)
            break  # on s'arrête après la première colonne timestamp trouvée
  renommer_colonne.printe = flag #pour ne printer qu'une seule fois le message d'erreur
  return df

def detecter_casse_film_v0(df, debut=None, fin=None, date_officielle=None):
  if debut is None:
    debut = df['date'].min()
  if fin is None:
    fin = df['date'].max()
    
    dft = extraction_date(debut, fin, df)
    dft = dft.sort_values(by='date', ascending=True)

    # Choix de la colonne contenant la tension
    if 'M : tension dérouleur (N)' in dft.columns:
        valeurs = dft['M : tension dérouleur (N)']
    else: 
        flag = False
        for cle in dft.columns:
            if 'tension' in cle.lower():
                print(f"Utilisation de la colonne '{cle}' pour detecter le casse-film")
                valeurs = dft[cle]
                flag = True
                break
        
        if not flag:
            for cle in dft.columns:
                if not isinstance(dft[cle].iloc[0], pd.Timestamp):
                    print(f"Utilisation de la colonne '{cle}' pour detecter le casse-film")
                    valeurs = dft[cle]
                    break

    # Calcul des différences
    der = np.diff(valeurs)
    max_der = np.max(np.abs(der))
    ind_extremes = np.where(np.abs(der) >= max_der/10)[0]
    print('ind_extremes', ind_extremes)

    if date_officielle is None:
        date_officielle = fin

    coupure = None
    for i in ind_extremes:
        if dft['date'].iloc[i] <= date_officielle and float(der[i]) <= 0:
            coupure = i

    detecter_casse_film.colonne_choisie = cle
    if coupure is not None:
        return dft['date'].iloc[coupure]
    else:
        return None
      

def extraire_table(colonnes: list[str],timestamp_debut: datetime, timestamp_fin: datetime,  table: str = "hive_metastore.agg_hist.dcm42_all"
) -> pd.DataFrame:
    """
    Extrait certaines colonnes de la table entre deux timestamps inclus.
    
    colonnes : liste de noms de colonnes à récupérer
    timestamp_debut, timestamp_fin : objets datetime.datetime
    table : chemin complet de la table dans le metastore
    """
    # Vérification du type des paramètres
    if not isinstance(timestamp_debut, datetime) or not isinstance(timestamp_fin, datetime):
        raise TypeError("Les paramètres timestamp_debut et timestamp_fin doivent être des datetime.datetime")

    # Ajout de la colonne time si elle n'est pas déjà demandée
    if "time" not in colonnes:
        colonnes = ["time"] + colonnes

    # Conversion en string ISO pour Spark
    ts_debut_str = timestamp_debut.strftime("%Y-%m-%d %H:%M:%S")
    ts_fin_str = timestamp_fin.strftime("%Y-%m-%d %H:%M:%S")

    # Lecture depuis Spark
    df_spark = (
        spark.table(table)
        .filter(
            (F.col("time").cast("timestamp") >= F.to_timestamp(F.lit(ts_debut_str))) &
            (F.col("time").cast("timestamp") <= F.to_timestamp(F.lit(ts_fin_str)))
        )
        .select(*colonnes)
    )
    
    # Conversion en pandas
    df_pandas = df_spark.toPandas()
    for col in df_pandas.columns:
        if pd.api.types.is_datetime64_any_dtype(df_pandas[col]):
            df_pandas.rename(columns={col: "date"}, inplace=True)
            break

    return df_pandas
