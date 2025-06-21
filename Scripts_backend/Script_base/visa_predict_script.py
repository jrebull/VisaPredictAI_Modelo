import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import joblib
import io
import sys
import json

from sklearn.pipeline import Pipeline
from sklearn.model_selection import KFold
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import PowerTransformer
from sklearn.preprocessing import FunctionTransformer
from sklearn.preprocessing import QuantileTransformer
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OrdinalEncoder

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LinearRegression

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.metrics import recall_score, precision_score, f1_score, accuracy_score

from sklearn.dummy import DummyClassifier
from sklearn.neighbors import KNeighborsClassifier

from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_predict
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC

from tabulate import tabulate

from collections import Counter
#from imblearn.over_sampling import SMOTE
#from imblearn.over_sampling import BorderlineSMOTE
#from imblearn.combine import SMOTEENN
#from imblearn.under_sampling import EditedNearestNeighbours
#from imblearn.under_sampling import TomekLinks
#from imblearn.combine import SMOTETomek

from sklearn.inspection import permutation_importance
from sklearn.feature_selection import SelectFromModel

import pickle
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, roc_curve, auc, ConfusionMatrixDisplay
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis

# Función para cargar el modelo predictor

def carga_modelo():

    with open('pipe_qda_cwc_model.pkl', 'rb') as file:
      pipe_qda_cwc = pickle.load(file)

    return pipe_qda_cwc


# Función para recibir el dataset de prueba y transformarlo para que pueda procesarlo el modelo predictor

def transforma_set(x_data):

    # Se crear el dataframe del conjutno transformado
    x_data_T = x_data.copy()

    # Se aplica label enconder a las variable binarias
    #print("inicia aplicacion de label encoder")
    le_full_time_position = joblib.load('le_full_time_position.joblib')
    le_has_job_experience = joblib.load('le_has_job_experience.joblib')
    #le_requires_job_training = joblib.load('le_requires_job_training.joblib')

    x_data_T['has_job_experience'] = le_has_job_experience.transform(x_data_T['has_job_experience'])
    x_data_T['full_time_position'] = le_full_time_position.transform(x_data_T['full_time_position'])
    #print("finaliza aplicacion de label encoder")

    # Se aplica la transformación ordinal encoder a las variables ordinales
    #print("inicia aplicacion de ordinal encoder")
    ord_col = ['education_of_employee']
    x_data_T[ord_col] = joblib.load('ordinal_encoder.joblib').transform(x_data_T[ord_col])
    #print("finaliza aplicacion de ordinal encoder")

    # Se aplica Quantile Transformer a las variables ordinales
    #print("inicia aplicacion de QT")
    x_data_T[ord_col] = joblib.load('quantile_transformer.joblib').transform(x_data_T[ord_col])
    #print("finaliza aplicacion de QT")

    # Se aplica MinMaxScaler a las variables ordinales
    #print("inicia aplicacion de MinMaxScaler")
    x_data_T[ord_col] = joblib.load('min_max_scaler.joblib').transform(x_data_T[ord_col])
    #print("finaliza aplicacion de MinMaxScaler")

    # No se aplica la transformación get_dummies() a las variables categóricas porque cuando se envía un solo registro se eliminan las columnas
    # Mejor mediante una funcion insertamos las columnas 'VehicleCategory_Sport', 'BasePolicy_Collision'
    # y mediante una función las llenamos con su valor correspondiente
    #print("inicia aplicacion de get_dummies")
    x_data_T = encode_categorical_columns(x_data_T)
    #print("finaliza aplicacion de get_dummies")


    # Generamos un dataframe que solamente contegan las columnas de la lista 'features_list_6'

    features_list_6 = ['full_time_position',
      'region_of_employment_South',
      'continent_North America',
      'region_of_employment_Midwest',
      'continent_Europe',
      'has_job_experience',
      'education_of_employee']

    XtestT_6 = x_data_T[features_list_6]

    return XtestT_6


def encode_categorical_columns(x_test_t):
  """
    Creates a copy of the dataset, encodes 'VehicleCategory' and 'BasePolicy'
    into binary columns, and drops the original columns.

    Args:
      x_test_t: The input pandas DataFrame with 'continent' and 'region_of_employment' columns.

      continent_Asia, continent_Europe, continent_North America, continent_Oceania, continent_South America

       region_of_employment_Midwest, region_of_employment_Northeast, region_of_employment_South, region_of_employment_West

    Returns:
      A new pandas DataFrame with the encoded columns and original columns dropped.
    """
  x_test_t_copy = x_test_t.copy()

  x_test_t_copy['continent_Asia'] = 0
  x_test_t_copy['continent_Europe'] = 0
  x_test_t_copy['continent_North America'] = 0
  x_test_t_copy['continent_Oceania'] = 0
  x_test_t_copy['continent_South America'] = 0
  x_test_t_copy['region_of_employment_Midwest'] = 0
  x_test_t_copy['region_of_employment_Northeast'] = 0
  x_test_t_copy['region_of_employment_South'] = 0
  x_test_t_copy['region_of_employment_West'] = 0

  for index, row in x_test_t_copy.iterrows():
    if row['continent'] == 'Asia':
      x_test_t_copy.at[index, 'continent_Asia'] = 1
    else:
      x_test_t_copy.at[index, 'continent_Asia'] = 0

    if row['continent'] == 'Europe':
      x_test_t_copy.at[index, 'continent_Europe'] = 1
    else:
      x_test_t_copy.at[index, 'continent_Europe'] = 0

    if row['continent'] == 'North America':
      x_test_t_copy.at[index, 'continent_North America'] = 1
    else:
      x_test_t_copy.at[index, 'continent_North America'] = 0

    if row['continent'] == 'Oceania':
      x_test_t_copy.at[index, 'continent_Oceania'] = 1
    else:
      x_test_t_copy.at[index, 'continent_Oceania'] = 0

    if row['continent'] == 'South America':
      x_test_t_copy.at[index, 'continent_South America'] = 1
    else:
      x_test_t_copy.at[index, 'continent_South America'] = 0

    if row['region_of_employment'] == 'Midwest':
      x_test_t_copy.at[index, 'region_of_employment_Midwest'] = 1
    else:
      x_test_t_copy.at[index, 'region_of_employment_Midwest'] = 0

    if row['region_of_employment'] == 'Northeast':
      x_test_t_copy.at[index, 'region_of_employment_Northeast'] = 1
    else:
      x_test_t_copy.at[index, 'region_of_employment_Northeast'] = 0

    if row['region_of_employment'] == 'South':
      x_test_t_copy.at[index, 'region_of_employment_South'] = 1
    else:
      x_test_t_copy.at[index, 'region_of_employment_South'] = 0

    if row['region_of_employment'] == 'West':
      x_test_t_copy.at[index, 'region_of_employment_West'] = 1
    else:
      x_test_t_copy.at[index, 'region_of_employment_West'] = 0

  x_test_t_copy = x_test_t_copy.drop(columns=['continent', 'region_of_employment'])

  return x_test_t_copy


# Función para desplegar la matriz de confusión (etiquetas_reales, etiquetas_de_predicciones)

def mi_cm(yreal, ypred):

  cm = confusion_matrix(yreal, ypred)
  frecs = cm.flatten()

  txt = ['Verdaderos Negativos','Falsos Positivos','Falsos Negativos','Verdaderos Positivos']
  vf = [ '( VN )', '( FP )', '( FN )', '( VP )']
  frecuencia = ["{0:0.0f}".format(value) for value in cm.flatten()]
  porcentaje = ["{0:.1%}".format(value) for value in cm.flatten()/np.sum(cm)]
  porcentaje_por_categoría = [frecs[0]/(frecs[0]+frecs[1]),
                              frecs[1]/(frecs[0]+frecs[1]),
                              frecs[2]/(frecs[2]+frecs[3]),
                              frecs[3]/(frecs[2]+frecs[3])]
  porcentaje_por_categoría = ['{0:.1%}'.format(value) for value in porcentaje_por_categoría]

  labels = [f"{v1}\n{v2}\n{v3}\n{v4}\n{v5}" for v1, v2, v3, v4, v5 in zip(txt,vf, frecuencia,porcentaje,porcentaje_por_categoría)]
  labels = np.asarray(labels).reshape(2,2)

  plt.figure(figsize=(6,4))
  ax = sns.heatmap(cm, annot=labels, fmt='', cmap='Pastel1', cbar=False)
  ax.set(ylabel="Etiquetas Reales", xlabel="Etiquetas de Predicción")
  plt.show()

# Función para obtener la predicción de visa al recibir un dataset

def predice_visa(x_dataset_json):

  # Al convertir el json a dataframe, todas las columnas deben convertirse como strings para que
  # no se genere error al hacer las transformaciones

  x_dataset = json_to_dataframe(x_dataset_json)

  pipe_qda_cwc = carga_modelo()

  x_dataset_T = transforma_set(x_dataset)

  y_pred = pipe_qda_cwc.predict(x_dataset_T)

  return y_pred

# Función auxiliar para descargar datasets

def download_dataframe_as_csv(df, filename="output.csv"):
    """Converts a pandas DataFrame to CSV and initiates download."""
    #df.to_csv(filename, index=False)
    #files.download(filename)

    df.to_csv('' + filename, index=False)

def dataframe_to_json(df):
    return df.to_json(orient='records')

def json_to_dataframe(json_df):
    # Check if the input is bytes and decode if necessary
    if isinstance(json_df, bytes):
        json_df = json_df.decode('utf-8')
    # Now json_df is guaranteed to be a string
    df = pd.read_json(io.StringIO(json_df), orient='records')
    df = df.astype(str)
    return df

def json_to_dataframe_with_object_types(json_df):
    # Check if the input is bytes and decode if necessary
    if isinstance(json_df, bytes):
        json_df = json_df.decode('utf-8')

    # Read the JSON into a DataFrame
    df = pd.read_json(io.StringIO(json_df), orient='records')

    # Convert all columns to 'object' dtype
    for col in df.columns:
        df[col] = df[col].astype('object')

    return df

def main_visa_prediction(json_txt):

  y_pred = predice_visa(json_txt)

  preds_list = []

  for pred in y_pred:
    if pred == 0:
      preds_list.append("Certified")
    else:
      preds_list.append("Denied")

  return preds_list

def obten_texto(nombre_archivo):
    with open(nombre_archivo, "r") as file:
        content = file.read()
        return content

def main(txt_file_name):
    # print("el argumento es: " + txt)
    txt = obten_texto(txt_file_name)
    lista_resps = main_visa_prediction(txt)

    print(lista_resps)

# Se llama al script pasando el nombre del archvio txt como argumento
# Así: python visa_predict_script.py row_data.txt

if __name__ == "__main__":
    #print(f"Script name: {sys.argv[0]}")
    if len(sys.argv) > 1:
    #    print("Arguments:")
    #    for i, arg in enumerate(sys.argv[1:]):
    #        print(f"  {i+1}: {arg}")
        main(sys.argv[1])
    else:
        print("No se ha recibido ningún argumento para analizarlo")