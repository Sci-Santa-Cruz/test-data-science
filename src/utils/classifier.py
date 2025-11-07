from pickle import load as load_
from mlflow import MlflowClient
from numpy import argmax, max as max_, around
from sklearn.feature_extraction.text import HashingVectorizer
# from transformers.models.auto import AutoConfig, AutoModelForSequenceClassification
# from transformers.models.auto.tokenization_auto import AutoTokenizer
from os.path import exists, join
from datetime import datetime
# from torch import device, no_grad, cuda, cat
from logging import getLogger
from os import getenv
import mlflow.sklearn
import joblib
import numpy as np

logger = getLogger(__name__)



class ModelClassifier:

    def __init__(self):
        
        self.model_path = getenv('MODEL_URI')

        self.client = MlflowClient()

        self.loaded_model = mlflow.sklearn.load_model(self.model_path)

    def load_config(self):
        """
        Carga la configuración del modelo desde el modelo padre definido en MODEL_PARENT_PATH.
        """
        self.logger.info('Load config for model ML')
        self.model_parent = getenv('MODEL_PARENT_PATH')
        self.logger.info('Done Load config for model ML ')


    
    def predict(self, X):
        return self.loaded_model.predict(X)

    def get_predict_proba(self,features):
        y_pred_prob = self.loaded_model.predict_proba(features)
        ix = y_pred_prob.argmax(1)
        values = np.max(y_pred_prob, axis=1)
        return zip(list(ix),list(values))
    
    def predict_proba(self, X):
        if hasattr(self.loaded_model, 'predict_proba'):
            return self.loaded_model.predict_proba(X)
        else:
            raise AttributeError("El modelo no admite predict_proba.")
    
    def get_best_params(self):
        return self.loaded_model.best_params_
    
    def get_best_score(self):
        return self.loaded_model.best_score_

# Uso de la clase para cargar y utilizar el modelo
if __name__ == "__main__":
    model_path = "ruta_al_modelo_entrenado"
    mlflow_model = ModelClassifier()
    
    # Cargar tus datos de prueba (reemplazar esto con tus propios datos)
    X_test = []    
    # Realizar predicciones
    predictions = mlflow_model.predict(X_test)
    
    # Obtener los parámetros y puntaje del mejor modelo
    best_params = mlflow_model.get_best_params()
    best_score = mlflow_model.get_best_score()
    
    print("Predicciones:", predictions)
    print("Mejores parámetros:", best_params)
    print("Mejor puntaje:", best_score)
