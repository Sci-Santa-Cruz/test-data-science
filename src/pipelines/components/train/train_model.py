
"""
Componente de entrenamiento de modelos en el pipeline MLOps.

Este módulo contiene la función train_model que entrena un modelo de ML
usando datos preparados. Utiliza el decorador @mlops_train para integración
automática con KFP y GCS, permitiendo configuración centralizada de
hiperparámetros y métricas.
"""

from typing import Tuple, Any
import pandas as pd
from mlops_lib.decorators import mlops_train
from config import PipelineConfig
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score
from utils.utils import split_data_local


@mlops_train
def train_model(df: pd.DataFrame) -> Tuple[Any, float, float]:
    """
    Entrena un modelo de machine learning con datos preparados.

    Esta función se ejecuta dentro de un componente KFP que automáticamente:
    - Descarga datos de entrenamiento desde GCS
    - Entrena el modelo usando hiperparámetros configurables
    - Calcula métricas de validación
    - Guarda el modelo entrenado y métricas en GCS

    Soporta diferentes tipos de modelos configurables y calcula métricas
    estándar para evaluación del rendimiento.

    Configuración utilizada:
        - components.train_model.target_column: Columna objetivo (default: 'Class')
        - components.train_model.validation_split: Proporción para validación (default: 0.2)
        - pipeline.model.type: Tipo de modelo ('LogisticRegression', etc.)
        - pipeline.model.hyperparameters.*: Hiperparámetros del modelo
        - components.train_model.metrics: Lista de métricas a calcular

    Args:
        df (pd.DataFrame): Dataset de entrenamiento con features y target.
                          Este parámetro es manejado automáticamente por el
                          decorador desde GCS.

    Returns:
        Tuple[Any, float, float]: Tupla con (modelo_entrenado, accuracy, auc_roc).
        - modelo_entrenado: Modelo de sklearn entrenado y listo para predicciones
        - accuracy: Precisión en conjunto de validación (0.0-1.0)
        - auc_roc: Área bajo la curva ROC para clasificación multiclase

    Raises:
        ValueError: Si el tipo de modelo no es soportado.
        KeyError: Si faltan columnas requeridas en el dataset.

    Example:
        >>> # Se ejecuta automáticamente por el pipeline KFP
        >>> model, acc, auc = train_model(df)
        >>> print(f"Accuracy: {acc:.3f}, AUC-ROC: {auc:.3f}")

    Note:
        Actualmente soporta LogisticRegression. Se puede extender para otros
        modelos agregando casos en la lógica de creación del modelo.
    """
    pass