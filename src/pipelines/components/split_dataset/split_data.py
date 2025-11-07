
"""
Componente para división de datasets en el pipeline MLOps.

Este módulo contiene la función split_data que divide un dataset en conjuntos
de entrenamiento y prueba. Utiliza el decorador @mlops_split_dataset para
integración automática con KFP y GCS, permitiendo configuración centralizada
de parámetros de división.
"""

from typing import Tuple
import pandas as pd
from mlops_lib.decorators import mlops_split_dataset
from config import PipelineConfig
from utils.utils import split_data_local


@mlops_split_dataset
def split_data(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Divide un dataset en conjuntos de entrenamiento y prueba.

    Esta función se ejecuta dentro de un componente KFP que automáticamente:
    - Descarga el dataset original desde GCS
    - Aplica la división usando parámetros configurables
    - Guarda ambos conjuntos (train/test) de vuelta en GCS como CSV

    Los parámetros de división se configuran centralizadamente permitiendo
    diferentes estrategias por entorno (desarrollo vs producción).

    Configuración utilizada:
        - components.split_dataset.target_column: Columna objetivo (default: 'Class')
        - components.split_dataset.test_size: Proporción del conjunto de prueba (default: 0.2)
        - components.split_dataset.random_state: Semilla para reproducibilidad (default: 42)
        - components.split_dataset.stratify: Si estratificar por variable objetivo (default: True)
        - utils.data_splitting.*: Configuración adicional de división de datos

    Args:
        df (pd.DataFrame): Dataset completo con features y target. Este parámetro
                          es manejado automáticamente por el decorador desde GCS.

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame]: Tupla con (X_train, X_test), donde cada
        DataFrame contiene features + target, listos para entrenamiento/validación.

    Raises:
        KeyError: Si target_column no existe en el DataFrame.
        ValueError: Si test_size no está en rango válido (0.0-1.0).

    Example:
        >>> # Se ejecuta automáticamente por el pipeline KFP
        >>> X_train, X_test = split_data(df)
        >>> print(f"Train: {len(X_train)}, Test: {len(X_test)}")

    Note:
        La estratificación asegura distribución balanceada de clases en train/test,
        especialmente importante para datasets desbalanceados.
    """
    pass
