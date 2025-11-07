
"""
Componente para creación de datasets en el pipeline MLOps.

Este módulo contiene la función create_dataset que se encarga de descargar
datasets desde BigQuery y prepararlos para el pipeline de ML. Utiliza el
decorador @mlops_create_dataset para integración automática con KFP y GCS.
"""

from typing import Optional
from mlops_lib.decorators import mlops_create_dataset
from config import PipelineConfig
from utils.utils import validate_dataframe


@mlops_create_dataset
def create_dataset() -> pd.DataFrame:
    """
    Crea un dataset descargándolo desde BigQuery público.

    Esta función se ejecuta dentro de un componente KFP que automáticamente:
    - Descarga el dataset usando la query configurada
    - Valida el dataset según las reglas de configuración
    - Guarda el resultado en GCS como CSV

    La configuración se carga automáticamente desde los archivos YAML del proyecto
    (base.yaml, dev.yaml/prod.yaml) permitiendo diferentes queries y validaciones
    por entorno.

    Configuración utilizada:
        - components.create_dataset.query: Query SQL para BigQuery
        - components.create_dataset.timeout_seconds: Timeout de la query
        - utils.data_validation.*: Reglas de validación del dataset

    Returns:
        pd.DataFrame: Dataset descargado y validado listo para procesamiento.

    Raises:
        google.api_core.exceptions.GoogleAPIError: Error de conexión con BigQuery.
        ValueError: Si el dataset no cumple las validaciones configuradas.

    Example:
        >>> # Se ejecuta automáticamente por el pipeline KFP
        >>> df = create_dataset()
        >>> print(f"Dataset shape: {df.shape}")

    Note:
        Requiere credenciales de GCP configuradas en el entorno donde se ejecuta
        el componente (Vertex AI, Cloud Run, o local con gcloud auth).
    """
    pass
