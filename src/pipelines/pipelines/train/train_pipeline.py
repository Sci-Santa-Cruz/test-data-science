import os
import sys
import argparse
from datetime import datetime
from typing import Optional
from kfp import compiler, dsl
from kfp.registry import RegistryClient
from google_cloud_pipeline_components.v1.vertex_notification_email import (
    VertexNotificationEmailOp,
)
# local imports
from mlops_lib.decorators import (
    mlops_create_dataset,
    mlops_split_dataset,
    mlops_train,
    mlops_validate,
    mlops_deploy_model
)
from utils.utils import get_next_available_version
from config import PipelineConfig

# Variables de entorno para compatibilidad
ENVIRONMENT = os.getenv("_ENVIRONMENT", os.getenv("ENVIRONMENT", "dev"))
TIMESTAMP = datetime.now().strftime("%Y%m%d%H%M%S")
PIPELINE_REPO = os.getenv("_PIPELINE_REPO")

sys.path.append("src/")

# =====================================================
# Pipeline principal
# =====================================================
def mlops_pipeline(
    config: Optional[PipelineConfig] = None,
    environment: str = None,
    **overrides
):
    """
    Pipeline de MLOps end-to-end con configuración centralizada.

    Args:
        config (PipelineConfig, opcional): Configuración del pipeline. Si None, se carga automáticamente.
        environment (str, opcional): Entorno ('dev', 'prod'). Default: variable de entorno.
        **overrides: Parámetros específicos para override de configuración.

    Returns:
        None: El pipeline se registra en KFP.
    """
    pass

# =====================================================
# Compilación y registro del pipeline
# =====================================================
def main():
    """
    Función principal para compilar y registrar el pipeline con configuración centralizada.

    Proporciona una interfaz de línea de comandos para:
    - Compilar el pipeline en un archivo YAML usando configuración del entorno
    - Registrar el pipeline compilado en el repositorio de Vertex AI

    La configuración se carga automáticamente desde archivos YAML basados en el entorno.

    Uso:
        python pipeline.py --compile
        python pipeline.py --register --env prod

    Variables de entorno:
    - ENVIRONMENT: Entorno ('dev', 'prod', etc.) - Default: 'dev'
    - _PIPELINE_REPO: URL del repositorio de pipelines (opcional)

    Returns:
        None
    """
    pass