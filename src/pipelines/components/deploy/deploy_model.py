
"""
Componente de despliegue de modelos en el pipeline MLOps.

Este módulo contiene la función deploy_model que despliega un modelo entrenado
en Vertex AI para inferencia en producción. Utiliza el decorador @mlops_deploy_model
para integración automática con KFP y Vertex AI.
"""

from mlops_lib.decorators import mlops_deploy_model
from config import PipelineConfig


@mlops_deploy_model
def deploy_model() -> None:
    """
    Despliega un modelo entrenado en Vertex AI para inferencia en producción.

    Esta función se ejecuta dentro de un componente KFP que automáticamente:
    - Descarga el modelo entrenado desde GCS
    - Crea o actualiza un endpoint en Vertex AI
    - Despliega el modelo con configuración de recursos y escalado
    - Configura monitoreo si está habilitado

    El despliegue se configura centralizadamente permitiendo diferentes
    estrategias por entorno (desarrollo vs producción).

    Configuración utilizada:
        - pipeline.deployment.display_name_model: Nombre del modelo en Vertex AI
        - pipeline.deployment.display_name_endpoint: Nombre del endpoint
        - pipeline.deployment.machine_type: Tipo de máquina para inferencia
        - pipeline.deployment.min_replica_count: Mínimo número de réplicas
        - pipeline.deployment.max_replica_count: Máximo número de réplicas
        - components.deploy_model.traffic_split: Porcentaje de tráfico (default: 100)
        - components.deploy_model.enable_monitoring: Si habilitar monitoreo (default: True)
        - components.deploy_model.monitoring_config: Configuración de monitoreo

    Returns:
        None: El despliegue se realiza automáticamente por el decorador.
             Los resultados se manejan a nivel de infraestructura Vertex AI.

    Raises:
        google.api_core.exceptions.GoogleAPIError: Error en APIs de Vertex AI.
        ValueError: Si la configuración de despliegue es inválida.

    Example:
        >>> # Se ejecuta automáticamente por el pipeline KFP
        >>> deploy_model()
        >>> # El modelo queda desplegado en Vertex AI endpoint

    Note:
        Requiere permisos de Vertex AI Administrator en el proyecto GCP.
        El despliegue puede tomar varios minutos en completarse.
        Se recomienda monitorear el estado del endpoint en Cloud Console.
    """
    pass