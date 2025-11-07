
"""
Componente de validación de modelos en el pipeline MLOps.

Este módulo contiene la función validate_model que evalúa el rendimiento
de un modelo entrenado usando métricas configurables. Utiliza el decorador
@mlops_validate para integración automática con KFP y logging de métricas.
"""

from typing import Dict, Any, Union
import pandas as pd
from mlops_lib.decorators import mlops_validate
from config import PipelineConfig
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report
)


@mlops_validate()
def validate_model(model: Any, df_test: pd.DataFrame) -> Dict[str, Any]:
    """
    Valida un modelo entrenado calculando métricas de rendimiento.

    Esta función se ejecuta dentro de un componente KFP que automáticamente:
    - Descarga modelo y datos de prueba desde GCS
    - Calcula métricas configurables de rendimiento
    - Registra métricas en KFP para tracking y comparación
    - Opcionalmente genera matriz de confusión y reportes detallados

    Las métricas calculadas se configuran centralizadamente permitiendo
    diferentes niveles de evaluación por entorno.

    Configuración utilizada:
        - components.validate_model.target_column: Columna objetivo (default: 'Class')
        - components.validate_model.metrics: Lista de métricas a calcular
          ['accuracy', 'precision', 'recall', 'f1_score']
        - components.validate_model.confusion_matrix: Si generar matriz de confusión (default: False)
        - components.validate_model.classification_report: Si generar reporte detallado (default: False)

    Args:
        model (Any): Modelo entrenado de sklearn. Este parámetro es manejado
                    automáticamente por el decorador desde GCS.
        df_test (pd.DataFrame): Dataset de prueba con features y target. Este
                               parámetro es manejado automáticamente por el
                               decorador desde GCS.

    Returns:
        Dict[str, Any]: Diccionario con métricas calculadas. Las claves dependen
        de la configuración:
        - 'accuracy': Precisión (0.0-1.0)
        - 'precision': Precisión weighted average (0.0-1.0)
        - 'recall': Recall weighted average (0.0-1.0)
        - 'f1_score': F1-score weighted average (0.0-1.0)
        - 'confusion_matrix': Matriz de confusión como lista de listas (opcional)
        - 'classification_report': Reporte completo como dict (opcional)

    Raises:
        KeyError: Si target_column no existe en df_test.
        ValueError: Si alguna métrica no puede calcularse.

    Example:
        >>> # Se ejecuta automáticamente por el pipeline KFP
        >>> metrics = validate_model(model, df_test)
        >>> print(f"Accuracy: {metrics['accuracy']:.3f}")
        >>> print(f"F1-Score: {metrics['f1_score']:.3f}")

    Note:
        Todas las métricas se calculan con average='weighted' para manejar
        clases desbalanceadas apropiadamente en problemas multiclase.
    """
    pass


