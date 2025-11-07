"""
Módulo de utilidades para el proyecto MLOps.

Este módulo contiene funciones auxiliares que pueden ser utilizadas
por componentes del pipeline y otros módulos del proyecto.
Todas las funciones están diseñadas para ser configurables y reutilizables.

Las utilidades pueden ser configuradas a través del sistema de configuración
centralizada del proyecto.
"""

import pandas as pd
from sklearn.model_selection import train_test_split
from typing import Tuple, Optional, Dict, Any, Union
from config import PipelineConfig
import re
from kfp.registry import RegistryClient
import time
import numpy as np


def split_data_local(
    df: pd.DataFrame,
    target_column: Optional[str] = None,
    test_size: Optional[float] = None,
    random_state: Optional[int] = None,
    stratify: Optional[bool] = None,
    config: Optional[PipelineConfig] = None
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Divide un DataFrame en conjuntos de entrenamiento y prueba localmente.

    Esta función es una versión local de la división de datos que puede ser
    utilizada para testing o procesamiento fuera del pipeline de KFP.
    Puede usar configuración centralizada o parámetros directos.

    Args:
        df (pd.DataFrame): DataFrame con features y target.
        target_column (str, opcional): Nombre de la columna objetivo.
        test_size (float, opcional): Proporción del conjunto de prueba (0.0-1.0).
        random_state (int, opcional): Semilla para reproducibilidad.
        stratify (bool, opcional): Si estratificar por la variable objetivo.
        config (PipelineConfig, opcional): Configuración centralizada.

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame]: (X_train, X_test) con features + target.

    Raises:
        KeyError: Si target_column no existe en el DataFrame.
        ValueError: Si test_size no está en rango válido.

    Example:
        >>> df = pd.read_csv('data.csv')
        >>> X_train, X_test = split_data_local(df, test_size=0.3)
        >>> print(f"Train: {len(X_train)}, Test: {len(X_test)}")

        >>> # Usando configuración
        >>> from config import PipelineConfig
        >>> config = PipelineConfig.load('dev')
        >>> X_train, X_test = split_data_local(df, config=config)
    """
    # Obtener configuración de utils
    if config is None:
        config = PipelineConfig.load()

    utils_config = config.utils.get('data_splitting', {})

    # Usar valores de config si no se especifican
    target_column = target_column or utils_config.get('target_column', 'Class')
    test_size = test_size if test_size is not None else utils_config.get('default_test_size', 0.2)
    random_state = random_state if random_state is not None else utils_config.get('default_random_state', 42)
    stratify = stratify if stratify is not None else utils_config.get('stratify_by_target', True)

    # Validaciones
    if target_column not in df.columns:
        raise KeyError(f"Columna objetivo '{target_column}' no encontrada en DataFrame")

    if not 0.0 < test_size < 1.0:
        raise ValueError("test_size debe estar entre 0.0 y 1.0")

    stratify_col = df[target_column] if stratify else None

    X_train, X_test, y_train, y_test = train_test_split(
        df.drop(target_column, axis=1),
        df[target_column],
        test_size=test_size,
        random_state=random_state,
        stratify=stratify_col
    )

    X_train = X_train.copy()
    X_test = X_test.copy()
    X_train[target_column] = y_train
    X_test[target_column] = y_test

    return X_train, X_test


def get_data_info(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Obtiene información básica sobre un DataFrame.

    Args:
        df (pd.DataFrame): DataFrame a analizar.

    Returns:
        Dict[str, Any]: Información del DataFrame (shape, columns, dtypes, etc.).

    Example:
        >>> info = get_data_info(df)
        >>> print(f"Shape: {info['shape']}")
    """
    return {
        "shape": df.shape,
        "columns": list(df.columns),
        "dtypes": df.dtypes.to_dict(),
        "null_counts": df.isnull().sum().to_dict(),
        "memory_usage": df.memory_usage(deep=True).sum(),
    }


def validate_dataframe(
    df: pd.DataFrame,
    required_columns: Optional[list] = None,
    min_rows: Optional[int] = None,
    config: Optional[PipelineConfig] = None
) -> bool:
    """
    Valida que un DataFrame cumpla con criterios básicos.

    Args:
        df (pd.DataFrame): DataFrame a validar.
        required_columns (list, opcional): Columnas requeridas.
        min_rows (int, opcional): Número mínimo de filas.
        config (PipelineConfig, opcional): Configuración centralizada.

    Returns:
        bool: True si el DataFrame es válido.

    Raises:
        ValueError: Si el DataFrame no cumple los criterios.
    """
    # Obtener configuración de validación
    if config is None:
        config = PipelineConfig.load()

    validation_config = config.utils.get('data_validation', {})

    # Usar valores de config si no se especifican
    min_rows = min_rows if min_rows is not None else validation_config.get('min_rows', 1)
    required_columns = required_columns or validation_config.get('required_columns', [])

    # Validaciones
    if len(df) < min_rows:
        raise ValueError(f"DataFrame debe tener al menos {min_rows} filas")

    if required_columns:
        missing_cols = set(required_columns) - set(df.columns)
        if missing_cols:
            raise ValueError(f"Columnas faltantes: {missing_cols}")

    # Validación de nulos si está configurada
    if not validation_config.get('allow_nulls', True):
        null_counts = df.isnull().sum()
        cols_with_nulls = null_counts[null_counts > 0]
        if not cols_with_nulls.empty:
            raise ValueError(f"Columnas con valores nulos no permitidos: {list(cols_with_nulls.index)}")

    return True


def check_pipeline_version_exists(
    registry_client: RegistryClient,
    pipeline_name: str,
    version: str
) -> bool:
    """
    Verifica si una versión específica de pipeline existe en el registro.

    Args:
        registry_client (RegistryClient): Cliente del registro de pipelines.
        pipeline_name (str): Nombre del pipeline.
        version (str): Versión a verificar.

    Returns:
        bool: True si la versión existe, False en caso contrario.
    """
    try:
        # Intentar obtener la información del pipeline
        pipeline_info = registry_client.get_pipeline(pipeline_name)
        if pipeline_info and 'versions' in pipeline_info:
            # Verificar si la versión específica existe
            return version in [v.get('name') or v.get('version') for v in pipeline_info['versions']]
        return False
    except Exception:
        # Si hay error (pipeline no existe, etc.), asumimos que no existe
        return False


def increment_version(version: str, pattern: str = "0.0.1") -> str:
    """
    Incrementa una versión siguiendo un patrón específico.

    Args:
        version (str): Versión actual (ej. "1.2.3").
        pattern (str): Patrón de incremento. Default: "0.0.1" (incrementa el último dígito).

    Returns:
        str: Nueva versión incrementada.

    Raises:
        ValueError: Si el formato de versión no es válido.

    Examples:
        >>> increment_version("1.0.0", "0.0.1")
        '1.0.1'
        >>> increment_version("1.0.9", "0.0.1")
        '1.0.10'
        >>> increment_version("1.0.99", "0.1.0")
        '1.1.0'
    """
    # Validar formato de versión (debe ser x.y.z)
    version_pattern = r'^(\d+)\.(\d+)\.(\d+)$'
    match = re.match(version_pattern, version)
    if not match:
        raise ValueError(f"Formato de versión inválido: {version}. Debe ser x.y.z")

    major, minor, patch = map(int, match.groups())

    # Parsear patrón de incremento
    pattern_match = re.match(version_pattern, pattern)
    if not pattern_match:
        raise ValueError(f"Patrón de incremento inválido: {pattern}. Debe ser x.y.z")

    inc_major, inc_minor, inc_patch = map(int, pattern_match.groups())

    # Incrementar según el patrón
    new_patch = patch + inc_patch
    new_minor = minor + inc_minor
    new_major = major + inc_major

    # Manejar carry-over (si patch >= 10, incrementar minor, etc.)
    if new_patch >= 10:
        carry_minor = new_patch // 10
        new_patch = new_patch % 10
        new_minor += carry_minor

    if new_minor >= 10:
        carry_major = new_minor // 10
        new_minor = new_minor % 10
        new_major += carry_major

    return f"{new_major}.{new_minor}.{new_patch}"


def get_next_available_version(
    registry_client: RegistryClient,
    pipeline_name: str,
    base_version: str = "1.0.0",
    increment_pattern: str = "0.0.1"
) -> str:
    """
    Obtiene la siguiente versión disponible para un pipeline en el registro.

    Args:
        registry_client (RegistryClient): Cliente del registro de pipelines.
        pipeline_name (str): Nombre del pipeline.
        base_version (str): Versión base para comenzar. Default: "1.0.0".
        increment_pattern (str): Patrón de incremento. Default: "0.0.1".

    Returns:
        str: Siguiente versión disponible.
    """
    version = base_version

    # Buscar versión disponible incrementando hasta encontrar una que no exista
    max_attempts = 100  # Límite de seguridad
    attempts = 0

    while attempts < max_attempts:
        if not check_pipeline_version_exists(registry_client, pipeline_name, version):
            return version
        version = increment_version(version, increment_pattern)
        attempts += 1

    # Si se alcanza el límite, usar timestamp como fallback
    timestamp_version = f"{base_version.split('.')[0]}.{int(time.time())}"
    return timestamp_version

