"""
Sistema de configuración centralizada para el pipeline MLOps.

Este módulo proporciona una clase PipelineConfig que carga configuraciones
desde archivos YAML y las pone a disposición de todo el pipeline de manera
centralizada y tipada.

Uso:
    from config import PipelineConfig

    # Cargar config para entorno actual
    config = PipelineConfig.load()

    # O especificar entorno
    config = PipelineConfig.load('prod')

    # Obtener parámetros para el pipeline
    params = config.get_pipeline_params()
"""

import os
import yaml
from pathlib import Path
from typing import Dict, Any, Optional, Union
from dataclasses import dataclass, field, asdict


@dataclass
class PipelineConfig:
    """
    Configuración centralizada del pipeline MLOps.

    Esta clase maneja la carga y validación de configuraciones desde archivos YAML,
    proporcionando una interfaz tipada y validada para acceder a todos los parámetros
    del pipeline.

    Attributes:
        project (dict): Información general del proyecto
        infrastructure (dict): Configuración de infraestructura GCP
        pipeline (dict): Parámetros del pipeline
        components (dict): Configuraciones específicas por componente
        utils (dict): Configuración de utilidades
        logging (dict): Configuración de logging
        monitoring (dict): Configuración de monitoreo
    """

    project: Dict[str, Any] = field(default_factory=dict)
    infrastructure: Dict[str, Any] = field(default_factory=dict)
    pipeline: Dict[str, Any] = field(default_factory=dict)
    components: Dict[str, Any] = field(default_factory=dict)
    utils: Dict[str, Any] = field(default_factory=dict)
    logging: Dict[str, Any] = field(default_factory=dict)
    monitoring: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def load(cls, environment: str = None) -> 'PipelineConfig':
        """
        Carga configuración desde archivos YAML.

        Carga la configuración base y la combina con la configuración específica
        del entorno seleccionado.

        Args:
            environment (str, opcional): Entorno ('dev', 'prod', etc.).
                                        Default: variable de entorno ENVIRONMENT o 'dev'.

        Returns:
            PipelineConfig: Instancia configurada.

        Raises:
            FileNotFoundError: Si no se encuentra el archivo de configuración base.
            yaml.YAMLError: Si hay error al parsear YAML.
        """
        if environment is None:
            environment = os.getenv('ENVIRONMENT', 'dev')

        config_dir = Path(__file__).parent

        # Cargar configuración base
        base_path = config_dir / 'base.yaml'
        if not base_path.exists():
            raise FileNotFoundError(f"Archivo de configuración base no encontrado: {base_path}")

        with open(base_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f) or {}

        # Cargar configuración específica del entorno
        env_path = config_dir / f'{environment}.yaml'
        if env_path.exists():
            with open(env_path, 'r', encoding='utf-8') as f:
                env_config = yaml.safe_load(f) or {}
                # Merge configs (entorno override base)
                config = cls._deep_merge(config, env_config)

        return cls(**config)

    @staticmethod
    def _deep_merge(base: dict, override: dict) -> dict:
        """
        Realiza un merge profundo de diccionarios.

        Args:
            base (dict): Diccionario base
            override (dict): Diccionario con overrides

        Returns:
            dict: Diccionario merged
        """
        result = base.copy()
        for key, value in override.items():
            if isinstance(value, dict) and key in result and isinstance(result[key], dict):
                result[key] = PipelineConfig._deep_merge(result[key], value)
            else:
                result[key] = value
        return result

    def get_pipeline_params(self) -> dict:
        """
        Retorna parámetros formateados para el pipeline principal.

        Formatea y valida todos los parámetros necesarios para ejecutar el pipeline,
        incluyendo interpolación de variables en paths.

        Returns:
            dict: Parámetros listos para pasar al pipeline.
        """
        params = {
            # Infraestructura
            'project_id': self.infrastructure.get('project_id'),
            'location': self.infrastructure.get('location'),
            'bucket_name': self.infrastructure.get('bucket_name'),

            # Dataset y modelo
            'dataset_name': self.pipeline.get('dataset_name'),
            'model_version': self.pipeline.get('model_version'),
            'dataset_version': self.pipeline.get('dataset_version'),

            # Hiperparámetros
            'hyperparams': self.pipeline.get('model', {}).get('hyperparameters', {}),

            # Configuración operacional
            'force_retrain': self.pipeline.get('operational', {}).get('force_retrain', False),
            'retry_count': self.pipeline.get('operational', {}).get('retry_count', 3),
            'log_level': self.pipeline.get('operational', {}).get('log_level', 'INFO'),

            # Notificaciones
            'email_addresses': self.pipeline.get('operational', {}).get('email_addresses', []),
        }

        # Formatear paths con variables
        paths_config = self.pipeline.get('paths', {})
        dataset_name = params['dataset_name']
        model_version = params['model_version']
        dataset_version = params['dataset_version']

        params.update({
            'raw_dataset_path': paths_config.get('raw_dataset', 'data/raw/{dataset_name}_{dataset_version}.csv').format(
                dataset_name=dataset_name, dataset_version=dataset_version),
            'train_dataset_path': paths_config.get('train_dataset', 'data/processed/{dataset_name}_train_{dataset_version}.csv').format(
                dataset_name=dataset_name, dataset_version=dataset_version),
            'test_dataset_path': paths_config.get('test_dataset', 'data/processed/{dataset_name}_test_{dataset_version}.csv').format(
                dataset_name=dataset_name, dataset_version=dataset_version),
            'model_path': paths_config.get('model', 'models/{dataset_name}_model_{model_version}.joblib').format(
                dataset_name=dataset_name, model_version=model_version),
            'metrics_path': paths_config.get('metrics', 'metrics/{dataset_name}_train_metrics_{model_version}.csv').format(
                dataset_name=dataset_name, model_version=model_version),
        })

        # Formatear nombres de despliegue
        deployment_config = self.pipeline.get('deployment', {})
        params.update({
            'display_name_model': deployment_config.get('display_name_model', '{dataset_name}-model-{model_version}').format(
                dataset_name=dataset_name, model_version=model_version),
            'display_name_endpoint': deployment_config.get('display_name_endpoint', '{dataset_name}-endpoint-{model_version}').format(
                dataset_name=dataset_name, model_version=model_version),
        })

        return params

    def get_component_config(self, component_name: str) -> dict:
        """
        Obtiene configuración específica para un componente.

        Args:
            component_name (str): Nombre del componente.

        Returns:
            dict: Configuración del componente.
        """
        return self.components.get(component_name, {})

    def validate(self) -> bool:
        """
        Valida que la configuración tenga todos los parámetros requeridos.

        Returns:
            bool: True si la configuración es válida.

        Raises:
            ValueError: Si faltan parámetros requeridos.
        """
        required_fields = [
            'infrastructure.project_id',
            'infrastructure.location',
            'infrastructure.bucket_name',
            'pipeline.dataset_name'
        ]

        for field_path in required_fields:
            keys = field_path.split('.')
            value = self._get_nested_value(keys)
            if not value:
                raise ValueError(f"Campo requerido faltante: {field_path}")

        return True

    def _get_nested_value(self, keys: list) -> Any:
        """Obtiene valor anidado de la configuración."""
        value = asdict(self)
        for key in keys:
            if isinstance(value, dict):
                value = value.get(key)
            else:
                return None
        return value

    def to_dict(self) -> dict:
        """Convierte la configuración a diccionario."""
        return asdict(self)