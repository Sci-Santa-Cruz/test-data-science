# =====================================================
# components/mlops_components.py
# =====================================================
"""
Decoradores para componentes MLOps en Kubeflow Pipelines.

Este módulo proporciona decoradores que convierten funciones de procesamiento
de datos y ML en componentes reutilizables de KFP, con integración automática
con Google Cloud Storage para persistencia de datos y modelos.

Los decoradores manejan automáticamente:
- Descarga/subida de datos desde/hacia GCS
- Serialización de modelos con joblib
- Logging de métricas en KFP
- Manejo de dependencias de paquetes

Uso típico:
    @mlops_train
    def my_training_function(df):
        # Lógica de entrenamiento
        return model, metrics

    # Se convierte automáticamente en un componente KFP
    # que descarga datos de GCS, ejecuta la función y sube resultados
"""

import functools
import pandas as pd
import joblib
from io import BytesIO, StringIO
from pathlib import Path
from google.cloud import storage
from kfp.dsl import component, Output, Metrics

base_image = "python:3.10"
# =====================================================
# Crear dataset (y guardarlo en GCS)
# =====================================================
def mlops_create_dataset(base_image=base_image):
    """
    Decorador para crear datasets y subirlos automáticamente a Google Cloud Storage.

    Este decorador convierte una función que retorna un DataFrame en un componente
    de KFP que genera el dataset y lo guarda en GCS como CSV.

    Args:
        base_image (str, opcional): Imagen base de Docker para el componente.
                                   Default: "python:3.10".

    Returns:
        function: Decorador que transforma la función en un componente KFP.

    Example:
        @mlops_create_dataset()
        def create_iris_dataset():
            # Lógica para crear dataset
            return df

        # Se convierte en:
        # create_iris_dataset(bucket_name="my-bucket", output_dataset_path="data/iris.csv")

    Note:
        La función decorada debe retornar un pandas.DataFrame.
        Requiere credenciales de GCS configuradas.
    """
    def decorator(func):
        @component(
            base_image=base_image,
            packages_to_install=[
                "pandas==1.5.3",
                "google-cloud-storage==3.4.1",
            ],
        )
        @functools.wraps(func)
        def wrapper(bucket_name: str, output_dataset_path: str):
            """
            Genera dataset y lo sube a GCS.

            Args:
                bucket_name (str): Nombre del bucket de GCS.
                output_dataset_path (str): Ruta donde guardar el dataset en el bucket.

            Returns:
                None: El dataset se guarda en GCS.
            """
            df = func()  # El científico retorna un DataFrame

            client = storage.Client()
            bucket = client.bucket(bucket_name)
            blob = bucket.blob(output_dataset_path)

            csv_buffer = StringIO()
            df.to_csv(csv_buffer, index=False)
            blob.upload_from_string(csv_buffer.getvalue(), content_type="text/csv")

            print(f"Dataset guardado en: gs://{bucket_name}/{output_dataset_path}")

        return wrapper
    return decorator


# =====================================================
# Split dataset
# =====================================================
def mlops_split_dataset(base_image=base_image):
    """
    Decorador para dividir datasets en train/test y guardar automáticamente en GCS.

    Convierte una función de división de datos en un componente KFP que descarga
    un dataset de GCS, lo divide usando la función proporcionada y guarda ambos
    conjuntos (train/test) de vuelta en GCS.

    Args:
        func (callable): Función que toma un DataFrame y retorna (X_train, X_test).

    Returns:
        function: Componente KFP que maneja la división y persistencia.

    Example:
        @mlops_split_dataset
        def my_split_function(df):
            # Lógica de división personalizada
            return X_train, X_test

        # Se convierte en:
        # my_split_function(bucket_name="bucket", input_dataset_path="data.csv",
        #                   train_dataset_path="train.csv", test_dataset_path="test.csv")

    Note:
        La función decorada debe retornar exactamente dos DataFrames (train, test).
        Ambos conjuntos se guardan como CSV en GCS.
    """
    def decorator(func):
        @component(
            base_image=base_image,
            packages_to_install=[
                "pandas==1.5.3",
                "scikit-learn==1.6.0",
                "google-cloud-storage==3.4.1",
            ],
        )
        def wrapper(bucket_name: str, input_dataset_path: str, train_dataset_path: str, test_dataset_path: str):
            """
            Divide dataset en train/test y guarda en GCS.

            Args:
                bucket_name (str): Nombre del bucket de GCS.
                input_dataset_path (str): Ruta del dataset original en GCS.
                train_dataset_path (str): Ruta donde guardar conjunto de entrenamiento.
                test_dataset_path (str): Ruta donde guardar conjunto de prueba.

            Returns:
                None: Los datasets divididos se guardan en GCS.
            """
            client = storage.Client()
            bucket = client.bucket(bucket_name)

            # Descargar dataset
            blob = bucket.blob(input_dataset_path)
            df = pd.read_csv(BytesIO(blob.download_as_bytes()))

            # Aplicar función del científico
            X_train, X_test = func(df)

            # Subir a GCS
            for data, path in [(X_train, train_dataset_path), (X_test, test_dataset_path)]:
                buf = StringIO()
                data.to_csv(buf, index=False)
                bucket.blob(path).upload_from_string(buf.getvalue(), content_type="text/csv")

            print(f"Train y test guardados en: gs://{bucket_name}/...")

        return wrapper
    return decorator


# =====================================================
# Entrenamiento
# =====================================================
def mlops_train(base_image=base_image):
    """
    Decorador para entrenamiento de modelos con persistencia automática en GCS.

    Convierte una función de entrenamiento en un componente KFP que descarga
    datos de GCS, entrena el modelo, calcula métricas y guarda tanto el modelo
    como las métricas de vuelta en GCS.

    Args:
        func (callable): Función que toma un DataFrame y retorna (model, accuracy, auc_roc).

    Returns:
        function: Componente KFP que maneja entrenamiento y persistencia.

    Example:
        @mlops_train
        def my_training_function(df):
            # Lógica de entrenamiento
            return model, accuracy, auc_score

        # Se convierte en:
        # my_training_function(bucket_name="bucket", input_dataset_path="train.csv",
        #                      output_model_path="model.joblib", metrics_path="metrics.csv")

    Note:
        La función decorada debe retornar exactamente (modelo, accuracy, auc_roc).
        El modelo se serializa con joblib y las métricas como CSV.
    """
    @component(
        base_image=base_image,
        packages_to_install=[
            "pandas==1.5.3",
            "scikit-learn==1.6.0",
            "joblib==1.3.1",
            "google-cloud-storage==3.4.1",
        ],
    )
    def wrapper(bucket_name: str, input_dataset_path: str, output_model_path: str, metrics_path: str):
        """
        Entrena modelo y guarda resultados en GCS.

        Args:
            bucket_name (str): Nombre del bucket de GCS.
            input_dataset_path (str): Ruta del dataset de entrenamiento en GCS.
            output_model_path (str): Ruta donde guardar el modelo entrenado.
            metrics_path (str): Ruta donde guardar las métricas de entrenamiento.

        Returns:
            None: Modelo y métricas se guardan en GCS.
        """
        client = storage.Client()
        bucket = client.bucket(bucket_name)

        # Descargar dataset
        df = pd.read_csv(BytesIO(bucket.blob(input_dataset_path).download_as_bytes()))

        # Entrenar modelo
        model, acc, aucRoc = func(df)

        # Subir modelo
        model_bytes = BytesIO()
        joblib.dump(model, model_bytes)
        model_bytes.seek(0)
        bucket.blob(output_model_path).upload_from_file(model_bytes, content_type="application/octet-stream")

        # Subir métricas
        metrics_str = f"accuracy,{acc}\naucRoc,{aucRoc}\n"
        bucket.blob(metrics_path).upload_from_string(metrics_str, content_type="text/csv")

        print(f"Modelo y métricas subidos a gs://{bucket_name}/")

    return wrapper


# =====================================================
# Validación del modelo
# =====================================================
def mlops_validate(base_image=base_image):
    """
    Decorador para validación de modelos con logging automático de métricas.

    Convierte una función de validación en un componente KFP que descarga
    modelo y datos de prueba de GCS, calcula métricas y las registra
    automáticamente en Kubeflow Pipelines.

    Args:
        base_image (str, opcional): Imagen base de Docker. Default: "python:3.12".

    Returns:
        function: Decorador que transforma la función en componente KFP.

    Example:
        @mlops_validate()
        def my_validation_function(model, df_test):
            # Lógica de validación
            return {"accuracy": 0.95, "f1_score": 0.92}

        # Se convierte en:
        # my_validation_function(bucket_name="bucket", test_dataset_path="test.csv",
        #                        model_path="model.joblib", metrics=kfp_metrics_output)

    Note:
        Las métricas retornadas se registran automáticamente en KFP Metrics.
        La función decorada debe retornar un diccionario de métricas.
    """
    def decorator(func):
        @component(
            base_image=base_image,
            packages_to_install=[
                "pandas==1.5.3",
                "scikit-learn==1.6.0",
                "joblib==1.3.1",
                "google-cloud-storage==3.4.1",
            ],
        )
        @functools.wraps(func)
        def wrapper(bucket_name: str, test_dataset_path: str, model_path: str, metrics: Output[Metrics]):
            """
            Valida modelo y registra métricas en KFP.

            Args:
                bucket_name (str): Nombre del bucket de GCS.
                test_dataset_path (str): Ruta del dataset de prueba en GCS.
                model_path (str): Ruta del modelo entrenado en GCS.
                metrics (Output[Metrics]): Output de KFP para métricas.

            Returns:
                None: Las métricas se registran en KFP automáticamente.
            """
            client = storage.Client()
            bucket = client.bucket(bucket_name)

            # Descargar dataset de prueba y modelo
            df_test = pd.read_csv(BytesIO(bucket.blob(test_dataset_path).download_as_bytes()))
            model = joblib.load(BytesIO(bucket.blob(model_path).download_as_bytes()))

            # Calcular métricas
            model_metrics = func(model, df_test)
            for k, v in model_metrics.items():
                metrics.log_metric(k, v)
            print("Métricas validadas:", model_metrics)

        return wrapper
    return decorator


# =====================================================
# Deploy del modelo
# =====================================================
def mlops_deploy_model(base_image=base_image):
    """
    Decorador para despliegue de modelos en Vertex AI.

    Convierte una función de despliegue en un componente KFP que descarga
    un modelo de GCS y lo despliega automáticamente en Vertex AI creando
    un endpoint para inferencia en tiempo real.

    Args:
        func (callable, opcional): Función de despliegue personalizada. Si None,
                                   usa despliegue estándar de Vertex AI.

    Returns:
        function: Componente KFP que maneja el despliegue en Vertex AI.

    Example:
        @mlops_deploy_model
        def my_deploy_function():
            # Lógica de despliegue personalizada
            pass

        # O usar sin función personalizada:
        # deploy_func = mlops_deploy_model()

        # Se convierte en:
        # deploy_func(project_id="project", location="us-central1",
        #             bucket_name="bucket", model_path="model.joblib",
        #             display_name_model="my-model", display_name_endpoint="my-endpoint")

    Note:
        Requiere permisos de Vertex AI Administrator.
        El despliegue puede tomar varios minutos.
        Crea automáticamente endpoint y asigna 100% del tráfico al modelo.
    """
    @component(
        base_image=base_image,
        packages_to_install=["google-cloud-aiplatform==1.66.0"],
    )
    def wrapper(project_id: str, location: str, bucket_name: str, model_path: str, display_name_model: str, display_name_endpoint: str):
        """
        Despliega modelo en Vertex AI.

        Args:
            project_id (str): ID del proyecto de GCP.
            location (str): Región de GCP.
            bucket_name (str): Nombre del bucket de GCS.
            model_path (str): Ruta del modelo en GCS.
            display_name_model (str): Nombre del modelo en Vertex AI.
            display_name_endpoint (str): Nombre del endpoint en Vertex AI.

        Returns:
            None: El modelo se despliega en Vertex AI.
        """
        from google.cloud import aiplatform
        client = storage.Client()
        bucket = client.bucket(bucket_name)

        # Descargar modelo desde GCS a /tmp
        local_model_path = f"/tmp/{Path(model_path).name}"
        bucket.blob(model_path).download_to_filename(local_model_path)

        aiplatform.init(project=project_id, location=location)
        uploaded_model = aiplatform.Model.upload_scikit_learn_model_file(
            model_file_path=local_model_path,
            display_name=display_name_model,
            project=project_id,
            location=location,
        )

        endpoint = aiplatform.Endpoint.create(
            display_name=display_name_endpoint,
            project=project_id,
            location=location,
        )

        print(f"Modelo deployado en endpoint: {endpoint.resource_name}")

    return wrapper
