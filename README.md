
# MLOps Project: Detección de Ocupación en Habitaciones

Este repositorio documenta un proyecto de **MLOps** enfocado en la **detección de ocupación en habitaciones** a partir de datos ambientales (temperatura, CO₂, humedad y luz). El proyecto está dividido en dos fases principales:

---

## 🎯 Fases del Proyecto

### **Fase 1: Experimentación (Notebooks)**

Etapa completamente implementada en **Jupyter Notebooks**, donde se desarrolla el análisis exploratorio, la ingeniería de características y el modelado.

### **Fase 2: Productivización (Pipelines MLOps)**

Propuesta de arquitectura para llevar los modelos a producción mediante **Kubeflow Pipelines**, integrando buenas prácticas de MLOps.
Actualmente se encuentra definida como **diseño arquitectónico**, pendiente de implementación.

---

## 📊 Fase 1: Experimentación - COMPLETA ✅

### Descripción del Problema

Sistema para detectar la **ocupación de una habitación** utilizando mediciones ambientales.

* **Variables**: Temperatura (°C), CO₂ (ppm), Humedad relativa (%), Luz (lux)
* **Objetivo**: Predecir ocupación binaria (0 = desocupada, 1 = ocupada)
* **Datos**: 8 días de mediciones continuas provenientes de sensores

### Estructura de Directorios - Fase 1

```
notebooks/
├── 1.Resumen Ejecutivo.ipynb                  # Contexto, objetivos y enfoque metodológico
├── 2.Profiling.ipynb                          # Calidad y estructura de los datos
├── 3.Integración y Formato.ipynb              # Limpieza, unificación y validación de registros
├── 4.Análisis Exploratorio.ipynb              # Relaciones entre variables y patrones temporales
├── 5.Ingeniería de Características.ipynb      # Outliers, escalado, transformaciones
├── 6.Modelado_RegresionLogistica.ipynb        # Modelo supervisado, métricas e interpretación
├── 7.KMeans.ipynb                             # Clustering exploratorio
└── utils/
    └── plots.py                               # Utilidades de visualización
```

### Flujo de Experimentación

1. **Resumen Ejecutivo** – Contexto, objetivos, metodología y alcance.
2. **Profiling de Datos** – Análisis de calidad, estructura y patrones básicos.
3. **Integración y Formato** – Limpieza, tratamiento de valores faltantes y normalización.
4. **Análisis Exploratorio** – Visualización de relaciones entre temperatura, luz, CO₂ y ocupación.
5. **Feature Engineering** – Eliminación de outliers, escalado y transformación de variables.
6. **Modelado (Regresión Logística)** – Entrenamiento, validación y análisis de coeficientes.
7. **Modelado (K-Means)** – Agrupamiento no supervisado y comparación con etiquetas reales.

### Resultados de Experimentación

**Modelo: Regresión Logística**

* **Accuracy**: 0.92
* **Recall (Clase 1)**: 0.86
* **Coeficientes principales**:

  * CO₂: +4.161 → Fuerte indicador de ocupación
  * Humedad: -0.723 → Asociación inversa
  * Temperatura: -0.260 → Efecto débil y posiblemente multicolineal

---

## 🚀 Fase 2: Productivización - PROPUESTA DE ARQUITECTURA

### Estado Actual

La arquitectura de pipelines MLOps se encuentra definida como **propuesta conceptual**, destinada a migrar el código experimental a componentes productivos, aislando la complejidad técnica mediante **decoradores MLOps**.

### Principios de Diseño

Los decoradores MLOps permiten que el científico de datos se enfoque en el modelo, mientras la infraestructura maneja automáticamente:

* Persistencia en **Cloud Storage**
* Registro de métricas
* Serialización de artefactos
* Escalado automático
* Manejo de dependencias

### Arquitectura Propuesta

```
BigQuery → GCS → Preprocesamiento → Entrenamiento → Validación → Despliegue
```

**Componentes Clave:**

* `create_dataset.py`: Extracción desde BigQuery
* `split_data.py`: División estratificada train/test
* `train_model.py`: Entrenamiento del modelo
* `validate_model.py`: Evaluación y métricas
* `deploy_model.py`: Despliegue a Vertex AI

### Infraestructura Cloud Propuesta

**Google Cloud Platform**

* **BigQuery** → Fuente de datos
* **Cloud Storage** → Almacenamiento de datasets y modelos
* **Vertex AI** → Despliegue y monitoreo de modelos
* **Kubeflow Pipelines** → Orquestación y versionado

---

## 🧱 Estructura de Directorios - Fase 2

```
src/
├── conf/
│   ├── .local/
│   └── data_dict.json
├── mlops_lib/
│   └── decorators.py
├── pipelines/
│   ├── components/
│   │   ├── create_dataset/
│   │   ├── split_dataset/
│   │   ├── train/
│   │   ├── validate/
│   │   └── deploy/
│   └── pipelines/
│       ├── train/
│       ├── batch/
│       └── online/
└── utils/
    ├── classifier.py
    ├── custom.py
    ├── display.py
    └── utils.py
```

---

## 🧪 Estructura de Tests

```
tests/
├── unit/
│   ├── test_components/
│   ├── test_utils/
│   └── test_config/
├── integration/
│   ├── test_pipeline/
│   └── test_deploy/
└── conftest.py
```

---

## 🛠️ Tecnologías Utilizadas

**Experimentación**

* Python 3.8+
* Jupyter
* Pandas / NumPy
* Scikit-learn
* Matplotlib / Seaborn / Plotly

**Productivización (Propuesta)**

* Kubeflow Pipelines
* Google Cloud Platform
* Vertex AI
* BigQuery
* Cloud Storage

