import numpy as np
import pandas as pd
import pickle


def remove_outliers_sigma(df: pd.DataFrame, variable: str, upper: float, lower: float) -> pd.DataFrame:
    """Elimina los registros fuera de los límites especificados para una variable numérica.

    Args:
        df (pd.DataFrame): DataFrame que contiene los datos.
        variable (str): Nombre de la variable sobre la que se eliminarán los outliers.
        upper (float): Límite superior permitido.
        lower (float): Límite inferior permitido.

    Returns:
        pd.DataFrame: DataFrame filtrado sin los registros fuera de los límites.
    """
    return df[(df[variable] >= lower) & (df[variable] <= upper)]


def compute_outlier_bounds(df: pd.DataFrame, variables: list[str], num_sigma: int = 2) -> dict:
    """Calcula los límites de outliers para cada variable numérica basada en desviaciones estándar.

    Args:
        df (pd.DataFrame): DataFrame que contiene las variables numéricas.
        variables (list[str]): Lista de nombres de columnas a evaluar.
        num_sigma (int, optional): Número de desviaciones estándar para definir los límites. Por defecto 2.

    Returns:
        dict: Diccionario con los límites inferior y superior para cada variable.
            Ejemplo:
            {
                'variable1': {'lower': valor_inferior, 'upper': valor_superior},
                'variable2': {'lower': valor_inferior, 'upper': valor_superior}
            }
    """
    bounds = {}
    for var in variables:
        mean = np.mean(df[var])
        std = np.std(df[var])
        lower = mean - num_sigma * std
        upper = mean + num_sigma * std
        bounds[var] = {'lower': lower, 'upper': upper}
    return bounds


def cumulative_classification_lift(y_true: np.ndarray, y_prob: np.ndarray) -> np.ndarray:
    """Calcula el lift acumulativo para un modelo de clasificación.

    Args:
        y_true (np.ndarray): Vector con las etiquetas verdaderas (0 o 1).
        y_prob (np.ndarray): Vector con las probabilidades predichas por el modelo.

    Returns:
        np.ndarray: Lift acumulativo, donde cada valor representa la ganancia relativa
        del modelo respecto a una selección aleatoria de igual tamaño.
    """
    sorted_indices = np.argsort(y_prob)[::-1]
    y_true_sorted = y_true[sorted_indices]
    n = len(y_true)
    cumulative_positive = np.cumsum(y_true_sorted)
    cumulative_positive_percentage = cumulative_positive / np.arange(1, n + 1)
    overall_positive_percentage = np.sum(y_true) / n
    return cumulative_positive_percentage / overall_positive_percentage
