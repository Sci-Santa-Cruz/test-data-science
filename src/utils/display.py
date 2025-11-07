from IPython.display import HTML
import matplotlib.pyplot as plt
import numpy as np

def horizontal(dfs):
    html = '<div style="display:flex">'
    for df in dfs:
        html += '<div style="margin-right: 32px">'
        html += df.to_html()
        html += '</div>'
    html += '</div>'
    display(HTML(html))



def plot_cumulative_lift(y_true, y_prob, title='Cumulative Lift Curve'):
    """
    Calcula y grafica la curva de Lift Acumulado para un modelo de clasificación binaria.
    
    Parámetros
    ----------
    y_true : array-like
        Etiquetas verdaderas (0 o 1), puede ser pandas.Series, categórica o booleana.
    y_prob : array-like
        Probabilidades predichas para la clase positiva.
    title : str, opcional
        Título del gráfico.
    
    Retorna
    -------
    cumulative_lift_values : np.ndarray
        Valores del lift acumulado calculados.
    """
    # Conversión segura a valores numéricos
    y_true = np.asarray(y_true)
    if y_true.dtype == 'bool':
        y_true = y_true.astype(int)
    elif np.issubdtype(y_true.dtype, np.number) is False:
        try:
            y_true = y_true.astype(int)
        except:
            raise ValueError("y_true debe ser convertible a valores 0/1 numéricos.")
    
    y_prob = np.asarray(y_prob, dtype=float)
    
    # Ordenar por probabilidad descendente
    sorted_indices = np.argsort(y_prob)[::-1]
    y_true_sorted = y_true[sorted_indices]
    
    n = len(y_true)
    cumulative_positive = np.cumsum(y_true_sorted)
    
    # Porcentaje acumulado de positivos
    cumulative_positive_percentage = cumulative_positive / np.arange(1, n + 1)
    overall_positive_percentage = np.sum(y_true) / n
    
    cumulative_lift_values = cumulative_positive_percentage / overall_positive_percentage
    
    # Gráfico
    plt.figure(figsize=(8, 6))
    plt.plot(np.arange(1, n + 1) / n,
             cumulative_lift_values,
             marker='o', linestyle='-', color='b')
    plt.axhline(y=1, color='red', linestyle='--')
    plt.xlabel('Proportion of Samples')
    plt.ylabel('Cumulative Lift')
    plt.title(title)
    plt.legend(['Model', 'Random'])
    plt.grid(True)
    plt.show()
    
    return cumulative_lift_values

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def plot_curve_and_decile_bars(y_true, y_prob, n_deciles=10,
                               figsize=(10,6), show_per_decile=False):
    """
    Dibuja la curva de cumulative lift y barras alineadas en los cortes por decil.

    - Las barras representan el CUMULATIVE LIFT en cada corte 10%,20%,... (alineadas con la curva).
    - Si show_per_decile=True, la función añade en la tabla el Lift por decil (no acumulado).
    
    Args:
        y_true: array-like (0/1)
        y_prob: array-like (probabilidades)
        n_deciles: int, número de cortes (por defecto 10)
        figsize: tupla tamaño figura
        show_per_decile: bool, si True calcula y añade lift por decil en la tabla
    
    Returns:
        decile_df: pd.DataFrame con columnas:
            Decil, Total, Positivos, Tasa, Lift (por decil, si show_per_decile=True),
            Cumulative_Positive_pct, Cumulative_Lift
        cumulative_lift_values: np.ndarray (valores punto-a-punto de la curva)
    """
    # --- Validaciones y conversiones ---
    y_true = np.asarray(y_true)
    if y_true.dtype == bool:
        y_true = y_true.astype(int)
    if not np.issubdtype(y_true.dtype, np.number):
        try:
            y_true = y_true.astype(int)
        except Exception:
            raise ValueError("y_true debe ser convertible a 0/1.")
    y_prob = np.asarray(y_prob, dtype=float)
    if len(y_true) != len(y_prob):
        raise ValueError("y_true y y_prob deben tener la misma longitud.")
    n = len(y_true)
    if n == 0:
        raise ValueError("Vectores vacíos.")
    
    # --- Ordenar por prob descendente y calcular cumulative lift punto-a-punto ---
    order = np.argsort(y_prob)[::-1]
    y_true_sorted = y_true[order]
    y_prob_sorted = y_prob[order]
    
    cumulative_positive = np.cumsum(y_true_sorted)                        # positivos acumulados
    cumulative_positive_pct = cumulative_positive / np.arange(1, n+1)     # % positivos acumulados (porcentaje local)
    overall_positive_pct = np.sum(y_true) / n
    cumulative_lift_values = np.zeros_like(cumulative_positive_pct)
    if overall_positive_pct > 0:
        cumulative_lift_values = cumulative_positive_pct / overall_positive_pct
    else:
        cumulative_lift_values[:] = 0.0
    
    # --- Índices de corte para los deciles (10%,20%,...,100%)
    # usamos ceil para asegurar que el primer corte no sea cero en pequeños n
    cut_positions = (np.ceil(np.linspace(n / n_deciles, n, n_deciles))).astype(int) - 1
    cut_positions = np.clip(cut_positions, 0, n-1)
    decile_percent = (np.arange(1, n_deciles+1) * 100 / n_deciles)
    
    # Valores acumulados (percent y lift) en cada corte
    cumulative_positive_at_cut = cumulative_positive[cut_positions]
    cumulative_positive_pct_at_cut = cumulative_positive_at_cut / (cut_positions + 1)
    cumulative_lift_at_cut = np.zeros_like(cumulative_positive_pct_at_cut, dtype=float)
    if overall_positive_pct > 0:
        cumulative_lift_at_cut = cumulative_positive_pct_at_cut / overall_positive_pct
    
    # --- Tabla por decil (por tamaño igual) para métricas por-decil ---
    # construir deciles en base a ranking posicional para evitar problemas de qcut con empates
    df = pd.DataFrame({'y_true': y_true, 'y_prob': y_prob})
    df = df.sort_values('y_prob', ascending=False).reset_index(drop=True)
    df['rank'] = np.arange(1, n+1)
    df['Decil'] = pd.qcut(df['rank'], q=n_deciles, labels=False, duplicates='drop') + 1
    decile_summary = (
        df.groupby('Decil', observed=True)
          .agg(Total=('y_true', 'count'),
               Positivos=('y_true', 'sum'))
          .reset_index()
    )
    decile_summary['Tasa'] = decile_summary['Positivos'] / decile_summary['Total']
    tasa_global = df['y_true'].mean()
    decile_summary['Lift'] = (decile_summary['Tasa'] / tasa_global).replace([np.inf, -np.inf], np.nan).fillna(0) \
                            if tasa_global > 0 else 0.0
    
    # --- Agregar columnas acumuladas a la tabla: usar los cortes definidos arriba ---
    decile_df = decile_summary.copy()
    # Para cada decil i, calcular cumulative totals hasta ese decil (usando la tabla posicional)
    decile_df = decile_df.sort_values('Decil').reset_index(drop=True)
    decile_df['Cumulative_Total'] = decile_df['Total'].cumsum()
    decile_df['Cumulative_Positivos'] = decile_df['Positivos'].cumsum()
    decile_df['Cumulative_Positive_pct'] = decile_df['Cumulative_Positivos'] / decile_df['Cumulative_Total']
    decile_df['Cumulative_Lift'] = (decile_df['Cumulative_Positive_pct'] / overall_positive_pct).replace([np.inf, -np.inf], np.nan).fillna(0) \
                                   if overall_positive_pct > 0 else 0.0
    
    # --- Plot: curva completa + barras en cortes de decil (cumulative lift at cut) ---
    fig, ax = plt.subplots(figsize=figsize)
    x_curve = np.arange(1, n+1) / n
    ax.plot(x_curve, cumulative_lift_values, marker='o', markersize=4, linewidth=1.5, label='Cumulative Lift (curve)')
    # barras en posiciones de decil (usar proporción de muestras en x)
    x_bars = (cut_positions + 1) / n
    bar_width = 1.0/n_deciles * 0.8  # visual width
    ax.bar(x_bars, cumulative_lift_at_cut, width=bar_width, alpha=0.6, edgecolor='black',
           label=f'Cumulative Lift @ {int(100/n_deciles)}% steps')
    ax.axhline(1.0, color='red', linestyle='--', linewidth=1.2, label='Random (Lift=1)')
    ax.set_xlabel('Proporción de muestras')
    ax.set_ylabel('Lift acumulado')
    ax.set_title('Curva de Cumulative Lift + Barras en Deciles')
    ax.legend()
    ax.grid(axis='y', linestyle='--', alpha=0.6)
    
    # Anotar valores encima de barras
    for xi, val, pct in zip(x_bars, cumulative_lift_at_cut, decile_percent):
        ax.text(xi, val + 0.03 * max(1, cumulative_lift_at_cut.max()), f"{val:.2f}\n({int(pct)}%)",
                ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    plt.show()
    
    # Devolver tabla con información y el vector de la curva
    # Añadimos columnas que muestran los cortes exactos (cumulative at cut) para verificación
    cuts_df = pd.DataFrame({
        'Decil': np.arange(1, n_deciles+1),
        'Proportion': decile_percent,
        'Cut_Position': cut_positions + 1,
        'Cumulative_Positivos_at_cut': cumulative_positive_at_cut,
        'Cumulative_Positive_pct_at_cut': cumulative_positive_pct_at_cut,
        'Cumulative_Lift_at_cut': cumulative_lift_at_cut
    })
    # Unir cuts_df con decile_df por Decil (ambos 1..n_deciles)
    result_df = decile_df.merge(cuts_df, on='Decil', how='left')
    
    return result_df, cumulative_lift_values

