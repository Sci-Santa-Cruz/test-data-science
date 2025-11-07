import pandas as pd
import math
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
import plotly.io as pio

pio.renderers.default = "notebook"  # mostrar en el notebook

# --- 1) Matplotlib + Seaborn: Temperatura vs Luz ---
def plot_temperature_vs_light(df, cols=2, point_size=40, figsize_per_row=(6,4)):
    """
    Genera subplots de Temperatura vs Luz por día con colores fijos por clase.
    """
    df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
    df['day'] = df['timestamp'].dt.date
    unique_days = sorted(df['day'].dropna().unique())
    n_days = len(unique_days)

    import matplotlib.pyplot as plt
    import seaborn as sns

    rows = math.ceil(n_days / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(figsize_per_row[0]*cols, figsize_per_row[1]*rows), sharey=True)
    axes = axes.flatten()

    for i, day in enumerate(unique_days):
        subset = df[df['day'] == day]
        sns.scatterplot(
            data=subset,
            x='light_lux',
            y='temperature_celsius',
            hue='class',
            alpha=0.7,
            s=point_size,
            ax=axes[i]
        )
        axes[i].set_title(f'Temperatura vs Luz – {day}')
        axes[i].set_xlabel('Luz (lux)')
        axes[i].set_ylabel('Temperatura (°C)')
        axes[i].grid(True, alpha=0.3)

    for j in range(i+1, len(axes)):
        fig.delaxes(axes[j])

    fig.tight_layout()
    plt.show()


# --- 2) Plotly: Temperatura vs Luz matrix ---
def plot_temperature_vs_light_matrix_colored(df, cols=2, marker_size=8):
    """
    Muestra todos los días en una matriz de subplots.
    Temperatura vs luz, colores consistentes por clase usando Plotly default palette.
    Hover muestra hora:minuto:segundo.
    """
    df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
    df['day'] = df['timestamp'].dt.date
    df['hour_str'] = df['timestamp'].dt.strftime('%H:%M:%S')  # hover con hora:minuto:segundo
    unique_days = sorted(df['day'].dropna().unique())
    classes = sorted(df['class'].unique())
    n_days = len(unique_days)

    color_map = {c: px.colors.qualitative.Plotly[i % len(px.colors.qualitative.Plotly)]
                 for i, c in enumerate(classes)}

    rows = math.ceil(n_days / cols)
    fig = make_subplots(
        rows=rows,
        cols=cols,
        subplot_titles=[str(d) for d in unique_days]
    )

    for i, day in enumerate(unique_days):
        df_day = df[df['day'] == day]
        row = (i // cols) + 1
        col = (i % cols) + 1
        for c in classes:
            df_class = df_day[df_day['class'] == c]
            if not df_class.empty:
                fig.add_trace(
                    go.Scatter(
                        x=df_class['light_lux'],
                        y=df_class['temperature_celsius'],
                        mode='markers',
                        marker=dict(size=marker_size, color=color_map[c], line=dict(width=0.5, color='DarkSlateGrey')),
                        name=f'Clase {c}',
                        legendgroup=str(c),
                        showlegend=(i==0),
                        text=df_class['hour_str'],
                        hovertemplate='Hora: %{text}<br>Luz: %{x:.2f} lux<br>Temp: %{y:.2f} °C<br>Clase: %{marker.color}'
                    ),
                    row=row,
                    col=col
                )

    fig.update_layout(
        height=400*rows,
        width=1300,
        title_text='Temperatura vs Luz – Todos los días',
        title_x=0.5,
        legend_title="Clase",
        hovermode='closest'
    )

    for i in range(n_days):
        fig.update_xaxes(title_text="Luz (lux)", row=(i//cols)+1, col=(i%cols)+1)
        fig.update_yaxes(title_text="Temperatura (°C)", row=(i//cols)+1, col=(i%cols)+1)

    fig.show()


# --- 3) Plotly: Temperatura vs Hora matrix ---
def plot_temperature_vs_hour_matrix(df, cols=2, marker_size=8):
    """
    Matriz de subplots: temperatura vs hora del día para cada día.
    Colores consistentes por clase.
    Hover muestra hora:minuto:segundo.
    Puntos agrupados solo por hora.
    """
    df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
    df['day'] = df['timestamp'].dt.date
    df['hour_num'] = df['timestamp'].dt.hour  # solo horas enteras
    df['hour_str'] = df['timestamp'].dt.strftime('%H:%M:%S')  # hover con hora:minuto:segundo
    unique_days = sorted(df['day'].dropna().unique())
    classes = sorted(df['class'].unique())
    n_days = len(unique_days)

    color_map = {c: px.colors.qualitative.Plotly[i % len(px.colors.qualitative.Plotly)]
                 for i, c in enumerate(classes)}

    rows = math.ceil(n_days / cols)
    fig = make_subplots(
        rows=rows,
        cols=cols,
        subplot_titles=[str(d) for d in unique_days]
    )

    for i, day in enumerate(unique_days):
        df_day = df[df['day'] == day]
        row = (i // cols) + 1
        col = (i % cols) + 1
        for c in classes:
            df_class = df_day[df_day['class'] == c]
            if not df_class.empty:
                fig.add_trace(
                    go.Scatter(
                        x=df_class['hour_num'],
                        y=df_class['temperature_celsius'],
                        mode='markers',
                        marker=dict(size=marker_size, color=color_map[c], line=dict(width=0.5, color='DarkSlateGrey')),
                        name=f'Clase {c}',
                        legendgroup=str(c),
                        showlegend=(i==0),
                        text=df_class['hour_str'],
                        hovertemplate='Hora: %{text}<br>Temp: %{y:.2f} °C<br>Clase: %{marker.color}'
                    ),
                    row=row,
                    col=col
                )

    fig.update_layout(
        height=400*rows,
        width=1300,
        title_text='Temperatura vs Hora – Todos los días',
        title_x=0.5,
        legend_title="Clase",
        hovermode='closest'
    )

    for i in range(n_days):
        fig.update_xaxes(title_text="Hora del día", row=(i//cols)+1, col=(i%cols)+1)
        fig.update_yaxes(title_text="Temperatura (°C)", row=(i//cols)+1, col=(i%cols)+1)

    fig.show()



import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import math
import plotly.express as px
import plotly.io as pio

pio.renderers.default = "notebook"  # para mostrar en el notebook

def plot_co2_vs_hour_matrix(df, cols=2, marker_size=8):
    """
    Matriz de subplots: CO2 vs hora del día para cada día.
    Colores consistentes por clase.
    Hover muestra hora:minuto:segundo.
    Puntos agrupados por hora completa.
    
    Args:
        df (pd.DataFrame): columnas 'timestamp', 'co2_ppm', 'class'
        cols (int): número de columnas de la matriz
        marker_size (int): tamaño de los puntos
    """
    df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
    df['day'] = df['timestamp'].dt.date
    df['hour_num'] = df['timestamp'].dt.hour  # solo horas enteras
    df['hour_str'] = df['timestamp'].dt.strftime('%H:%M:%S')  # hover con hora:minuto:segundo
    unique_days = sorted(df['day'].dropna().unique())
    classes = sorted(df['class'].unique())
    n_days = len(unique_days)

    # Colores consistentes por clase
    color_map = {c: px.colors.qualitative.Plotly[i % len(px.colors.qualitative.Plotly)]
                 for i, c in enumerate(classes)}

    # Crear subplots
    rows = math.ceil(n_days / cols)
    fig = make_subplots(
        rows=rows,
        cols=cols,
        subplot_titles=[str(d) for d in unique_days]
    )

    # Agregar scatter por día y clase
    for i, day in enumerate(unique_days):
        df_day = df[df['day'] == day]
        row = (i // cols) + 1
        col = (i % cols) + 1
        for c in classes:
            df_class = df_day[df_day['class'] == c]
            if not df_class.empty:
                fig.add_trace(
                    go.Scatter(
                        x=df_class['hour_num'],
                        y=df_class['co2_ppm'],
                        mode='markers',
                        marker=dict(size=marker_size, color=color_map[c], line=dict(width=0.5, color='DarkSlateGrey')),
                        name=f'Clase {c}',
                        legendgroup=str(c),
                        showlegend=(i==0),
                        text=df_class['hour_str'],
                        hovertemplate='Hora: %{text}<br>CO2: %{y:.2f} ppm<br>Clase: %{marker.color}'
                    ),
                    row=row,
                    col=col
                )

    fig.update_layout(
        height=400*rows,
        width=1300,
        title_text='CO2 vs Hora – Todos los días',
        title_x=0.5,
        legend_title="Clase",
        hovermode='closest'
    )

    for i in range(n_days):
        fig.update_xaxes(title_text="Hora del día", row=(i//cols)+1, col=(i%cols)+1)
        fig.update_yaxes(title_text="CO2 (ppm)", row=(i//cols)+1, col=(i%cols)+1)

    fig.show()


import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

def boxplots_numeric_by_class(df, class_col='class', figsize=(12,8), dpi=90):
    """
    Crea boxplots de todas las columnas numéricas segmentadas por clase.
    
    Args:
        df (pd.DataFrame): DataFrame con variables numéricas y columna de clase.
        class_col (str): nombre de la columna que indica la clase.
        figsize (tuple): tamaño de la figura (ancho, alto).
        dpi (int): resolución de la figura.
    """
    numeric_cols = df.select_dtypes(include='number').columns.tolist()
    
    n_cols = 2  # número de columnas en la matriz de subplots
    n_rows = (len(numeric_cols) + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize, dpi=dpi)
    axes = axes.flatten()
    
    sns.set(style='darkgrid')
    
    for i, col in enumerate(numeric_cols):
        sns.boxplot(x=class_col, y=col, data=df, ax=axes[i])
        axes[i].set_title(f'{col} por {class_col}')
        axes[i].grid(True, alpha=0.3)
    
    # Eliminar ejes vacíos si existen
    for j in range(i+1, len(axes)):
        fig.delaxes(axes[j])
    
    fig.tight_layout()
    plt.show()


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler

def boxplots_numeric_minmax_by_class(df, class_col='class', figsize=(12,8), dpi=90):
    """
    Crea boxplots de todas las columnas numéricas segmentadas por clase,
    aplicando MinMaxScaler a los valores.
    
    Args:
        df (pd.DataFrame): DataFrame con variables numéricas y columna de clase.
        class_col (str): nombre de la columna que indica la clase.
        figsize (tuple): tamaño de la figura (ancho, alto).
        dpi (int): resolución de la figura.
    """
    numeric_cols = df.select_dtypes(include='number').columns.tolist()
    if class_col in numeric_cols:
        numeric_cols.remove(class_col)
    
    df_scaled = df.copy()
    scaler = MinMaxScaler()
    
    # Escalar solo columnas numéricas
    df_scaled[numeric_cols] = scaler.fit_transform(df_scaled[numeric_cols])
    
    n_cols = 2  # número de columnas en la matriz de subplots
    n_rows = (len(numeric_cols) + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize, dpi=dpi)
    axes = axes.flatten()
    
    sns.set(style='darkgrid')
    
    for i, col in enumerate(numeric_cols):
        sns.boxplot(x=class_col, y=col, data=df_scaled, ax=axes[i])
        axes[i].set_title(f'{col} (MinMax) por {class_col}')
        axes[i].grid(True, alpha=0.3)
    
    # Eliminar ejes vacíos si existen
    for j in range(i+1, len(axes)):
        fig.delaxes(axes[j])
    
    fig.tight_layout()
    plt.show()


import pandas as pd
import math
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px

def plot_scatter_matrix_by_day(
    df,
    x_col,
    y_col,
    class_col='class',
    hover_col='timestamp',
    cols=2,
    marker_size=8
):
    """
    Muestra todos los días en una matriz de subplots.
    X vs Y, colores consistentes por clase.
    Hover muestra el contenido de hover_col.
    
    Parámetros:
    - df: DataFrame con los datos.
    - x_col, y_col: columnas a graficar en los ejes.
    - class_col: columna que indica la clase.
    - hover_col: columna que se mostrará al pasar el mouse.
    - cols: cantidad de subplots por fila.
    - marker_size: tamaño de los marcadores.
    """
    
    df = df.copy()
    df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
    df['day'] = df['timestamp'].dt.date
    df['hover_text'] = df[hover_col].astype(str)
    
    unique_days = sorted(df['day'].dropna().unique())
    classes = sorted(df[class_col].unique())
    n_days = len(unique_days)
    
    color_map = {c: px.colors.qualitative.Plotly[i % len(px.colors.qualitative.Plotly)]
                 for i, c in enumerate(classes)}
    
    rows = math.ceil(n_days / cols)
    fig = make_subplots(
        rows=rows,
        cols=cols,
        subplot_titles=[str(d) for d in unique_days]
    )
    
    for i, day in enumerate(unique_days):
        df_day = df[df['day'] == day]
        row = (i // cols) + 1
        col = (i % cols) + 1
        for c in classes:
            df_class = df_day[df_day[class_col] == c]
            if not df_class.empty:
                hovertemplate = (
                    '%{text}<br>' +
                    x_col + ': %{x}<br>' +
                    y_col + ': %{y}<br>' +
                    'Clase: ' + str(c)
                )
                fig.add_trace(
                    go.Scatter(
                        x=df_class[x_col],
                        y=df_class[y_col],
                        mode='markers',
                        marker=dict(size=marker_size, color=color_map[c], line=dict(width=0.5, color='DarkSlateGrey')),
                        name=f'Clase {c}',
                        legendgroup=str(c),
                        showlegend=(i==0),
                        text=df_class['hover_text'],
                        hovertemplate=hovertemplate
                    ),
                    row=row,
                    col=col
                )
    
    fig.update_layout(
        height=400*rows,
        width=1300,
        title_text=f'{y_col} vs {x_col} – Todos los días',
        title_x=0.5,
        legend_title="Clase",
        hovermode='closest'
    )
    
    for i in range(n_days):
        fig.update_xaxes(title_text=x_col, row=(i//cols)+1, col=(i%cols)+1)
        fig.update_yaxes(title_text=y_col, row=(i//cols)+1, col=(i%cols)+1)
    
    fig.show()
