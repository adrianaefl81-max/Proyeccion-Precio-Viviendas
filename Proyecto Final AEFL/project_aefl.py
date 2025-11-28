import pandas as pd
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.ticker as mticker # Importar matplotlib.ticker para formato de ejes

# Configuración de la página
st.set_page_config(page_title="Predicción de Precios de Viviendas", layout="wide")

# Cargar el dataset y realizar preprocesamiento
@st.cache_data
def load_data():
    data = pd.read_excel('Housing.xlsx')
    
    # --- Guardar una copia del DF original para gráficos que usen categorías no one-hot ---
    # Esto es para evitar re-leer el Excel múltiples veces para los gráficos de dispersión y cajas
    # si 'Estado de Mobiliario' es la categoría seleccionada para el 'hue' o el 'boxplot'.
    # st.session_state es ideal para almacenar esto entre reruns de Streamlit.
    if 'original_df_for_plotting' not in st.session_state:
        st.session_state.original_df_for_plotting = data.copy()
    
    # Preprocesamiento para convertir columnas binarias a numéricas (0/1)
    # 'Estacionamiento' ha sido eliminado de esta lista, ya que es una columna numérica.
    # Estas columnas siguen siendo preprocesadas para que puedan usarse en los gráficos exploratorios.
    binary_cols = ['Calle Principal', 'Habitacion invitados', 'Sotano', 'Calefaccion Agua', 'Aire Acondicionado', 'Casa prefabricada']
    
    for col in binary_cols:
        if col in data.columns:
            mapped_series = data[col].map({'yes': 1, 'no': 0})
            data[col] = mapped_series.fillna(0).astype(int)
    
    # --- Procesamiento específico para 'Estacionamiento' (ya que es numérica: 0, 1, 2, 3) ---
    if 'Estacionamiento' in data.columns: # Asegurarse de que la columna existe
        data['Estacionamiento'] = pd.to_numeric(data['Estacionamiento'], errors='coerce')
        data['Estacionamiento'] = data['Estacionamiento'].fillna(0).astype(int)
    
    # Aplicar One-Hot Encoding para 'Estado de Mobiliario'
    # Esta transformación sigue ocurriendo en el dataframe 'df', para que pueda usarse en los gráficos exploratorios.
    if 'Estado de Mobiliario' in data.columns:
        data['Estado de Mobiliario'] = data['Estado de Mobiliario'].replace({
            'semi-furnished': 'semifurnished',
            'unfurnished': 'unfurnished',
            'furnished': 'furnished'
        })
        furnishing_col_name = 'Estado de Mobiliario'
        data = pd.get_dummies(data, columns=[furnishing_col_name], drop_first=True, prefix=furnishing_col_name)
    
    return data

df = load_data()

# --- Formato Monetario para los Ejes ---
monetary_formatter = mticker.FuncFormatter(lambda x, p: f'${x:,.0f}')


# --- Gráfica de Distribución de Precios de Viviendas (Histograma) ---
st.subheader("Distribución de Precios de Viviendas")
fig_hist_price, ax_hist_price = plt.subplots(figsize=(10, 5))
sns.histplot(df['Precio de Venta'], bins=30, kde=True, ax=ax_hist_price)
ax_hist_price.set_title("Distribución de Precio de Venta")
ax_hist_price.set_xlabel("Precio de Venta")
ax_hist_price.set_ylabel("Cantidad")
ax_hist_price.xaxis.set_major_formatter(monetary_formatter)
st.pyplot(fig_hist_price)


# --- Gráfico de Dispersión Dinámico ---
st.subheader("Análisis de Relación Dinámica: Precio de Venta vs. Características Numéricas")

# Opciones para el eje X (columnas numéricas) - Incluido 'Estacionamiento'
numeric_cols = ['Area', 'Habitaciones', 'Baños', 'Referencias', 'Estacionamiento']
x_axis_selection = st.selectbox("Selecciona la característica numérica para el Eje X:", numeric_cols)

# Opciones para el color (hue) (columnas categóricas)
categorical_cols_original = ['Calle Principal', 'Habitacion invitados', 'Sotano', 'Calefaccion Agua', 'Aire Acondicionado', 'Casa prefabricada', 'Estado de Mobiliario']
hue_selection = st.selectbox("Selecciona una característica categórica para colorear los puntos:", ['Ninguno'] + categorical_cols_original)

fig_scatter, ax_scatter = plt.subplots(figsize=(12, 6))
if hue_selection == 'Ninguno':
    sns.scatterplot(x=df[x_axis_selection], y=df['Precio de Venta'], ax=ax_scatter, alpha=0.6)
else:
    original_df_for_plotting = st.session_state.original_df_for_plotting
    sns.scatterplot(x=df[x_axis_selection], y=df['Precio de Venta'], hue=original_df_for_plotting[hue_selection], ax=ax_scatter, alpha=0.6)

ax_scatter.set_title(f"Precio de Venta vs. {x_axis_selection}")
ax_scatter.set_xlabel(x_axis_selection)
ax_scatter.set_ylabel("Precio de Venta")
ax_scatter.yaxis.set_major_formatter(monetary_formatter)
st.pyplot(fig_scatter)

# --- Gráfico de Cajas Dinámico ---
st.subheader("Distribución de Precios por Categoría")

boxplot_category_selection = st.selectbox("Selecciona una característica categórica para el Box Plot:", categorical_cols_original)

fig_boxplot, ax_boxplot = plt.subplots(figsize=(12, 6))

original_df_for_plotting = st.session_state.original_df_for_plotting
sns.boxplot(x=original_df_for_plotting[boxplot_category_selection], y=df['Precio de Venta'], ax=ax_boxplot)

ax_boxplot.set_title(f"Distribución de Precio de Venta por {boxplot_category_selection}")
ax_boxplot.set_xlabel(boxplot_category_selection)
ax_boxplot.set_ylabel("Precio de Venta")
ax_boxplot.yaxis.set_major_formatter(monetary_formatter)
if len(original_df_for_plotting[boxplot_category_selection].unique()) > 2:
    plt.xticks(rotation=45, ha='right')
st.pyplot(fig_boxplot)


# --- Mapa de Calor de Correlación ---
st.subheader("Mapa de Calor de Correlación entre Características Seleccionadas")

# Lista de todas las columnas que el modelo usa (incluye las one-hot encoded)
# Y 'Precio de Venta', incluido 'Estacionamiento'
all_model_features_and_target = ['Precio de Venta'] + [col for col in df.columns if col not in ['Precio de Venta']]

# Permite al usuario seleccionar qué columnas quiere ver en el heatmap
selected_heatmap_cols = st.multiselect(
    "Selecciona las características para el mapa de calor:",
    options=all_model_features_and_target,
    default=all_model_features_and_target # Por defecto, selecciona todas las válidas
)

# Lógica de renderizado mejorada para el mapa de calor
if len(selected_heatmap_cols) >= 2:
    corr_matrix = df[selected_heatmap_cols].corr()
    fig_heatmap, ax_heatmap = plt.subplots(figsize=(12, 10))
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=.5, ax=ax_heatmap)
    ax_heatmap.set_title("Mapa de Calor de Correlación de Características Seleccionadas")
    st.pyplot(fig_heatmap)
elif len(selected_heatmap_cols) == 1:
    st.info("Por favor, selecciona al menos dos características para calcular y mostrar un mapa de calor de correlación significativo.")
else:
    st.write("Por favor, selecciona al menos una característica para comenzar a configurar el mapa de calor.")


# --- Preparación de datos para el modelo (X e Y mejorados) ---
# Definir la lista de características a usar en el modelo.
# Se han eliminado 'Referencias' y las columnas de 'Estado de Mobiliario'
# También se eliminan las características binarias de esta lista
feature_cols = [
    'Area', 'Habitaciones', 'Baños', 'Estacionamiento'
    # Las características binarias 'Calle Principal', 'Habitacion invitados', etc.,
    # y las de 'Estado de Mobiliario' han sido removidas de feature_cols
]

# Filtrar feature_cols para asegurar que solo incluimos las que realmente existen en df
feature_cols_final = [col for col in feature_cols if col in df.columns]

X = df[feature_cols_final]
y = df['Precio de Venta']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Entrenamiento del modelo
model = RandomForestRegressor(random_state=42)
model.fit(X_train, y_train)

# --- Predicción Interactiva (con las características deseadas) ---
st.sidebar.header("Parámetros de Entrada para Predicción")

# Sliders para las características numéricas
min_rooms, max_rooms = int(df['Habitaciones'].min()), int(df['Habitaciones'].max())
default_rooms = int(df['Habitaciones'].mean())
min_area, max_area = int(df['Area'].min()), int(df['Area'].max())
default_area = int(df['Area'].mean())
min_bath, max_bath = int(df['Baños'].min()), int(df['Baños'].max())
default_bath = int(df['Baños'].mean())
# 'Referencias' (stories) ha sido eliminado de los inputs del sidebar
min_estacionamiento, max_estacionamiento = int(df['Estacionamiento'].min()), int(df['Estacionamiento'].max())
default_estacionamiento = int(df['Estacionamiento'].mean())

num_rooms_input = st.sidebar.slider("Número de Habitaciones:", min_rooms, max_rooms, default_rooms)
area_input = st.sidebar.slider("Área (en pies cuadrados):", min_area, max_area, default_area)
bathrooms_input = st.sidebar.slider("Número de Baños:", min_bath, max_bath, default_bath)
estacionamiento_input = st.sidebar.slider("Plazas de Estacionamiento:", min_estacionamiento, max_estacionamiento, default_estacionamiento)


# Los checkboxes para las características binarias han sido eliminados de los inputs del sidebar
# 'Estado de Mobiliario' ha sido eliminado de los inputs del sidebar

# Construir el DataFrame de entrada para la predicción
# Debe tener las mismas columnas y en el mismo orden que X_train
input_data = pd.DataFrame(np.zeros((1, len(feature_cols_final))), columns=feature_cols_final)

# Asignar valores a las columnas numéricas
input_data['Area'] = area_input
input_data['Habitaciones'] = num_rooms_input
input_data['Baños'] = bathrooms_input
input_data['Estacionamiento'] = estacionamiento_input

# Las columnas binarias y de Estado de Mobiliario ya no están en feature_cols_final, por lo tanto no se asignan.


# Predicción
predicted_price = model.predict(input_data)

# --- Resultados de la predicción ---
st.subheader("Predicción de Precio de Vivienda")
st.write("Datos de entrada:")
st.write(f"Número de Habitaciones: {num_rooms_input}")
st.write(f"Área: {area_input} pies cuadrados")
st.write(f"Número de Baños: {bathrooms_input}")
st.write(f"Plazas de Estacionamiento: {estacionamiento_input}")
# Los st.write para las características binarias y Estado de Mobiliario han sido eliminados
st.write(f"**Precio Predicho: {predicted_price[0]:,.2f}**")

# --- Evaluación del Modelo ---
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

st.subheader("Evaluación del Modelo")
st.write(f"Error Cuadrático Medio (MSE): {mse:.2f}")
st.write(f"Coeficiente de Determinación (R²): {r2:.2f}")
st.write(f"Raíz del Error Cuadrático Medio (RMSE): {np.sqrt(mse):,.2f}")