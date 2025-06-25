import streamlit as st
import pandas as pd
import altair as alt
import requests
from io import StringIO

# Cargar datos desde URL
@st.cache_data
def load_data():
    try:
        # URL del dataset simulado mejorado
        url = "https://hebbkx1anhila5yf.public.blob.vercel-storage.com/predicciones_alerta_vih_2025_2030_simulado-sJhB4luINPdUPcTCLKSgmVsfrVb0ui.csv"
        
        # Cargar desde URL
        response = requests.get(url)
        response.raise_for_status()
        df_pred = pd.read_csv(StringIO(response.text))
        
        # Corregir tipos de datos
        df_pred['Anio'] = pd.to_numeric(df_pred['Anio'], errors='coerce')
        df_pred['CasosEstimados_Predichos'] = pd.to_numeric(df_pred['CasosEstimados_Predichos'], errors='coerce')
        df_pred['PromHist'] = pd.to_numeric(df_pred['PromHist'], errors='coerce')
        df_pred['Alerta'] = df_pred['Alerta'].map({'True': True, 'False': False, True: True, False: False})
        
        # Limpiar datos nulos
        df_pred = df_pred.dropna()
        
        # Intentar cargar datos históricos (opcional)
        try:
            df_hist = pd.read_csv('DATASET_VIH.csv')
        except:
            # Crear datos históricos de ejemplo si no existen
            df_hist = pd.DataFrame()
        
        return df_pred, df_hist
        
    except Exception as e:
        st.error(f"Error cargando datos desde URL: {e}")
        # Fallback: crear datos de ejemplo
        return pd.DataFrame(), pd.DataFrame()

df_pred, df_hist = load_data()

# Verificar que los datos se cargaron
if df_pred.empty:
    st.error("❌ No se pudieron cargar los datos.")
    st.stop()

# Configuración de la página
st.set_page_config(
    page_title="Predicción y Alerta de VIH en Perú",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Título principal
st.markdown(
    """
    # 🦠 Predicción y Alerta de Casos de VIH en Perú
    Consulta los **casos estimados y predichos de VIH** por Departamento, Sexo y Año.
    También puedes ver si hay una **alerta** respecto al promedio histórico.
    """,
    unsafe_allow_html=True
)

# --- Barra lateral de filtros ---
st.sidebar.header("Filtros de Consulta")

# Obtener opciones únicas
available_years = sorted(df_pred['Anio'].unique())
available_departments = sorted(df_pred['Departamento'].unique())
available_sex = sorted(df_pred['Sexo'].unique())

# Filtros interactivos
year = st.sidebar.selectbox(
    "Año",
    options=available_years,
    index=0
)

departamento = st.sidebar.selectbox(
    "Departamento",
    options=available_departments,
    index=0
)

sexo = st.sidebar.selectbox(
    "Sexo",
    options=available_sex,
    index=0
)

tipo_grafico = st.sidebar.radio(
    "Tipo de gráfico:",
    options=["Barras", "Líneas", "Área"],
    index=0
)

# --- Función para obtener datos exactos ---
def get_exact_data(year, departamento, sexo):
    """Obtiene los valores exactos para la combinación seleccionada"""
    mask = (
        (df_pred['Anio'] == year) &
        (df_pred['Departamento'] == departamento) &
        (df_pred['Sexo'] == sexo)
    )
    exact_data = df_pred[mask]
    
    if not exact_data.empty:
        return {
            'casos_pred': int(exact_data['CasosEstimados_Predichos'].iloc[0]),
            'prom_hist': float(exact_data['PromHist'].iloc[0]),
            'alerta': bool(exact_data['Alerta'].iloc[0])
        }
    return None

# Obtener valores actuales
current_values = get_exact_data(year, departamento, sexo)

# --- Mostrar resultados ---
if current_values:
    st.subheader(f"Resultados para {departamento} - {sexo} - {year}")
    
    # Mostrar valores principales
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Casos estimados predichos", current_values['casos_pred'])
    with col2:
        st.metric("Promedio histórico", f"{current_values['prom_hist']:.1f}")

    # Mostrar alerta
    if current_values['alerta']:
        st.error("⚠️ ¡Alerta! El valor predicho está fuera del rango histórico.", icon="🚨")
    else:
        st.success("✅ Sin alerta. El valor predicho está dentro del rango histórico.", icon="✅")

    # --- Gráficos ---
    st.markdown("---")
    st.subheader(f"Visualización: {tipo_grafico}")

    # Preparar datos para gráficos
    # Datos históricos (si existen)
    if not df_hist.empty:
        df_hist_filtrado = df_hist[
            (df_hist['Departamento'] == departamento) &
            (df_hist['Sexo'] == sexo)
        ][['Anio', 'CasosEstimados']].rename(columns={'CasosEstimados': 'Casos'})
    else:
        df_hist_filtrado = pd.DataFrame()
    
    # Datos de predicción (2025-2030)
    df_pred_filtrado = df_pred[
        (df_pred['Departamento'] == departamento) &
        (df_pred['Sexo'] == sexo)
    ][['Anio', 'CasosEstimados_Predichos']].rename(columns={'CasosEstimados_Predichos': 'Casos'})
    
    # Combinar datos históricos y de predicción
    df_completo = pd.concat([df_hist_filtrado, df_pred_filtrado]).sort_values('Anio')
    df_completo = df_completo[df_completo['Anio'] <= year]  # Filtrar hasta el año seleccionado
    df_completo['Tipo'] = df_completo['Anio'].apply(lambda x: 'Histórico' if x <= 2024 else 'Predicción')

    # Gráfico de Barras (comparación año actual)
    if tipo_grafico == "Barras":
        chart_data = pd.DataFrame({
            'Tipo': ['Promedio histórico', 'Predicción'],
            'Valor': [current_values['prom_hist'], current_values['casos_pred']]
        })
        chart = alt.Chart(chart_data).mark_bar().encode(
            x='Tipo',
            y='Valor',
            color=alt.Color('Tipo', scale=alt.Scale(range=["#1f77b4", "#ff7f0e"]))
        ).properties(
            title=f"Comparación para {year}",
            width=600,
            height=400
        )

    # Gráfico de Líneas (evolución)
    elif tipo_grafico == "Líneas":
        chart = alt.Chart(df_completo).mark_line(point=True).encode(
            x=alt.X('Anio:O', title='Año'),
            y=alt.Y('Casos:Q', title='Casos'),
            color=alt.Color('Tipo:N', scale=alt.Scale(domain=['Histórico', 'Predicción'], 
                          range=['#1f77b4', '#d62728'])),
            tooltip=['Anio', 'Casos', 'Tipo']
        ).properties(
            title=f"Evolución hasta {year}",
            width=600,
            height=400
        )

    # Gráfico de Área
    else:
        chart = alt.Chart(df_completo).mark_area(opacity=0.7).encode(
            x=alt.X('Anio:O', title='Año'),
            y=alt.Y('Casos:Q', title='Casos'),
            color=alt.Color('Tipo:N', scale=alt.Scale(domain=['Histórico', 'Predicción'], 
                          range=['#1f77b4', '#d62728'])),
            tooltip=['Anio', 'Casos', 'Tipo']
        ).properties(
            title=f"Tendencia hasta {year}",
            width=600,
            height=400
        )

    st.altair_chart(chart, use_container_width=True)

    # --- Tabla de datos ---
    st.markdown("---")
    st.subheader("Datos completos")
    st.dataframe(df_completo.rename(columns={
        'Anio': 'Año',
        'Casos': 'Casos reportados/predichos'
    }))

else:
    st.warning("No hay datos para la combinación seleccionada.")

# Pie de página
st.markdown("---")
st.markdown(
    """
    <small>
    Desarrollado con Streamlit para el Proyecto de Aprendizaje Estadístico sobre VIH.<br>
    Inspirado en la Sala Situacional VIH del MINSA Perú.
    </small>
    """,
    unsafe_allow_html=True
)
