import streamlit as st
import pandas as pd
import altair as alt

# Cargar datos sin caché para forzar actualización
def load_data():
    df_pred = pd.read_csv('predicciones_alerta_vih_2025_2030.csv')
    df_hist = pd.read_csv('DATASET_VIH.csv')
    return df_pred, df_hist

df_pred, df_hist = load_data()

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

# --- Función para obtener datos EXACTOS del año seleccionado ---
def get_exact_data(year, departamento, sexo):
    """Obtiene los valores PRECISOS para el año, departamento y sexo seleccionados"""
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
            'alerta': exact_data['Alerta'].iloc[0]
        }
    return None

# Obtener datos exactos para la selección actual
current_data = get_exact_data(year, departamento, sexo)

# --- Mostrar resultados ---
if current_data:
    st.subheader(f"Resultados para {departamento} - {sexo} - {year}")
    st.markdown(f"**Casos estimados predichos:** `{current_data['casos_pred']}`  \n**Promedio histórico:** `{current_data['prom_hist']:.1f}`")

    # Mostrar alerta
    if current_data['alerta']:
        st.error("⚠️ ¡Alerta! El valor predicho está fuera del rango histórico.", icon="🚨")
    else:
        st.success("✅ Sin alerta. El valor predicho está dentro del rango histórico.", icon="✅")

    # Métricas
    col1, col2 = st.columns(2)
    col1.metric("Casos predichos", current_data['casos_pred'])
    col2.metric("Promedio histórico", f"{current_data['prom_hist']:.1f}")

    # --- Gráficos ---
    st.markdown("---")
    st.subheader(f"Visualización: {tipo_grafico}")

    # Preparar datos combinados (históricos + predicciones)
    df_hist_filtrado = df_hist[
        (df_hist['Departamento'] == departamento) &
        (df_hist['Sexo'] == sexo)
    ][['Anio', 'CasosEstimados']].rename(columns={'CasosEstimados': 'Casos'})
    
    df_pred_filtrado = df_pred[
        (df_pred['Departamento'] == departamento) &
        (df_pred['Sexo'] == sexo)
    ][['Anio', 'CasosEstimados_Predichos']].rename(columns={'CasosEstimados_Predichos': 'Casos'})
    
    df_completo = pd.concat([df_hist_filtrado, df_pred_filtrado]).sort_values('Anio')
    df_completo = df_completo[df_completo['Anio'] <= year]  # Filtrar hasta el año seleccionado
    df_completo['Tipo'] = df_completo['Anio'].apply(lambda x: 'Histórico' if x <= 2024 else 'Predicción')

    # Gráfico de Barras (comparación específica para el año seleccionado)
    if tipo_grafico == "Barras":
        datos_barras = pd.DataFrame({
            'Tipo': ['Promedio histórico', 'Predicción actual'],
            'Casos': [current_data['prom_hist'], current_data['casos_pred']]
        })
        chart = alt.Chart(datos_barras).mark_bar().encode(
            x='Tipo',
            y='Casos',
            color=alt.Color('Tipo', scale=alt.Scale(range=["#1f77b4", "#ff7f0e"]))
        ).properties(
            title=f"Comparación para {year}",
            width=600,
            height=400
        )

    # Gráfico de Líneas (evolución completa hasta el año seleccionado)
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
