import streamlit as st
import pandas as pd
import altair as alt

# Cargar datos
@st.cache_data
def load_data():
    df_pred = pd.read_csv('predicciones_alerta_vih_2025_2030_simulado.csv')
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
    index=0,
    key="year_selector"  # Agregar key único
)

departamento = st.sidebar.selectbox(
    "Departamento",
    options=available_departments,
    index=0,
    key="dept_selector"  # Agregar key único
)

sexo = st.sidebar.selectbox(
    "Sexo",
    options=available_sex,
    index=0,
    key="sex_selector"  # Agregar key único
)

tipo_grafico = st.sidebar.radio(
    "Tipo de gráfico:",
    options=["Barras", "Líneas", "Área"],
    index=0,
    key="chart_type_selector"  # Agregar key único
)

# --- Filtrar datos (CORREGIDO) ---
def get_filtered_data(departamento, sexo):
    """
    Filtra los datos por departamento y sexo.
    No filtra por año aquí para mantener todos los datos disponibles.
    """
    # Filtrar datos históricos
    hist_filtrado = df_hist[
        (df_hist['Departamento'] == departamento) &
        (df_hist['Sexo'] == sexo)
    ].copy()
    
    # Filtrar datos de predicción
    pred_filtrado = df_pred[
        (df_pred['Departamento'] == departamento) &
        (df_pred['Sexo'] == sexo)
    ].copy()
    
    return hist_filtrado, pred_filtrado

# Obtener datos filtrados
hist_filtrado, pred_filtrado = get_filtered_data(departamento, sexo)

# --- Mostrar resultados ---
if not pred_filtrado.empty:
    # Obtener datos para el año seleccionado (CORREGIDO)
    datos_año = pred_filtrado[pred_filtrado['Anio'] == year]
    
    if not datos_año.empty:
        casos_pred = int(datos_año['CasosEstimados_Predichos'].iloc[0])
        prom_hist = float(datos_año['PromHist'].iloc[0])
        alerta = datos_año['Alerta'].iloc[0]

        st.subheader(f"Resultados para {departamento} - {sexo} - {year}")
        st.markdown(f"**Casos estimados predichos:** `{casos_pred}`  \n**Promedio histórico:** `{prom_hist:.1f}`")

        # Mostrar alerta
        if alerta:
            st.error("⚠️ ¡Alerta! El valor predicho está fuera del rango histórico.", icon="🚨")
        else:
            st.success("✅ Sin alerta. El valor predicho está dentro del rango histórico.", icon="✅")

        # Métricas
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Casos predichos", casos_pred)
        with col2:
            st.metric("Promedio histórico", f"{prom_hist:.1f}")
        with col3:
            diferencia = casos_pred - prom_hist
            st.metric("Diferencia", f"{diferencia:+.1f}", delta=f"{diferencia:+.1f}")

        # --- Gráficos (MEJORADO) ---
        st.markdown("---")
        st.subheader(f"Visualización: {tipo_grafico}")

        # Preparar datos combinados para visualización
        df_hist_viz = hist_filtrado[['Anio', 'CasosEstimados']].rename(columns={'CasosEstimados': 'Casos'})
        df_hist_viz['Tipo'] = 'Histórico'
        
        df_pred_viz = pred_filtrado[['Anio', 'CasosEstimados_Predichos']].rename(columns={'CasosEstimados_Predichos': 'Casos'})
        df_pred_viz['Tipo'] = 'Predicción'
        
        df_completo = pd.concat([df_hist_viz, df_pred_viz]).sort_values('Anio').reset_index(drop=True)

        # Gráfico de Barras - Solo año seleccionado
        if tipo_grafico == "Barras":
            # Datos para el año seleccionado
            datos_barras = pd.DataFrame({
                'Categoría': ['Promedio histórico', f'Predicción {year}'],
                'Casos': [prom_hist, casos_pred],
                'Color': ['Promedio', 'Predicción']
            })
            
            chart = alt.Chart(datos_barras).mark_bar(size=60).encode(
                x=alt.X('Categoría:N', title='', axis=alt.Axis(labelAngle=0)),
                y=alt.Y('Casos:Q', title='Número de Casos'),
                color=alt.Color('Color:N', 
                              scale=alt.Scale(domain=['Promedio', 'Predicción'], 
                                            range=["#1f77b4", "#ff7f0e"]),
                              legend=alt.Legend(title="Tipo")),
                tooltip=['Categoría', 'Casos']
            ).properties(
                title=f"Comparación para {departamento} - {sexo} - {year}",
                width=600,
                height=400
            )

        # Gráfico de Líneas
        elif tipo_grafico == "Líneas":
            # Resaltar el año seleccionado
            df_completo['Destacado'] = df_completo['Anio'] == year
            
            # Línea base
            base_chart = alt.Chart(df_completo).mark_line(point=True, strokeWidth=2).encode(
                x=alt.X('Anio:O', title='Año'),
                y=alt.Y('Casos:Q', title='Número de Casos'),
                color=alt.Color('Tipo:N', 
                              scale=alt.Scale(domain=['Histórico', 'Predicción'], 
                                            range=['#1f77b4', '#d62728']),
                              legend=alt.Legend(title="Tipo de Dato")),
                tooltip=['Anio:O', 'Casos:Q', 'Tipo:N']
            )
            
            # Punto destacado para el año seleccionado
            highlight_chart = alt.Chart(df_completo[df_completo['Destacado']]).mark_circle(
                size=200, stroke='black', strokeWidth=2
            ).encode(
                x='Anio:O',
                y='Casos:Q',
                color=alt.Color('Tipo:N', 
                              scale=alt.Scale(domain=['Histórico', 'Predicción'], 
                                            range=['#1f77b4', '#d62728'])),
                tooltip=['Anio:O', 'Casos:Q', 'Tipo:N']
            )
            
            chart = (base_chart + highlight_chart).properties(
                title=f"Evolución de casos - {departamento} - {sexo} (Año destacado: {year})",
                width=700,
                height=400
            ).resolve_scale(color='independent')

        # Gráfico de Área
        else:  # Área
            chart = alt.Chart(df_completo).mark_area(opacity=0.7, line=True).encode(
                x=alt.X('Anio:O', title='Año'),
                y=alt.Y('Casos:Q', title='Número de Casos'),
                color=alt.Color('Tipo:N', 
                              scale=alt.Scale(domain=['Histórico', 'Predicción'], 
                                            range=['#1f77b4', '#d62728']),
                              legend=alt.Legend(title="Tipo de Dato")),
                tooltip=['Anio:O', 'Casos:Q', 'Tipo:N']
            ).properties(
                title=f"Tendencia de casos - {departamento} - {sexo}",
                width=700,
                height=400
            )

        st.altair_chart(chart, use_container_width=True)

        # --- Tabla de datos filtrada por año ---
        st.markdown("---")
        st.subheader("Datos del año seleccionado")
        
        # Mostrar datos históricos y predicciones para el año seleccionado
        datos_tabla = []
        
        # Datos históricos para el año
        hist_año = hist_filtrado[hist_filtrado['Anio'] == year]
        if not hist_año.empty:
            datos_tabla.append({
                'Año': year,
                'Tipo': 'Histórico',
                'Casos': int(hist_año['CasosEstimados'].iloc[0])
            })
        
        # Datos de predicción para el año
        pred_año = pred_filtrado[pred_filtrado['Anio'] == year]
        if not pred_año.empty:
            datos_tabla.append({
                'Año': year,
                'Tipo': 'Predicción',
                'Casos': int(pred_año['CasosEstimados_Predichos'].iloc[0])
            })
        
        if datos_tabla:
            df_tabla = pd.DataFrame(datos_tabla)
            st.dataframe(df_tabla, use_container_width=True)
        else:
            st.warning(f"No hay datos disponibles para el año {year}")

        # Mostrar tabla completa como expandible
        with st.expander("Ver todos los datos disponibles"):
            df_completo_tabla = df_completo.rename(columns={
                'Anio': 'Año',
                'Casos': 'Casos reportados/predichos',
                'Tipo': 'Tipo de dato'
            })
            st.dataframe(df_completo_tabla, use_container_width=True)

    else:
        st.warning(f"No hay datos para {departamento} - {sexo} en el año {year}.")
        
        # Mostrar años disponibles para esta combinación
        años_disponibles = sorted(pred_filtrado['Anio'].unique())
        st.info(f"Años disponibles para {departamento} - {sexo}: {', '.join(map(str, años_disponibles))}")
        
else:
    st.warning("No hay datos para la combinación seleccionada.")
    st.info("Verifica que los archivos CSV contengan datos para el departamento y sexo seleccionados.")

# Información adicional
st.markdown("---")
st.subheader("Información del Sistema")

col1, col2 = st.columns(2)

with col1:
    st.markdown("**Datos cargados:**")
    st.write(f"- Datos históricos: {len(df_hist)} registros")
    st.write(f"- Datos de predicción: {len(df_pred)} registros")
    st.write(f"- Departamentos disponibles: {len(available_departments)}")
    st.write(f"- Años de predicción: {min(available_years)} - {max(available_years)}")

with col2:
    st.markdown("**Filtros actuales:**")
    st.write(f"- Año seleccionado: {year}")
    st.write(f"- Departamento: {departamento}")
    st.write(f"- Sexo: {sexo}")
    st.write(f"- Tipo de gráfico: {tipo_grafico}")

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
