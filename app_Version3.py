import streamlit as st
import pandas as pd
import altair as alt
import numpy as np

# Cargar datos
@st.cache_data
def load_data():
    try:
        # Cargar datos de predicción (2025-2030)
        df_pred = pd.read_csv('predicciones_alerta_vih_2025_2030_simulado.csv')
        
        # Cargar datos históricos (2015-2024)
        df_hist = pd.read_csv('DATASET_VIH.csv')
        
        # Verificar y limpiar datos históricos
        if 'Tendencia' not in df_hist.columns:
            df_hist['Tendencia'] = 'Histórico'
        
        # Asegurar que los casos sean numéricos
        df_hist['CasosEstimados'] = pd.to_numeric(df_hist['CasosEstimados'], errors='coerce')
        df_hist = df_hist.dropna(subset=['CasosEstimados'])
        
        return df_pred, df_hist
    except Exception as e:
        st.error(f"Error cargando datos: {str(e)}")
        return pd.DataFrame(), pd.DataFrame()

df_pred, df_hist = load_data()

# Verificar que los datos se cargaron correctamente
if df_pred.empty or df_hist.empty:
    st.error("No se pudieron cargar los datos. Verifica que los archivos CSV estén disponibles.")
    st.stop()

# Configuración de la página
st.set_page_config(
    page_title="Sistema de Alerta Temprana VIH - Perú",
    layout="wide",
    initial_sidebar_state="expanded",
    page_icon="🦠"
)

# CSS personalizado para mejorar la apariencia
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(90deg, #1f77b4, #ff7f0e);
        padding: 1.5rem;
        border-radius: 10px;
        color: white;
        margin-bottom: 2rem;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    .metric-container {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 10px;
        border-left: 4px solid #1f77b4;
        margin-bottom: 1rem;
    }
    .alert-danger {
        background-color: #f8d7da;
        border: 1px solid #f5c6cb;
        color: #721c24;
        padding: 1rem;
        border-radius: 5px;
        margin: 1rem 0;
    }
    .alert-success {
        background-color: #d4edda;
        border: 1px solid #c3e6cb;
        color: #155724;
        padding: 1rem;
        border-radius: 5px;
        margin: 1rem 0;
    }
    .stSelectbox, .stRadio {
        margin-bottom: 1rem;
    }
    .stDataFrame {
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
</style>
""", unsafe_allow_html=True)

# Título principal
st.markdown("""
<div class="main-header">
    <h1>🦠 Sistema de Alerta Temprana para VIH en Perú</h1>
    <p>Visualización interactiva de predicciones 2025-2030 basadas en datos simulados</p>
</div>
""", unsafe_allow_html=True)

# --- Barra lateral de filtros ---
st.sidebar.header("🔍 Filtros de Consulta")

# Obtener opciones únicas del dataset de predicción
available_years = sorted(df_pred['Anio'].unique())
available_departments = sorted(df_pred['Departamento'].unique())
available_sex = sorted(df_pred['Sexo'].unique())

# Información sobre los datos
st.sidebar.markdown("---")
st.sidebar.markdown("**📊 Datos de Predicción:**")
st.sidebar.write(f"• Años: {min(available_years)}-{max(available_years)}")
st.sidebar.write(f"• Departamentos: {len(available_departments)}")
st.sidebar.write(f"• Total registros: {len(df_pred):,}")

st.sidebar.markdown("---")

# Filtros interactivos
year = st.sidebar.selectbox(
    "📅 Año de predicción",
    options=available_years,
    index=0,
    key="year_selector"
)

departamento = st.sidebar.selectbox(
    "🏛️ Departamento",
    options=available_departments,
    index=0,
    key="dept_selector"
)

sexo = st.sidebar.selectbox(
    "👥 Sexo",
    options=available_sex,
    index=0,
    key="sex_selector"
)

tipo_grafico = st.sidebar.radio(
    "📈 Tipo de visualización",
    options=["Barras", "Líneas", "Área"],
    index=0,
    key="chart_type_selector"
)

# Mostrar filtros actuales
st.sidebar.markdown("---")
st.sidebar.markdown("**🎯 Filtros aplicados:**")
st.sidebar.markdown(f"- Año: **{year}**")
st.sidebar.markdown(f"- Departamento: **{departamento}**")
st.sidebar.markdown(f"- Sexo: **{sexo}**")

# --- Funciones para procesamiento de datos ---
def get_current_prediction(year, departamento, sexo):
    """Obtiene los datos de predicción para la selección actual"""
    mask = (
        (df_pred['Anio'] == year) &
        (df_pred['Departamento'] == departamento) &
        (df_pred['Sexo'] == sexo)
    )
    return df_pred[mask]

def get_historical_data(departamento, sexo):
    """Obtiene datos históricos para el departamento y sexo seleccionados"""
    mask = (
        (df_hist['Departamento'] == departamento) &
        (df_hist['Sexo'] == sexo)
    )
    return df_hist[mask].sort_values('Anio')

def prepare_combined_data(hist_data, pred_data):
    """Combina y prepara datos históricos y de predicción para visualización"""
    # Preparar datos históricos
    df_hist_viz = hist_data[['Anio', 'CasosEstimados']].rename(columns={'CasosEstimados': 'Casos'})
    df_hist_viz['Tipo'] = 'Histórico'
    
    # Preparar datos de predicción
    df_pred_viz = pred_data[['Anio', 'CasosEstimados_Predichos']].rename(columns={'CasosEstimados_Predichos': 'Casos'})
    df_pred_viz['Tipo'] = 'Predicción'
    
    # Combinar y ordenar
    df_completo = pd.concat([df_hist_viz, df_pred_viz]).sort_values('Anio')
    
    # Calcular promedio histórico
    prom_hist = df_hist_viz['Casos'].mean() if not df_hist_viz.empty else 0
    
    return df_completo, prom_hist

# --- Obtener datos para la selección actual ---
current_pred = get_current_prediction(year, departamento, sexo)
hist_data = get_historical_data(departamento, sexo)
df_completo, prom_hist = prepare_combined_data(hist_data, df_pred[df_pred['Departamento'] == departamento])

# --- Mostrar resultados ---
if not current_pred.empty:
    # Extraer valores de la predicción actual
    casos_pred = int(current_pred['CasosEstimados_Predichos'].iloc[0])
    prom_hist_pred = float(current_pred['PromHist'].iloc[0])  # Usar el promedio del dataset de predicción
    alerta = current_pred['Alerta'].iloc[0]
    
    # Calcular diferencias
    diferencia = casos_pred - prom_hist_pred
    porcentaje = (diferencia / prom_hist_pred * 100) if prom_hist_pred != 0 else 0
    
    # Mostrar encabezado
    st.markdown(f"## 📋 Resultados para {departamento} - {sexo} - {year}")
    
    # Métricas principales
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric(
            "🎯 Casos Predichos", 
            f"{casos_pred:,}",
            delta=f"{diferencia:+,.0f} ({porcentaje:+.1f}%)"
        )
    
    with col2:
        st.metric(
            "📊 Promedio Histórico", 
            f"{prom_hist_pred:,.1f}"
        )
    
    with col3:
        if alerta:
            st.metric("🚨 Estado", "ALERTA", delta="Fuera de rango", delta_color="inverse")
        else:
            st.metric("✅ Estado", "NORMAL", delta="Dentro de rango")

    # Mostrar alerta
    if alerta:
        st.markdown(f"""
        <div class="alert-danger">
            <h4>⚠️ Alerta Epidemiológica Detectada</h4>
            <p>Los casos predichos para {year} ({casos_pred}) están significativamente fuera del rango histórico promedio ({prom_hist_pred:.1f}).</p>
            <p>Diferencia: <strong>{diferencia:+,.0f} casos</strong> ({porcentaje:+.1f}%)</p>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div class="alert-success">
            <h4>✅ Situación Normal</h4>
            <p>Los casos predichos para {year} están dentro del rango histórico esperado.</p>
            <p>Diferencia: <strong>{diferencia:+,.0f} casos</strong> ({porcentaje:+.1f}%)</p>
        </div>
        """, unsafe_allow_html=True)

    # --- Visualización de datos ---
    st.markdown("---")
    st.markdown(f"## 📊 Evolución Temporal - {departamento} ({sexo})")
    
    # Gráfico de Barras (comparación específica)
    if tipo_grafico == "Barras":
        # Datos para el gráfico de barras
        bar_data = pd.DataFrame({
            'Tipo': ['Promedio Histórico', 'Predicción'],
            'Casos': [prom_hist_pred, casos_pred],
            'Color': ['#1f77b4', '#ff7f0e']
        })
        
        chart = alt.Chart(bar_data).mark_bar(size=60).encode(
            x=alt.X('Tipo:N', title='', axis=alt.Axis(labelAngle=0)),
            y=alt.Y('Casos:Q', title='Número de Casos'),
            color=alt.Color('Color:N', scale=alt.Scale(range=['#1f77b4', '#ff7f0e']), legend=None),
            tooltip=['Tipo:N', 'Casos:Q']
        ).properties(
            title=f"Comparación para {year}",
            width=600,
            height=400
        )
        
        # Agregar texto con los valores
        text = chart.mark_text(
            align='center',
            baseline='bottom',
            dy=-5,
            fontSize=14
        ).encode(
            text='Casos:Q'
        )
        
        chart = (chart + text)

    # Gráfico de Líneas (evolución completa)
    elif tipo_grafico == "Líneas":
        # Crear gráfico base
        line_chart = alt.Chart(df_completo).mark_line(point=True).encode(
            x=alt.X('Anio:O', title='Año'),
            y=alt.Y('Casos:Q', title='Número de Casos'),
            color=alt.Color('Tipo:N', 
                          scale=alt.Scale(domain=['Histórico', 'Predicción'], 
                                        range=['#1f77b4', '#d62728']),
                          legend=alt.Legend(title="Tipo de Datos")),
            tooltip=['Anio:O', 'Casos:Q', 'Tipo:N']
        )
        
        # Destacar el punto del año seleccionado
        highlight = alt.Chart(df_completo[df_completo['Anio'] == year]).mark_circle(
            size=100, color='red'
        ).encode(
            x='Anio:O',
            y='Casos:Q',
            tooltip=['Anio:O', 'Casos:Q', 'Tipo:N']
        )
        
        # Línea vertical para marcar el año seleccionado
        rule = alt.Chart(pd.DataFrame({'x': [year]})).mark_rule(
            color='red', strokeDash=[3, 3]
        ).encode(
            x='x:O'
        )
        
        chart = (line_chart + highlight + rule).properties(
            width=800,
            height=450
        )

    # Gráfico de Área
    else:
        chart = alt.Chart(df_completo).mark_area(opacity=0.7, line=True).encode(
            x=alt.X('Anio:O', title='Año'),
            y=alt.Y('Casos:Q', title='Número de Casos'),
            color=alt.Color('Tipo:N', 
                          scale=alt.Scale(domain=['Histórico', 'Predicción'], 
                                        range=['#1f77b4', '#d62728']),
                          legend=alt.Legend(title="Tipo de Datos")),
            tooltip=['Anio:O', 'Casos:Q', 'Tipo:N']
        ).properties(
            width=800,
            height=450
        )
        
        # Agregar línea vertical para el año seleccionado
        rule = alt.Chart(pd.DataFrame({'x': [year]})).mark_rule(
            color='red', strokeDash=[3, 3]
        ).encode(
            x='x:O'
        )
        
        chart = chart + rule

    st.altair_chart(chart, use_container_width=True)

    # --- Análisis detallado ---
    st.markdown("---")
    st.markdown("## 🔍 Análisis Detallado")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 📈 Estadísticas Históricas")
        if not hist_data.empty:
            casos_min = hist_data['CasosEstimados'].min()
            casos_max = hist_data['CasosEstimados'].max()
            casos_avg = hist_data['CasosEstimados'].mean()
            casos_std = hist_data['CasosEstimados'].std()
            
            st.write(f"**Mínimo histórico:** {casos_min:,.0f} casos")
            st.write(f"**Máximo histórico:** {casos_max:,.0f} casos")
            st.write(f"**Promedio histórico:** {casos_avg:,.1f} casos")
            st.write(f"**Desviación estándar:** {casos_std:,.1f}")
            
            # Mostrar rango normal (promedio ± 1 desviación estándar)
            st.write(f"**Rango normal esperado:** {casos_avg-casos_std:,.1f} - {casos_avg+casos_std:,.1f} casos")
    
    with col2:
        st.markdown("### 🧮 Evaluación de Riesgo")
        if diferencia > 0:
            st.write(f"**Predicción por encima del promedio:** {diferencia:,.0f} casos ({porcentaje:+.1f}%)")
        else:
            st.write(f"**Predicción por debajo del promedio:** {abs(diferencia):,.0f} casos ({porcentaje:+.1f}%)")
        
        # Clasificación de riesgo basada en desviaciones estándar
        if casos_std > 0:
            desviaciones = abs(casos_pred - casos_avg) / casos_std
        else:
            desviaciones = 0
            
        if desviaciones < 1:
            riesgo = "🟢 Bajo"
            explicacion = "Dentro del rango histórico normal"
        elif desviaciones < 2:
            riesgo = "🟡 Moderado"
            explicacion = "Fuera del rango normal pero dentro de lo esperado"
        else:
            riesgo = "🔴 Alto"
            explicacion = "Desviación significativa del patrón histórico"
        
        st.write(f"**Nivel de riesgo:** {riesgo}")
        st.write(f"**Explicación:** {explicacion}")

    # --- Tablas de datos ---
    st.markdown("---")
    st.markdown("## 📋 Datos Detallados")
    
    # Tabla con datos del año seleccionado
    st.markdown(f"### Datos para {year}")
    datos_año = pd.DataFrame({
        'Tipo': ['Histórico', 'Predicción', 'Promedio Histórico'],
        'Casos': [
            hist_data[hist_data['Anio'] == year]['CasosEstimados'].iloc[0] if not hist_data[hist_data['Anio'] == year].empty else 'N/A',
            casos_pred,
            prom_hist_pred
        ],
        'Estado': [
            'Real' if not hist_data[hist_data['Anio'] == year].empty else 'No disponible',
            'Alerta' if alerta else 'Normal',
            'Referencia'
        ]
    })
    st.dataframe(datos_año, hide_index=True, use_container_width=True)
    
    # Tabla completa expandible
    with st.expander("📊 Ver todos los datos históricos y de predicción"):
        st.dataframe(df_completo.rename(columns={
            'Anio': 'Año',
            'Casos': 'Número de Casos',
            'Tipo': 'Tipo de Dato'
        }), use_container_width=True)

else:
    st.error("No se encontraron datos para la combinación seleccionada.")
    
    # Mostrar opciones disponibles
    st.markdown("### 🔍 Datos disponibles para:")
    st.write(f"**Departamento:** {departamento}")
    available_years_dept = df_pred[
        (df_pred['Departamento'] == departamento) & 
        (df_pred['Sexo'] == sexo)
    ]['Anio'].unique()
    
    if len(available_years_dept) > 0:
        st.write(f"Años con datos: {sorted(available_years_dept)}")
    else:
        st.write("No hay datos para esta combinación de departamento y sexo")

# --- Pie de página ---
st.markdown("---")
st.markdown("""
<div style="text-align: center; padding: 1rem; color: #666;">
    <p>Sistema de Alerta Temprana VIH - Perú | Datos simulados para predicciones 2025-2030</p>
    <p><small>Desarrollado con Streamlit | Versión 3.2</small></p>
</div>
""", unsafe_allow_html=True)
