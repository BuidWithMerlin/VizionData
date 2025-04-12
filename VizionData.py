import streamlit as st
from modules import loader, cleaner, eda, visualizer, modeler

# Configuración inicial de la app
st.set_page_config(
    page_title="Data Analyzer App",
    layout="wide",
    page_icon="📊"
)

# Encabezado
st.title("📊 Data Analyzer App")
st.markdown("Sube tu archivo de datos y obtén análisis útiles para la **toma de decisiones**.")

# Sidebar
st.sidebar.header("Opciones de carga")
uploaded_file = st.sidebar.file_uploader("📁 Sube un archivo (CSV, Excel o JSON)", type=["csv", "xlsx", "json"])

# Análisis solo si hay archivo
if uploaded_file:
    # Cargar datos
    df = loader.load_data(uploaded_file)

    if df is not None and not df.empty:
        st.success("✅ Archivo cargado correctamente.")
        
        # Mostrar vista previa
        st.subheader("👁 Vista previa de los datos")
        st.dataframe(df.head(), use_container_width=True)

        # Limpieza de datos
        st.subheader("🧹 Limpieza de datos")
        df_clean = cleaner.clean_data(df)
        st.dataframe(df_clean.head(), use_container_width=True)

        # Estadísticas
        st.subheader("📈 Estadísticas Descriptivas")
        stats_df = eda.basic_stats(df_clean)
        st.dataframe(stats_df, use_container_width=True)

        # Visualizaciones
        st.subheader("📊 Visualizaciones")
        visualizer.display_charts(df_clean)

        # Selección de la columna objetivo para el modelo
        st.subheader("🔮 Entrenamiento de modelo predictivo o clustering")
        target_column = st.selectbox("Selecciona la columna objetivo (para predecir):", df_clean.columns)

        # Selección del número de componentes principales
        n_components = st.slider("Número de componentes principales (PCA):", min_value=1, max_value=10, value=2)

        # Selección del tipo de análisis (regresión o clustering)
        analysis_type = st.radio("Selecciona el tipo de análisis:", ("Predicción (Regresión)", "Clustering (K-means)"))

        # Realizar análisis dependiendo de la elección
        if analysis_type == "Predicción (Regresión)":
            if st.button("Entrenar modelo predictivo"):
                if target_column:
                    model, rmse, X_reduced, pca = modeler.train_model(df_clean, target_column, n_components)
                    if model is not None:
                        st.success(f"Modelo entrenado con éxito. RMSE: {rmse:.2f}")
                    else:
                        st.error("❌ No se pudo entrenar el modelo.")
                else:
                    st.warning("⚠️ Por favor, selecciona una columna objetivo para entrenar el modelo.")
        
        elif analysis_type == "Clustering (K-means)":
            n_clusters = st.slider("Número de clusters para K-means:", min_value=2, max_value=10, value=3)
            if st.button("Realizar clustering"):
                kmeans, df_clustered = modeler.perform_clustering(df_clean, n_components, n_clusters)
                st.write("Resultados del clustering (con K-means):")
                st.dataframe(df_clustered.head(), use_container_width=True)

    else:
        st.error("❌ No se pudo cargar el archivo o está vacío.")
else:
    st.info("🔍 Esperando que subas un archivo...")
