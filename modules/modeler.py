import pandas as pd
import streamlit as st
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

def reduce_dimensions(df, n_components):
    """
    Realiza la reducción de dimensionalidad usando PCA y devuelve los datos transformados.
    """
    pca = PCA(n_components=n_components)
    df_reduced = pca.fit_transform(df)
    st.subheader(f"🔄 Datos reducidos a {n_components} dimensiones")
    st.write(f"Varianza explicada por los {n_components} componentes principales: {sum(pca.explained_variance_ratio_):.2f}")
    return df_reduced, pca

def train_model(df, target_column, n_components=2):
    """
    Función de ejemplo para entrenar un modelo predictivo después de realizar PCA.
    """
    # Verificar que la columna objetivo existe en los datos
    if target_column not in df.columns:
        st.error(f"❌ La columna objetivo '{target_column}' no se encuentra en los datos.")
        return None

    # Separar características (X) y variable objetivo (y)
    X = df.drop(columns=[target_column])
    y = df[target_column]

    # Reducir dimensiones
    X_reduced, pca = reduce_dimensions(X, n_components)

    # Dividir los datos en conjunto de entrenamiento y prueba
    X_train, X_test, y_train, y_test = train_test_split(X_reduced, y, test_size=0.2, random_state=42)

    # Crear y entrenar el modelo de regresión lineal
    model = LinearRegression()
    model.fit(X_train, y_train)

    # Realizar predicciones
    y_pred = model.predict(X_test)

    # Calcular el error cuadrático medio (RMSE)
    rmse = mean_squared_error(y_test, y_pred, squared=False)

    st.subheader(f"📈 Modelo de Regresión Lineal: Error cuadrático medio (RMSE) = {rmse:.2f}")
    
    return model, rmse, X_reduced, pca

def perform_clustering(df, n_components=2, n_clusters=3):
    """
    Realiza un clustering (K-means) después de la reducción de dimensionalidad.
    """
    # Separar características
    X = df

    # Reducir dimensiones
    X_reduced, pca = reduce_dimensions(X, n_components)

    # Entrenar el modelo de clustering K-means
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    kmeans.fit(X_reduced)

    # Añadir las etiquetas de los clusters al dataframe
    df['Cluster'] = kmeans.labels_

    st.subheader(f"📊 Clustering con K-means: {n_clusters} clusters")
    st.dataframe(df.head())

    return kmeans, df
