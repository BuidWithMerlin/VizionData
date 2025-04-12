import pandas as pd
import streamlit as st

def basic_stats(df):
    """
    Genera estadísticas descriptivas básicas de un DataFrame.
    """
    # Estadísticas descriptivas para columnas numéricas
    stats = df.describe().T  # Transpuesta para una visualización más clara
    stats['range'] = stats['max'] - stats['min']  # Añadir rango (diferencia entre max y min)
    
    # Mostrar información sobre columnas categóricas
    cat_cols = df.select_dtypes(include=['object', 'category']).columns
    if len(cat_cols) > 0:
        for col in cat_cols:
            stats.loc[col, 'unique_values'] = df[col].nunique()
            stats.loc[col, 'top'] = df[col].mode()[0]
            stats.loc[col, 'freq'] = df[col].value_counts().iloc[0]

    # Mostrar las estadísticas de forma ordenada
    return stats

def correlation_matrix(df):
    """
    Muestra la matriz de correlación para las variables numéricas.
    """
    corr = df.corr()
    st.subheader("🔍 Matriz de correlación")
    st.dataframe(corr, use_container_width=True)
