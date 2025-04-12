import pandas as pd
import streamlit as st

def clean_data(df):
    initial_shape = df.shape

    # Eliminar columnas completamente vacías
    df_clean = df.dropna(axis=1, how='all')

    # Eliminar filas duplicadas
    df_clean = df_clean.drop_duplicates()

    # Intentar detectar columnas que parezcan fechas
    for col in df_clean.columns:
        if df_clean[col].dtype == 'object':
            try:
                df_clean[col] = pd.to_datetime(df_clean[col], errors='ignore', utc=True)
            except Exception:
                pass

    final_shape = df_clean.shape

    # Reportar cambios
    st.markdown(f"🧽 **Filas originales:** `{initial_shape[0]}` → **Después de limpieza:** `{final_shape[0]}`")
    st.markdown(f"🧼 **Columnas originales:** `{initial_shape[1]}` → **Después de limpieza:** `{final_shape[1]}`")

    return df_clean
