import pandas as pd
import json
from io import StringIO, BytesIO
import streamlit as st

def load_data(uploaded_file):
    try:
        filename = uploaded_file.name.lower()

        if filename.endswith('.csv'):
            return pd.read_csv(uploaded_file)

        elif filename.endswith('.xlsx') or filename.endswith('.xls'):
            return pd.read_excel(uploaded_file, engine='openpyxl')

        elif filename.endswith('.json'):
            return pd.read_json(uploaded_file)

        else:
            st.warning("⚠️ Formato de archivo no soportado.")
            return None

    except Exception as e:
        st.error(f"❌ Error al cargar el archivo: {e}")
        return None
