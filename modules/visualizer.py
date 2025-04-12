import streamlit as st
import seaborn as sns
import matplotlib.pyplot as plt

def display_charts(df):
    # Configurar el tamaño y el estilo de los gráficos
    sns.set(style="whitegrid")
    
    # Ver columnas numéricas y categóricas
    num_cols = df.select_dtypes(include=['float64', 'int64']).columns
    cat_cols = df.select_dtypes(include=['object', 'category']).columns

    # 1. Gráfico de barras para columnas categóricas
    if len(cat_cols) > 0:
        st.subheader("📊 Distribución de columnas categóricas")
        for col in cat_cols:
            plt.figure(figsize=(10, 6))
            sns.countplot(data=df, x=col, palette='Set2')
            plt.title(f"Distribución de {col}")
            plt.xticks(rotation=45)
            st.pyplot(plt)

    # 2. Histogramas para columnas numéricas
    if len(num_cols) > 0:
        st.subheader("📈 Distribución de columnas numéricas")
        for col in num_cols:
            plt.figure(figsize=(10, 6))
            sns.histplot(df[col], kde=True, color='skyblue', bins=20)
            plt.title(f"Distribución de {col}")
            st.pyplot(plt)

    # 3. Boxplot para detectar outliers en columnas numéricas
    if len(num_cols) > 0:
        st.subheader("📉 Boxplots para detectar outliers")
        for col in num_cols:
            plt.figure(figsize=(10, 6))
            sns.boxplot(x=df[col], color='lightcoral')
            plt.title(f"Boxplot de {col}")
            st.pyplot(plt)

    # 4. Correlación entre variables numéricas
    if len(num_cols) > 1:
        st.subheader("🔍 Correlación entre variables numéricas")
        corr = df[num_cols].corr()
        plt.figure(figsize=(12, 8))
        sns.heatmap(corr, annot=True, cmap='coolwarm', linewidths=0.5)
        plt.title("Matriz de correlación")
        st.pyplot(plt)
