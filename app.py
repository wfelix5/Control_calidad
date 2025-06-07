import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import f

plt.style.use('seaborn-v0_8-darkgrid')
st.set_page_config(page_title="Control Multivariado", layout="wide")
st.title("📈 Control Multivariado T² de Hotelling - Fase I y Fase II (con límites separados)")

archivo = st.file_uploader("🔼 Sube un archivo Excel con al menos dos hojas", type=["xlsx", "xls"])

if archivo:
    try:
        hojas = pd.ExcelFile(archivo).sheet_names
        st.markdown("### 📄 Selección de hojas para análisis")

        hoja_fase1 = st.selectbox("Selecciona la hoja para la **Fase I**", hojas)
        hoja_fase2 = st.selectbox("Selecciona la hoja para la **Fase II**", hojas, index=1 if len(hojas) > 1 else 0)

        if hoja_fase1 != hoja_fase2:
            df_fase1 = pd.read_excel(archivo, sheet_name=hoja_fase1)
            df_fase2 = pd.read_excel(archivo, sheet_name=hoja_fase2)

            columnas_comunes = list(set(df_fase1.select_dtypes(include='number').columns) &
                                     set(df_fase2.select_dtypes(include='number').columns))

            if len(columnas_comunes) < 2:
                st.warning("Se requieren al menos 2 columnas numéricas comunes en ambas hojas.")
            else:
                seleccionadas = st.multiselect(
                    "Selecciona las columnas comunes para análisis multivariado:",
                    columnas_comunes,
                    default=columnas_comunes[:2]
                )

                if len(seleccionadas) >= 2:
                    alpha = 0.05  # nivel de significancia

                    # --- FASE I ---
                    fase1_data = df_fase1[seleccionadas].dropna()
                    n1 = len(fase1_data)
                    p = len(seleccionadas)

                    mean1 = fase1_data.mean().values
                    cov1 = fase1_data.cov().values
                    inv_cov1 = np.linalg.inv(cov1)

                    T2_fase1 = [np.dot(np.dot((fase1_data.iloc[i].values - mean1).T, inv_cov1),
                                       (fase1_data.iloc[i].values - mean1)) for i in range(n1)]

                    UCL1 = (p * (n1 - 1) * (n1 + 1)) / (n1 * (n1 - p)) * f.ppf(1 - alpha, p, n1 - p)

                    # --- FASE II ---
                    fase2_data = df_fase2[seleccionadas].dropna()
                    n2 = len(fase2_data)

                    mean2 = fase2_data.mean().values
                    cov2 = fase2_data.cov().values
                    inv_cov2 = np.linalg.inv(cov2)

                    T2_fase2 = [np.dot(np.dot((fase2_data.iloc[i].values - mean2).T, inv_cov2),
                                       (fase2_data.iloc[i].values - mean2)) for i in range(n2)]

                    UCL2 = (p * (n2 - 1) * (n2 + 1)) / (n2 * (n2 - p)) * f.ppf(1 - alpha, p, n2 - p)

                    # --- GRÁFICOS ---
                    st.subheader("📊 Carta de Control - Fase I (parámetros propios)")
                    fig1, ax1 = plt.subplots(figsize=(10, 4))
                    ax1.plot(T2_fase1, marker='o', linestyle='-', color='tab:green', label='T² Fase I')
                    ax1.axhline(UCL1, color='red', linestyle='--', linewidth=2, label=f'UCL Fase I: {UCL1:.2f}')
                    ax1.set_title(f"Fase I - Variables: {', '.join(seleccionadas)}")
                    ax1.set_xlabel("Índice de Observación")
                    ax1.set_ylabel("Estadístico T²")
                    ax1.legend()
                    st.pyplot(fig1)

                    st.subheader("📊 Carta de Control - Fase II (parámetros propios)")
                    fig2, ax2 = plt.subplots(figsize=(10, 4))
                    ax2.plot(T2_fase2, marker='o', linestyle='-', color='tab:blue', label='T² Fase II')
                    ax2.axhline(UCL2, color='red', linestyle='--', linewidth=2, label=f'UCL Fase II: {UCL2:.2f}')
                    ax2.set_title(f"Fase II - Variables: {', '.join(seleccionadas)}")
                    ax2.set_xlabel("Índice de Observación")
                    ax2.set_ylabel("Estadístico T²")
                    ax2.legend()
                    st.pyplot(fig2)

                    # --- TABLA ---
                    if st.checkbox("📋 Mostrar tabla con valores T² y UCL"):
                        tabla = pd.DataFrame({
                            "Observación": list(range(1, n1 + n2 + 1)),
                            "Fase": ["Fase I"] * n1 + ["Fase II"] * n2,
                            "T²": T2_fase1 + T2_fase2,
                            "UCL aplicado": [UCL1] * n1 + [UCL2] * n2,
                            "Fuera de control": ["Sí" if t > ucl else "No" for t, ucl in zip(T2_fase1 + T2_fase2, [UCL1]*n1 + [UCL2]*n2)]
                        })
                        st.dataframe(tabla)

                else:
                    st.info("Selecciona al menos dos columnas comunes para continuar.")

        else:
            st.warning("La hoja de Fase I debe ser diferente a la hoja de Fase II.")

    except Exception as e:
        st.error(f"❌ Error al procesar el archivo: {e}")
else:
    st.info("📥 Por favor, sube un archivo Excel para comenzar.")
