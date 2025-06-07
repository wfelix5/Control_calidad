import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import f

plt.style.use('seaborn-v0_8-darkgrid')
st.set_page_config(page_title="Control Multivariado", layout="wide")
st.title("\U0001F4C8 Control Multivariado T² de Hotelling con Índices CBW")

# === LÍMITES DE ESPECIFICACIÓN PREDEFINIDOS ===
LIMITES_ESPECIFICACION = {
    "Volumen de llenado (ml)": {"LSL": 590.0, "USL": 610.0},
    "Peso de la botella llena (g)": {"LSL": 620.0, "USL": 640.0}
}


# === FUNCIONES ===
def agrupar_subgrupos(df, columnas, k):
    n_subgrupos = len(df) // k
    df = df.iloc[:n_subgrupos * k]
    return df.groupby(df.index // k)[columnas].mean()

# === CARGA DE ARCHIVO ===
archivo = st.file_uploader("\U0001F53C Sube un archivo Excel con al menos dos hojas", type=["xlsx", "xls"])

if archivo:
    try:
        hojas = pd.ExcelFile(archivo).sheet_names
        st.markdown("### \U0001F4C4 Selección de hojas para análisis")

        hoja_fase1 = st.selectbox("Selecciona la hoja para la **Fase I**", hojas)
        hoja_fase2 = st.selectbox("Selecciona la hoja para la **Fase II**", hojas, index=1 if len(hojas) > 1 else 0)

        if hoja_fase1 != hoja_fase2:
            df_fase1 = pd.read_excel(archivo, sheet_name=hoja_fase1).dropna().iloc[:30]
            df_fase2 = pd.read_excel(archivo, sheet_name=hoja_fase2).dropna().iloc[:15]

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
                    alpha = 0.05
                    confianza = int((1 - alpha) * 100)
                    k = 3  # Tamaño del subgrupo

                    # Agrupar por subgrupos
                    fase1_data = df_fase1[seleccionadas]
                    fase2_data = df_fase2[seleccionadas]
                    fase1_grupos = agrupar_subgrupos(fase1_data, seleccionadas, k)
                    fase2_grupos = agrupar_subgrupos(fase2_data, seleccionadas, k)

                    # Estadísticos Fase I
                    n1 = len(fase1_grupos)
                    p = len(seleccionadas)
                    mean1 = fase1_grupos.mean().values
                    cov1 = fase1_grupos.cov().values
                    inv_cov1 = np.linalg.inv(cov1)
                    T2_fase1 = [np.dot(np.dot((x - mean1).T, inv_cov1), (x - mean1)) for x in fase1_grupos.values]
                    UCL1 = (p * (n1 - 1) * (n1 + 1)) / (n1 * (n1 - p)) * f.ppf(1 - alpha, p, n1 - p)

                    # Estadísticos Fase II
                    n2 = len(fase2_grupos)
                    mean2 = fase2_grupos.mean().values
                    cov2 = fase2_grupos.cov().values
                    inv_cov2 = np.linalg.inv(cov2)
                    T2_fase2 = [np.dot(np.dot((x - mean2).T, inv_cov2), (x - mean2)) for x in fase2_grupos.values]
                    UCL2 = (p * (n2 - 1) * (n2 + 1)) / (n2 * (n2 - p)) * f.ppf(1 - alpha, p, n2 - p)

                    # Mostrar límites utilizados
                    st.subheader("\U0001F4D0 Especificaciones utilizadas")
                    LSL_input = []
                    USL_input = []
                    for var in seleccionadas:
                        if var in LIMITES_ESPECIFICACION:
                            lsl = LIMITES_ESPECIFICACION[var]["LSL"]
                            usl = LIMITES_ESPECIFICACION[var]["USL"]
                            st.markdown(f"- **{var}**: LSL = {lsl}, USL = {usl}")
                            LSL_input.append(lsl)
                            USL_input.append(usl)
                        else:
                            st.error(f"❌ No hay límites de especificación definidos para {var}.")
                            st.stop()

                    # Capacidad CBW Fase I
                    rango1 = np.array(USL_input) - np.array(LSL_input)
                    MCp1_cbw = np.dot(rango1, rango1) / (36 * np.trace(cov1))
                    MCpk1_cbw = min(
                        np.dot(np.array(USL_input) - mean1, np.array(USL_input) - mean1) / (9 * np.trace(cov1)),
                        np.dot(mean1 - np.array(LSL_input), mean1 - np.array(LSL_input)) / (9 * np.trace(cov1))
                    )

                    st.markdown("### \U0001F9EE Índices de Capacidad Multivariada - Fase I")
                    st.markdown(f"- **MCp ** = {MCp1_cbw:.3f}")
                    st.markdown(f"- **MCpk ** = {MCpk1_cbw:.3f}")

                    # Capacidad CBW Fase II
                    MCp2_cbw = np.dot(rango1, rango1) / (36 * np.trace(cov2))
                    MCpk2_cbw = min(
                        np.dot(np.array(USL_input) - mean2, np.array(USL_input) - mean2) / (9 * np.trace(cov2)),
                        np.dot(mean2 - np.array(LSL_input), mean2 - np.array(LSL_input)) / (9 * np.trace(cov2))
                    )

                    st.markdown("### \U0001F9EE Índices de Capacidad Multivariada - Fase II")
                    st.markdown(f"- **MCp ** = {MCp2_cbw:.3f}")
                    st.markdown(f"- **MCpk ** = {MCpk2_cbw:.3f}")

                    # Gráficos
                    st.subheader(f"\U0001F4CA Carta de Control - Fase I (Confianza {confianza}%)")
                    fig1, ax1 = plt.subplots(figsize=(10, 4))
                    ax1.plot(range(1, n1 + 1), T2_fase1, marker='o', linestyle='-', color='tab:green', label='T² Fase I')
                    ax1.axhline(UCL1, color='red', linestyle='--', linewidth=2, label=f'UCL Fase I: {UCL1:.2f}')
                    ax1.set_title(f"Carta de control Multivariado Fase I ")
                    ax1.set_xlabel("Subgrupo")
                    ax1.set_ylabel("Estadístico T²")
                    ax1.legend()
                    st.pyplot(fig1)

                    st.subheader(f"\U0001F4CA Carta de Control - Fase II (Confianza {confianza}%)")
                    fig2, ax2 = plt.subplots(figsize=(10, 4))
                    ax2.plot(range(1, n2 + 1), T2_fase2, marker='o', linestyle='-', color='tab:blue', label='T² Fase II')
                    ax2.axhline(UCL2, color='red', linestyle='--', linewidth=2, label=f'UCL Fase II: {UCL2:.2f}')
                    ax2.set_title(f"Carta de control multivariado Fase II ")
                    ax2.set_xlabel("Subgrupo")
                    ax2.set_ylabel("Estadístico T²")
                    ax2.legend()
                    st.pyplot(fig2)

                    if st.checkbox("\U0001F4CB Mostrar tabla con valores T² y UCL"):
                        tabla = pd.DataFrame({
                            "Subgrupo": list(range(1, n1 + n2 + 1)),
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
    st.info("\U0001F4E5 Por favor, sube un archivo Excel para comenzar.")
