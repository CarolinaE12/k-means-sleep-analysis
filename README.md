# k-means-sleep-analysis
Aplicación web con Flask que utiliza K-Means para agrupar especies según sus hábitos de sueño y niveles de peligro.
# K-Means Sleep Analysis

Este proyecto aplica un modelo de **Aprendizaje No Supervisado (K-Means)** para agrupar especies de mamíferos según sus hábitos de sueño, características físicas y factores de riesgo. La aplicación está desarrollada en **Python** utilizando el framework **Flask**.

## Proceso de Preparación de Datos (Orange Data Mining)

Antes de la implementación en Flask, los datos pasaron por un riguroso proceso de limpieza y curación:

1.  **Imputación de Datos:** Se eliminaron los valores faltantes (missing values) utilizando técnicas de imputación por media, logrando un dataset con **0 valores nulos**.
2.  **Data Augmentation:** Dado que el dataset original contaba con 62 instancias, se aplicó un script de Python para generar una réplica con **ruido Gaussiano ($\pm 5\%$)**, alcanzando un total de **124 registros** para mejorar la robustez del modelo.
3.  **Análisis de Escalas:** Mediante gráficas de dispersión iniciales, se detectó una gran disparidad de magnitudes 

##  Tecnologías Utilizadas

*   **Backend:** Flask (Python)
*   **Machine Learning:** Scikit-Learn (K-Means, StandardScaler)
*   **Preprocesamiento:** Orange Data Mining
*   **Visualización:** Matplotlib / Seaborn
