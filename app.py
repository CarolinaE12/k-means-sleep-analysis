from flask import Flask, render_template, request
import pandas as pd
import matplotlib
matplotlib.use('Agg')

import matplotlib.pyplot as plt
import os

from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

app = Flask(__name__)

# Cargar datos
df = pd.read_csv('sleep_data_set.csv', skiprows=[1, 2])


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/ejecutar', methods=['POST', 'GET'])
def ejecutar():

    k = int(request.form.get('clusters', 3))

    # Variables para clustering
    X = df[['total_sleep', 'danger_index']]

    # Normalización
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Modelo K-Means
    kmeans = KMeans(
        n_clusters=k,
        random_state=42,
        n_init=10
    )

    df['cluster'] = kmeans.fit_predict(X_scaled)

    # Crear gráfica
    plt.figure(figsize=(8, 6))

    plt.scatter(
        X_scaled[:, 0],
        X_scaled[:, 1],
        c=df['cluster'],
        cmap='viridis'
    )

    plt.scatter(
        kmeans.cluster_centers_[:, 0],
        kmeans.cluster_centers_[:, 1],
        s=300,
        c='red',
        marker='X',
        label='Centroides'
    )

    plt.title(f'Clusters de Sueño (k={k})')
    plt.xlabel('Total Sleep (Normalizado)')
    plt.ylabel('Danger Index (Normalizado)')

    plt.tight_layout()

    # Guardar gráfica
    graph_path = os.path.join(
        app.static_folder,
        'cluster_plot.png'
    )

    plt.savefig(graph_path)
    plt.close()

    return render_template(
        'resultados.html',
        graph_url='cluster_plot.png',
        data=df.to_html(classes='table table-striped')
    )


if __name__ == '__main__':
    app.run(debug=True)