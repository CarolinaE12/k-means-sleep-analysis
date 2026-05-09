from flask import Flask, render_template, request
import pandas as pd
import matplotlib.pyplot as plt
import os
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

app = Flask(__name__)

# Cargar datos
df = pd.read_csv('sleep data set.csv', skiprows=[1, 2])

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/ejecutar', methods=['POST', 'GET'])
def ejecutar():
    k = int(request.form.get('clusters', 3)) # El usuario elige K
    
    # Seleccionar variables para el agrupamiento
    X = df[['total_sleep', 'danger_index']]
    
    # NORMALIZACIÓN
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Modelo K-Means
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    df['cluster'] = kmeans.fit_predict(X_scaled)
    
    # Generar Gráfica
    plt.figure(figsize=(8,6))
    plt.scatter(X_scaled[:, 0], X_scaled[:, 1], c=df['cluster'], cmap='viridis')
    plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], 
                s=300, c='red', marker='X', label='Centroides')
    plt.title(f'Clusters de Sueño (k={k})')
    plt.xlabel('Total Sleep (Normalizado)')
    plt.ylabel('Danger Index (Normalizado)')
    
    # Guardar gráfica en static
    graph_path = os.path.join('static', 'cluster_plot.png')
    plt.savefig(graph_path)
    plt.close()
    
    return render_template('resultados.html', graph_url=graph_path, 
                           data=df.to_html(classes='table table-striped'))

if __name__ == '__main__':
    app.run(debug=True)