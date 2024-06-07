from flask import Flask, request, jsonify, send_file
import pandas as pd
from sklearn.cluster import KMeans, DBSCAN
import os
from sklearn.preprocessing import StandardScaler

app = Flask(__name__)

df = pd.DataFrame() 

@app.route('/upload', methods=['POST'])
def upload_file():
    global df
    file = request.files['file']
    df = pd.read_csv(file)
    
    # Convert the 'K' values to numeric
    df['L1'] = df['L1'].apply(lambda x: float(x[:-1]) * 1000 if 'K' in x else float(x))
    df['L2'] = df['L2'].apply(lambda x: float(x[:-1]) * 1000 if 'K' in x else float(x))
    df['L3'] = df['L3'].apply(lambda x: float(x[:-1]) * 1000 if 'K' in x else float(x))
    return jsonify(df.to_dict(orient='records'))

@app.route('/cluster', methods=['POST'])
def cluster_data():
    global df
    n_clusters = request.json['n_clusters']
    kmeans = KMeans(n_clusters=n_clusters)
    data = df.drop(columns=['Time'])
    scaler = StandardScaler()
    data_scaled = scaler.fit_transform(data)
    df['label'] = kmeans.fit_predict(data_scaled)
    return df.to_json(orient='records')

if __name__ == '__main__':
    app.run(port=5000)