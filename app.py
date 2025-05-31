from flask import Flask, render_template, send_file, request
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import io

app = Flask(__name__)

def load_and_process():
    df = pd.read_csv('Mall_Customers.csv')
    # KMeans clustering
    X = df[['Annual Income (k$)', 'Spending Score (1-100)']]
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    kmeans = KMeans(n_clusters=5, init='k-means++', random_state=42)
    clusters = kmeans.fit_predict(X_scaled)
    df['Cluster'] = clusters
    return df

def get_dashboard_data(df):
    # Gender distribution
    gender_counts = df['Genre'].value_counts().to_dict()
    # Age distribution
    age_bins = pd.cut(df['Age'], bins=[15, 25, 35, 45, 55, 70], labels=["15-25","26-35","36-45","46-55","56-70"])
    age_dist = age_bins.value_counts().sort_index().to_dict()
    # Cluster scatter data
    scatter_data = df[['Annual Income (k$)', 'Spending Score (1-100)', 'Cluster']].values.tolist()
    # Cluster counts
    cluster_counts = df['Cluster'].value_counts().sort_index().to_dict()
    # Recent customers
    recent_customers = df.tail(5).to_dict(orient='records')
    return {
        'gender_counts': gender_counts,
        'age_dist': age_dist,
        'scatter_data': scatter_data,
        'cluster_counts': cluster_counts,
        'recent_customers': recent_customers
    }

def get_segments_data(df):
    # Per-cluster stats
    clusters = []
    for c in sorted(df['Cluster'].unique()):
        group = df[df['Cluster'] == c]
        clusters.append({
            'cluster': c,
            'count': len(group),
            'avg_age': round(group['Age'].mean(), 1),
            'avg_income': round(group['Annual Income (k$)'].mean(), 1),
            'avg_score': round(group['Spending Score (1-100)'].mean(), 1)
        })
    return clusters

@app.route('/')
def dashboard():
    df = load_and_process()
    data = get_dashboard_data(df)
    return render_template('dashboard.html', data=data)

@app.route('/segments')
def segments():
    df = load_and_process()
    clusters = get_segments_data(df)
    return render_template('segments.html', clusters=clusters)

@app.route('/reports')
def reports():
    df = load_and_process()
    clusters = sorted(df['Cluster'].unique())
    return render_template('reports.html', clusters=clusters)

@app.route('/download/<what>')
def download(what):
    df = load_and_process()
    if what == 'all':
        out = io.StringIO()
        df.to_csv(out, index=False)
        out.seek(0)
        return send_file(io.BytesIO(out.getvalue().encode()), mimetype='text/csv', as_attachment=True, download_name='Mall_Customers.csv')
    else:
        try:
            cluster = int(what)
            out = io.StringIO()
            df[df['Cluster'] == cluster].to_csv(out, index=False)
            out.seek(0)
            return send_file(io.BytesIO(out.getvalue().encode()), mimetype='text/csv', as_attachment=True, download_name=f'Cluster_{cluster}_Customers.csv')
        except:
            return 'Invalid cluster', 400

if __name__ == '__main__':
    app.run(debug=True) 