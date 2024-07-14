from flask import Flask, render_template
import pandas as pd
import matplotlib.pyplot as plt
from io import BytesIO
import base64
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from scipy.cluster.hierarchy import dendrogram, linkage
import json

app = Flask(__name__)

# Function to load data
def load_data():
    df = pd.read_csv('data/sales_data.csv')
    return df

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/sales_dashboard')
def sales_dashboard():
    df = load_data()
    customer_metrics_kmeans = perform_customer_segmentation(df)
    customer_metrics_hierarchical, Z = perform_hierarchical_clustering(df)

    customer_metrics_kmeans_json = customer_metrics_kmeans.reset_index().to_json(orient='records')
    Z_json = json.dumps(Z.tolist())

    # Generate plot images
    top_products_plot = plot_top_products(df)
    monthly_sales_plot = plot_monthly_sales(df)

    return render_template('sales_dashboard.html',
                           customer_metrics_kmeans=customer_metrics_kmeans_json,
                           Z=Z_json,
                           top_products_plot=top_products_plot,
                           monthly_sales_plot=monthly_sales_plot)

def perform_customer_segmentation(df):
    total_spending = df.groupby('Customer Name')['Sales'].sum()
    purchase_frequency = df['Customer Name'].value_counts()
    average_order_value = df.groupby('Customer Name')['Sales'].mean()

    customer_metrics = pd.DataFrame({
        'Total Spending': total_spending,
        'Purchase Frequency': purchase_frequency,
        'Average Order Value': average_order_value
    })

    scaler = StandardScaler()
    customer_metrics_scaled = scaler.fit_transform(customer_metrics)

    kmeans = KMeans(n_clusters=3, random_state=42)
    clusters = kmeans.fit_predict(customer_metrics_scaled)

    customer_metrics['Cluster'] = clusters

    return customer_metrics

def perform_hierarchical_clustering(df):
    total_spending = df.groupby('Customer Name')['Sales'].sum()
    purchase_frequency = df['Customer Name'].value_counts()
    average_order_value = df.groupby('Customer Name')['Sales'].mean()

    customer_metrics = pd.DataFrame({
        'Total Spending': total_spending,
        'Purchase Frequency': purchase_frequency,
        'Average Order Value': average_order_value
    })

    scaler = StandardScaler()
    customer_metrics_scaled = scaler.fit_transform(customer_metrics)

    Z = linkage(customer_metrics_scaled, method='ward')

    return customer_metrics, Z

def plot_monthly_sales(df):
    df['Order Date'] = pd.to_datetime(df['Order Date'], format='%d/%m/%Y', errors='coerce')
    df.dropna(subset=['Order Date'], inplace=True)

    df['Month'] = df['Order Date'].dt.to_period('M')
    monthly_sales = df.groupby('Month')['Sales'].sum()

    plt.figure(figsize=(10, 6))
    plt.plot(monthly_sales.index.astype(str), monthly_sales.values, marker='o')
    plt.xlabel('Month')
    plt.ylabel('Sales ($)')
    plt.title('Monthly Sales Trends')
    plt.xticks(rotation=45)
    plt.tight_layout()

    buffer = BytesIO()
    plt.savefig(buffer, format='png')
    buffer.seek(0)
    plot_data_uri = base64.b64encode(buffer.getvalue()).decode()

    return plot_data_uri

def plot_top_products(df, n=5):
    top_products = df.groupby('Product Name')['Sales'].sum().nlargest(n)

    plt.figure(figsize=(10, 6))
    top_products.plot(kind='bar')
    plt.xlabel('Product')
    plt.ylabel('Total Sales ($)')
    plt.title(f'Top {n} Products by Sales')
    plt.xticks(rotation=45)
    plt.tight_layout()

    buffer = BytesIO()
    plt.savefig(buffer, format='png')
    buffer.seek(0)
    plot_data_uri = base64.b64encode(buffer.getvalue()).decode()

    return plot_data_uri

if __name__ == '__main__':
    app.run(debug=True)
