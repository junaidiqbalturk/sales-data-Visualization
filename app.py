from flask import Flask, render_template
import pandas as pd
import matplotlib.pyplot as plt
from io import BytesIO
import base64
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
app = Flask(__name__)

# Load data
df = pd.read_csv('data/sales_data.csv')

def perform_customer_segmentation(df):

    # Calculate customer metrics
    total_spending = df.groupby('Customer Name')['Sales'].sum()
    purchase_frequency = df['Customer Name'].value_counts()
    average_order_value = df.groupby('Customer Name')['Sales'].mean()

    # Create customer metrics dataframe
    customer_metrics = pd.DataFrame({
        'Total Spending': total_spending,
        'Purchase Frequency': purchase_frequency,
        'Average Order Value': average_order_value
    })

    # Standardize data if needed
    scaler = StandardScaler()
    customer_metrics_scaled = scaler.fit_transform(customer_metrics)

    # Perform K-means clustering
    num_clusters = 3  # Example number of clusters
    kmeans = KMeans(n_clusters=num_clusters, random_state=42)
    clusters = kmeans.fit_predict(customer_metrics_scaled)

    # Add cluster labels to customer metrics dataframe
    customer_metrics['Cluster'] = clusters

    return customer_metrics


@app.route('/')
def index():
    return render_template('index.html')

@app.route('/sales_dashboard')
def sales_dashboard():
    # Perform data analysis and generate plots
    monthly_sales_plot = plot_monthly_sales(df)
    top_products_plot = plot_top_products(df)
    customer_metrics = perform_customer_segmentation(df)
    # Render dashboard with plots embedded in HTML
    return render_template('sales_dashboard.html', monthly_sales_plot=monthly_sales_plot, top_products_plot=top_products_plot, customer_metrics=customer_metrics)


def plot_monthly_sales(df):
    # Ensure 'Order Date' is parsed correctly
    df['Order Date'] = pd.to_datetime(df['Order Date'], format='%d/%m/%Y', errors='coerce')

    # Drop rows with NaN in 'Order Date'
    df.dropna(subset=['Order Date'], inplace=True)

    # Calculate monthly sales
    df['Month'] = df['Order Date'].dt.to_period('M')
    monthly_sales = df.groupby('Month')['Sales'].sum()

    # Plot monthly sales trends
    plt.figure(figsize=(10, 6))
    plt.plot(monthly_sales.index.astype(str), monthly_sales.values, marker='o')
    plt.xlabel('Month')
    plt.ylabel('Sales ($)')
    plt.title('Monthly Sales Trends')
    plt.xticks(rotation=45)
    plt.tight_layout()

    # Save plot to a BytesIO object
    buffer = BytesIO()
    plt.savefig(buffer, format='png')
    buffer.seek(0)
    plot_data_uri = base64.b64encode(buffer.getvalue()).decode()

    return plot_data_uri


def plot_top_products(df, n=5):
    # Calculate top products by sales
    top_products = df.groupby('Product Name')['Sales'].sum().nlargest(n)

    # Plot top products
    plt.figure(figsize=(10, 6))
    top_products.plot(kind='bar')
    plt.xlabel('Product')
    plt.ylabel('Total Sales ($)')
    plt.title(f'Top {n} Products by Sales')
    plt.xticks(rotation=45)
    plt.tight_layout()

    # Save plot to a BytesIO object
    buffer = BytesIO()
    plt.savefig(buffer, format='png')
    buffer.seek(0)
    plot_data_uri = base64.b64encode(buffer.getvalue()).decode()

    return plot_data_uri

if __name__ == '__main__':
    app.run(debug=True)
