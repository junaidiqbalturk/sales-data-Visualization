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

#Processing and transformation of Dates in the data
def process_transform_dates(df):
    df['order Date'] = pd.to_datetime(df['Order Date'], format='%d/%m/%Y')
    #checking the datatype of order date
    print("datatype of order_Date\n",df['Order Date'].dtypes)
    df['ship Date'] = pd.to_datetime(df['Ship Date'], format='%d/%m/%Y')
    # checking the datatype of shipping date
    print("datatype of shipping_Date\n", df['Ship Date'].dtypes)
    df.dropna(subset=['order Date', 'Ship Date'], inplace=True)
    return df

# Introduces a function to load data from csv
def load_data():
    #loading csv file
    df = pd.read_csv('data/sales_data.csv')
    print("what are the colum names\n",df.columns)
    #sending the data for transformation
    process_transform_dates(df)
    #print("just after start\n",df.head()) #need to check the data that is geting loaded


    #checking for null dates in orders and ship dates
    print("checking empty dates in orders\n",df[df['order Date'].isna()])

    # checking for null dates in ship dates
    print("checking empty dates in Shipping\n", df[df['Ship Date'].isna()])
    #needs to check if the empty dates are dropped from teh data then what is the situation of data
   # print('After dropping empty dates columns\n',df.head())
   # print("Data after loading into pd\n",df.head())
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

    # Generate Region sales plot
    sales_by_region_plot = plot_sales_by_region(df)
    # Generate Order processesing time plot
    order_processing_time_plot = plot_order_processing_time(df)
    # Generate Shipping Performance Plot
    shipping_performance_plot = plot_shipping_performance(df)
    #Sales Forecast plot
    sales_forecast_plot = forecast_sales(df)
    #Geographical Analysis plot for sales
    geographical_analysis_plot = plot_geographical_analysis(df)

    return render_template('sales_dashboard.html',
                           customer_metrics_kmeans=customer_metrics_kmeans_json,
                           Z=Z_json,
                           top_products_plot=top_products_plot,
                           monthly_sales_plot=monthly_sales_plot,
                           sales_by_region_plot=sales_by_region_plot,
                           order_processing_time_plot=order_processing_time_plot,
                           shipping_performance_plot=shipping_performance_plot,
                           sales_forecast_plot=sales_forecast_plot,
                           geographical_analysis_plot=geographical_analysis_plot)

def perform_customer_segmentation(df):
    total_spending = df.groupby('Customer Name')['Sales'].sum()
    purchase_frequency = df['Customer Name'].value_counts()
    average_order_value = df.groupby('Customer Name')['Sales'].mean()

    ##Trying to debug the function through console
    print('Spending sum\n',total_spending)
    print('frequence\n',purchase_frequency)
    print('avg_order_Val\n',average_order_value)

    customer_metrics = pd.DataFrame({
        'Total Spending': total_spending,
        'Purchase Frequency': purchase_frequency,
        'Average Order Value': average_order_value
    })

    #print statement so that it can be checked what has populated into customer_metrics
    print("Metrics before scaling:\n", customer_metrics.head())

    if customer_metrics.shape[0] == 0:
        raise ValueError("Customer data is not available for segments")

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

    print("Customer Metrics before hierarchical clustering:\n", customer_metrics.head())  # Debug statement

    if customer_metrics.shape[0] == 0:
        raise ValueError("No customer data available for hierarchical clustering.")

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

# Function to plot total sales by region
def plot_sales_by_region(df):
    sales_by_region = df.groupby('Region')['Sales'].sum()
    plt.figure(figsize=(10, 6))
    sales_by_region.plot(kind='bar')
    plt.xlabel('Region')
    plt.ylabel('Total Sales ($)')
    plt.title('Total Sales by Region')
    plt.xticks(rotation=45)
    plt.tight_layout()
    buffer = BytesIO()
    plt.savefig(buffer, format='png')
    buffer.seek(0)
    plot_data_uri = base64.b64encode(buffer.getvalue()).decode()
    return plot_data_uri


# Function to plot order processing time
def plot_order_processing_time(df):
    df['Processing Time'] = (df['Ship Date'] - df['Order Date']).dt.days
    print("Processing Time\n",df['Processing Time']) #statemet for checking processing time
    #processing_time = df.groupby('Order ID')['Processing Time'].mean()
    #plt.figure(figsize=(10, 6))
    #processing_time.plot(kind='hist', bins=20)
    #plt.xlabel('Processing Time (days)')
    #plt.ylabel('Frequency')
    #plt.title('Order Processing Time Distribution')
    #plt.tight_layout()
    #buffer = BytesIO()
    #plt.savefig(buffer, format='png')
    #buffer.seek(0)
    #plot_data_uri = base64.b64encode(buffer.getvalue()).decode()
    return plot_data_uri

# Function to plot shipping performance
def plot_shipping_performance(df):
    shipping_performance = df.groupby('Ship Mode')['Sales'].sum()
    plt.figure(figsize=(10, 6))
    shipping_performance.plot(kind='bar')
    plt.xlabel('Ship Mode')
    plt.ylabel('Total Sales ($)')
    plt.title('Shipping Performance by Sales')
    plt.xticks(rotation=45)
    plt.tight_layout()
    buffer = BytesIO()
    plt.savefig(buffer, format='png')
    buffer.seek(0)
    plot_data_uri = base64.b64encode(buffer.getvalue()).decode()
    return plot_data_uri

# Function to forecast sales
def forecast_sales(df):
    sales_df = df.groupby('Order Date')['Sales'].sum().reset_index()
    sales_df.columns = ['ds', 'y']
    model = Prophet()
    model.fit(sales_df)
    future = model.make_future_dataframe(periods=12, freq='M')
    forecast = model.predict(future)
    fig = model.plot(forecast)
    buffer = BytesIO()
    fig.savefig(buffer, format='png')
    buffer.seek(0)
    plot_data_uri = base64.b64encode(buffer.getvalue()).decode()
    return plot_data_uri

# Function to plot geographical analysis
def plot_geographical_analysis(df):
    sales_by_city = df.groupby('City')['Sales'].sum().reset_index()
    fig = px.scatter_geo(sales_by_city, locations="City", locationmode='USA-states', size="Sales", title='Sales by City')
    plot_data_uri = fig.to_html(full_html=False)
    return plot_data_uri


if __name__ == '__main__':
    app.run(debug=True)
