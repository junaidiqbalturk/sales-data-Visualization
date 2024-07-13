from flask import Flask
import pandas as pd
app = Flask(__name__)

@app.route('/')
def index():
    return 'Hello, World!'

if __name__ == '__main__':
    app.run(debug=True)
# Load data
df = pd.read_csv('data/sales_data.csv')

# Print first few rows to verify data loading
print(df.head())