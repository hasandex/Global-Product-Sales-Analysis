# ==============================
# Global Product Sales Analysis
# ==============================

# Mount Google Drive to access the dataset (used in Google Colab)
from google.colab import drive
drive.mount('/content/gdrive')

# ------------------------------
# Load and prepare the dataset
# ------------------------------
import pandas as pd

# Load sales dataset from Google Drive
df = pd.read_csv(
    '/content/gdrive/MyDrive/sales_data_sample.csv',
    encoding='unicode_escape'
)

# Rename columns for better readability and consistency
columns_new = {
    'ORDERNUMBER': 'ORDER_NUMBER',
    'QUANTITYORDERED': 'QUANTITY_ORDER',
    'PRICEEACH': 'PRICE',
    'ORDERLINENUMBER': 'ORDER_LINE_NUMBER',
    'ORDERDATE': 'ORDER_DATE',
    'PRODUCTLINE': 'PRODUCT_LINE',
    'PRODUCTCODE': 'PRODUCT_CODE',
    'CUSTOMERNAME': 'CUSTOMER_NAME',
    'POSTALCODE': 'POSTAL_CODE',
    'CONTACTLASTNAME': 'CONTACT_LAST_NAME',
    'CONTACTFIRSTNAME': 'CONTACT_FIRST_NAME',
    'DEALSIZE': 'DEAL_SIZE'
}
df.rename(columns=columns_new, inplace=True)

# Preview dataset
df.head()

# Check data types and structure
df.info()

# Check missing values
df.isnull().sum()

# Summary statistics
df.describe().T

# Drop unnecessary columns
df = df.drop(columns=['ADDRESSLINE2', 'STATE', 'POSTAL_CODE', 'TERRITORY'])

# Check duplicates
df.duplicated().sum()

# ------------------------------
# Date handling and feature creation
# ------------------------------
# Convert ORDER_DATE to datetime format
df['ORDER_DATE'] = pd.to_datetime(df['ORDER_DATE'])

# Extract month name from order date
df['Month'] = df['ORDER_DATE'].dt.strftime('%B')

# ------------------------------
# Exploratory Data Analysis (EDA)
# ------------------------------
import matplotlib.pyplot as plt

# Distribution of products across countries
grouped = df.groupby(['PRODUCT_LINE', 'COUNTRY']).size().reset_index(name='COUNT')
plt.figure(figsize=(20, 6))
plt.bar(grouped['COUNTRY'], grouped['COUNT'])
plt.xlabel('Country')
plt.ylabel('Count')
plt.title('Distribution of Products Across Countries')
plt.show()

# Monthly total sales trend
monthly_sales = df.groupby(
    pd.Grouper(key='ORDER_DATE', freq='M')
)['SALES'].sum().reset_index()

plt.figure(figsize=(10, 5))
plt.plot(monthly_sales['ORDER_DATE'], monthly_sales['SALES'])
plt.title('Total Sales Over Time')
plt.xlabel('Month')
plt.ylabel('Sales')
plt.show()

# ------------------------------
# Correlation Analysis
# ------------------------------
import seaborn as sns

numeric_df = df[['QUANTITY_ORDER', 'PRICE', 'SALES', 'MSRP']]
corr_matrix = numeric_df.corr(method='pearson')

sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Correlation Matrix')
plt.show()

# ------------------------------
# Country and product performance
# ------------------------------
# Total sales by country
total_sales_country = df.groupby('COUNTRY')['SALES'].sum().reset_index()
total_sales_country.sort_values(by='SALES', ascending=False)

# Product line with highest revenue
product_revenue = df.groupby('PRODUCT_LINE')['SALES'].sum().reset_index()
product_revenue.loc[product_revenue['SALES'].idxmax()]

# ------------------------------
# Monthly country-level aggregation
# ------------------------------
avg_total_df = df[['ORDER_DATE', 'COUNTRY', 'QUANTITY_ORDER', 'PRICE']].copy()
avg_total_df.set_index('ORDER_DATE', inplace=True)

grouped = avg_total_df.groupby([pd.Grouper(freq='M'), 'COUNTRY'])
result = grouped.agg({
    'PRICE': 'mean',
    'QUANTITY_ORDER': 'sum'
}).reset_index()

result['ORDER_DATE'] = result['ORDER_DATE'].dt.strftime('%Y-%m')

# ------------------------------
# Distribution and outlier analysis
# ------------------------------
df.boxplot(column='PRICE', by='COUNTRY', figsize=(20, 6))
plt.title('Price Distribution by Country')
plt.xticks(rotation=45)
plt.suptitle('')
plt.show()

# Scatter plot: Quantity vs Sales
df.plot.scatter(x='QUANTITY_ORDER', y='SALES', figsize=(10, 6))
plt.show()

# Pairplot for numerical variables
sns.pairplot(numeric_df)
plt.show()

# Histogram of product prices
plt.hist(df['PRICE'], bins=30, edgecolor='k')
plt.xlabel('Price')
plt.ylabel('Frequency')
plt.title('Distribution of Product Prices')
plt.show()

# ------------------------------
# Sales trends by country over time
# ------------------------------
total_sales_time = df.groupby(['COUNTRY', 'ORDER_DATE'])['SALES'].sum().reset_index()

for country in total_sales_time['COUNTRY'].unique():
    country_data = total_sales_time[total_sales_time['COUNTRY'] == country]
    plt.plot(country_data['ORDER_DATE'], country_data['SALES'], label=country)

plt.legend(bbox_to_anchor=(1.05, 1))
plt.xticks(rotation=60)
plt.show()

# ------------------------------
# Geographic sales visualization
# ------------------------------
import plotly.express as px

geo = df.groupby('COUNTRY')['SALES'].sum().reset_index()
fig = px.scatter_geo(
    geo,
    locations='COUNTRY',
    color='SALES',
    size='SALES',
    projection='natural earth',
    title='Global Sales Distribution'
)
fig.show()

# ------------------------------
# Customer/Product Segmentation using K-Means
# ------------------------------
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

X = df[['PRICE', 'QUANTITY_ORDER']]

# Evaluate optimal K using silhouette score
for k in range(2, 11):
    model = KMeans(n_clusters=k, n_init=10)
    labels = model.fit_predict(X)
    score = silhouette_score(X, labels)
    print(f'K={k}, Silhouette Score={score:.2f}')

# Apply K-Means with selected K
model_km = KMeans(n_clusters=2, n_init=10)
labels = model_km.fit_predict(X)

# Visualize clusters
plt.scatter(df['PRICE'], df['QUANTITY_ORDER'], c=labels, cmap='rainbow')
plt.scatter(
    model_km.cluster_centers_[:, 0],
    model_km.cluster_centers_[:, 1],
    color='black'
)
plt.xlabel('Price')
plt.ylabel('Quantity Ordered')
plt.title('Product Segmentation Using K-Means')
plt.show()

# ------------------------------
# Time Series Decomposition
# ------------------------------
from statsmodels.tsa.seasonal import seasonal_decompose

df_ts = df.copy()
df_ts.set_index('ORDER_DATE', inplace=True)
df_ts = df_ts.resample('M').sum()

decomposition = seasonal_decompose(df_ts['SALES'], model='additive')

decomposition.plot()
plt.show()

# ------------------------------
# Interactive sales visualization
# ------------------------------
fig = px.scatter(
    df,
    x='ORDER_DATE',
    y='SALES',
    color='COUNTRY',
    hover_data=['PRODUCT_LINE']
)
fig.update_layout(xaxis_rangeslider_visible=True)
fig.show()

# ------------------------------
# Monthly sales comparison across years
# ------------------------------
grouped_sales = df.groupby(['YEAR_ID', 'MONTH_ID'])['SALES'].sum().reset_index()
pivot_sales = grouped_sales.pivot(index='MONTH_ID', columns='YEAR_ID', values='SALES')

pivot_sales.plot(kind='bar', figsize=(10, 6))
plt.xlabel('Month')
plt.ylabel('Total Sales')
plt.title('Monthly Sales by Year')
plt.show()
