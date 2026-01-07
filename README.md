# Global-Product-Sales-Analysis
This project analyzes global product sales data to understand sales distribution across countries, product lines, and time. It applies exploratory data analysis (EDA), visualization techniques, and basic clustering to uncover patterns in customer purchasing behavior and support data-driven business insights.

# Project Objective

Analyze global product sales performance

Understand product and country-level sales distribution

Explore relationships between price, quantity, and sales

Identify basic customer/product segments using clustering

Highlight trends and seasonality in sales over time

# Dataset

Global sales dataset (CSV format)

Contains order details, product information, customer data, sales amount, and dates

# Methodology
1. Data Preparation

Loaded and cleaned raw sales data using Pandas

Renamed columns for readability

Removed unnecessary columns and checked for duplicates

Converted order dates to datetime format

2. Exploratory Data Analysis (EDA)

Analyzed product distribution across countries

Examined monthly sales trends over time

Studied correlations between numerical variables (price, quantity, sales)

Used histograms, boxplots, scatter plots, and pair plots to understand distributions and outliers

3. Sales Performance Analysis

Identified top-performing countries by total sales

Determined product lines generating the highest revenue

Compared monthly sales across multiple years

4. Customer / Product Segmentation

Applied K-Means clustering using price and quantity features

Evaluated optimal cluster count using silhouette score

Visualized clusters to understand purchasing behavior patterns

5. Time Series Analysis

Performed seasonal decomposition to identify trend and seasonality in sales data

6. Geographic Visualization

Created interactive geographic maps to visualize global sales concentration

# Key Insights

Sales are unevenly distributed across countries, with a few regions generating most revenue

Certain product lines dominate total sales performance

Price and quantity show strong relationships with total sales

Sales data exhibits clear seasonal patterns

Clustering reveals distinct purchasing behavior patterns

# Tools Used

Python

Pandas, NumPy

Matplotlib, Seaborn

Plotly

Scikit-learn

Google Colab

# Conclusion

This project demonstrates a complete exploratory data analysis workflow, combining data cleaning, visualization, clustering, and time series analysis to derive meaningful business insights from global sales data.
