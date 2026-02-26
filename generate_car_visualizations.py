# file: generate_car_visualizations.py
# --------------------------------------------------------------
# Description: Load car_details.csv and create three visualizations:
#   1) Histogram of selling_price
#   2) Bar chart of fuel distribution (column 'fuel')
#   3) Scatter plot of selling_price vs km_driven
# --------------------------------------------------------------

import os
import pandas as pd
import matplotlib.pyplot as plt

# --------------------------------------------------------------
# Load data
# --------------------------------------------------------------
DATA_PATH = os.path.join('data', 'car_details.csv')
df = pd.read_csv(DATA_PATH)
print('Data loaded. Shape:', df.shape)
print('Columns:', df.columns.tolist())

# --------------------------------------------------------------
# 1) Histogram – selling_price distribution
# --------------------------------------------------------------
plt.figure(figsize=(8, 5))
df['selling_price'].dropna().plot.hist(
    bins=30,
    edgecolor='black',
    color='#4C72B0',
    alpha=0.75,
)
plt.title('Distribution of Selling Price')
plt.xlabel('Selling Price (₹)')
plt.ylabel('Number of Cars')
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
hist_path = 'selling_price_hist.png'
plt.savefig(hist_path, dpi=300)
print(f"[Info] Histogram saved to {hist_path}")

# --------------------------------------------------------------
# 2) Bar chart – fuel type distribution (column 'fuel')
# --------------------------------------------------------------
plt.figure(figsize=(8, 5))
fuel_counts = df['fuel'].value_counts()
fuel_counts.plot.bar(
    color='#55A868',
    edgecolor='black',
    alpha=0.8,
)
plt.title('Fuel Type Distribution')
plt.xlabel('Fuel Type')
plt.ylabel('Number of Cars')
plt.xticks(rotation=0)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
fuel_bar_path = 'fuel_type_bar.png'
plt.savefig(fuel_bar_path, dpi=300)
print(f"[Info] Fuel bar chart saved to {fuel_bar_path}")

# --------------------------------------------------------------
# 3) Scatter plot – selling_price vs km_driven
# --------------------------------------------------------------
plt.figure(figsize=(8, 5))
plt.scatter(
    df['km_driven'],
    df['selling_price'],
    c='#C44E52',
    edgecolor='black',
    alpha=0.6,
)
plt.title('Selling Price vs. Kilometers Driven')
plt.xlabel('Kilometers Driven')
plt.ylabel('Selling Price (₹)')
plt.grid(True, linestyle='--', alpha=0.7)
plt.tight_layout()
scatter_path = 'price_vs_km_scatter.png'
plt.savefig(scatter_path, dpi=300)
print(f"[Info] Scatter plot saved to {scatter_path}")

# --------------------------------------------------------------
# Show plots (optional for interactive use)
# --------------------------------------------------------------
plt.show()