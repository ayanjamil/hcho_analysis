import pandas as pd
import matplotlib.pyplot as plt
import os
from datetime import datetime

# ------------------------------
# Ask for CSV Path
# ------------------------------
csv_file = input("Enter path to your HCHO summary CSV file: ").strip()
if not os.path.exists(csv_file):
    raise FileNotFoundError(f"CSV file not found: {csv_file}")

print(f"Using file: {csv_file}")

# ------------------------------
# Create results folder
# ------------------------------
base_name = os.path.splitext(os.path.basename(csv_file))[0]
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
out_dir = os.path.join("results", f"{base_name}_{timestamp}")
os.makedirs(out_dir, exist_ok=True)
print(f"Results folder created: {out_dir}")

# ------------------------------
# Load CSV
# ------------------------------
df = pd.read_csv(csv_file)

# Normalize column names
df.columns = df.columns.str.strip()

# Pick date column
date_col = "date"
df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
df = df.sort_values(date_col)

# Pick HCHO columns (Dobson Units)
mean_col = "HCHO_Mean (DU)"
min_col  = "HCHO_Min (DU)"
max_col  = "HCHO_Max (DU)"

# Convert to numeric
for col in [mean_col, min_col, max_col]:
    df[col] = pd.to_numeric(df[col], errors="coerce")

# ------------------------------
# Clean Data: remove invalid/extreme values
# ------------------------------
df = df[(df[mean_col] >= 0) & (df[mean_col] <= 5)]
df = df.dropna(subset=[mean_col, min_col, max_col])

print(f"Cleaned dataset: {len(df)} valid rows remain")
print("Value ranges:")
print(df[[mean_col, min_col, max_col]].describe())

# ------------------------------
# 1. Time Series with Min-Max Shading
# ------------------------------
plt.figure(figsize=(12,6))
plt.plot(df[date_col], df[mean_col], label="Mean HCHO (DU)", color="blue")
plt.fill_between(df[date_col], df[min_col], df[max_col],
                 color="skyblue", alpha=0.3, label="Min-Max Range")

plt.title("Daily Formaldehyde (HCHO) Column Amount (Cleaned)")
plt.xlabel("Date")
plt.ylabel("HCHO (Dobson Units)")
plt.legend()
plt.grid(True)
plt.savefig(os.path.join(out_dir, "time_series.png"), dpi=300, bbox_inches="tight")
plt.close()

# ------------------------------
# 2. Histogram
# ------------------------------
plt.figure(figsize=(8,5))
plt.hist(df[mean_col].dropna(), bins=30, color="green", alpha=0.7)
plt.title("Distribution of Daily Mean HCHO (Cleaned)")
plt.xlabel("HCHO (Dobson Units)")
plt.ylabel("Frequency")
plt.grid(True)
plt.savefig(os.path.join(out_dir, "histogram.png"), dpi=300, bbox_inches="tight")
plt.close()

# ------------------------------
# 3. Rolling Average (7-day)
# ------------------------------
df["HCHO_7day_MA"] = df[mean_col].rolling(window=7).mean()

plt.figure(figsize=(12,6))
plt.plot(df[date_col], df[mean_col], label="Daily Mean", alpha=0.5)
plt.plot(df[date_col], df["HCHO_7day_MA"], label="7-day Moving Avg", color="red", linewidth=2)

plt.title("Smoothed Trend of HCHO (7-day Moving Avg, Cleaned)")
plt.xlabel("Date")
plt.ylabel("HCHO (Dobson Units)")
plt.legend()
plt.grid(True)
plt.savefig(os.path.join(out_dir, "rolling_average.png"), dpi=300, bbox_inches="tight")
plt.close()

# ------------------------------
# Save cleaned CSV
# ------------------------------
cleaned_csv = os.path.join(out_dir, f"{base_name}_cleaned.csv")
df.to_csv(cleaned_csv, index=False)

print(f"\nAll results saved in: {out_dir}")
