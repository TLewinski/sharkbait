import pandas as pd

# Load the shark attack dataset
df = pd.read_csv("data/global_shark_attacks.csv")

# Show the first 5 rows
print("FIRST 5 ROWS:")
print(df.head())

# Show number of rows and columns
print("\nDATASET SIZE:")
print(df.shape)

# Show all column names
print("\nCOLUMN NAMES:")
print(df.columns.tolist())

# Show information about each column
print("\nDATASET INFO:")
df.info()