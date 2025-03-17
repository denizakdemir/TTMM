from tabular_transformer.examples.data_utils import download_wine_quality_dataset

# Download both red and white wine datasets
print("Red wine dataset columns:")
wine_red = download_wine_quality_dataset(save_csv=False, variant='red')
print(wine_red.columns.tolist())

print("\nWhite wine dataset columns:")
wine_white = download_wine_quality_dataset(save_csv=False, variant='white')
print(wine_white.columns.tolist())

# Check if 'quality' columns exist
print("\nQuality column in red wine?", 'quality' in wine_red.columns)
print("Quality column in white wine?", 'quality' in wine_white.columns)

# Look for similar column names
for col in wine_red.columns:
    if 'qual' in col.lower():
        print(f"Found potential quality column in red wine: {col}")

for col in wine_white.columns:
    if 'qual' in col.lower():
        print(f"Found potential quality column in white wine: {col}")
