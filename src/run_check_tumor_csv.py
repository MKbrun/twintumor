from src.io.tumor_volume_csv import load_tumor_volume_csv

CSV_PATH = "data/raw/Mets_010/remission/tumor_volume.csv"

df = load_tumor_volume_csv(CSV_PATH)

print("Columns:", df.columns.tolist())
print()
print(df.head())
print()
print(df.describe(include="all"))