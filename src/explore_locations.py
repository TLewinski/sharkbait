import pandas as pd


# Load cleaned data
df = pd.read_csv("data/clean_attacks.csv")


print("COUNTRIES")
print("---------")

print(
    df["country"]
    .value_counts()
    .head(20)
)


print("\n\nAREAS")
print("-----")

print(
    df["area"]
    .value_counts()
    .head(20)
)


print("\n\nLOCATIONS")
print("---------")

print(
    df["location"]
    .value_counts()
    .head(30)
)


print("\n\nEXAMPLE RECORDS")
print("----------------")

print(
    df[
        ["country", "area", "location"]
    ].head(20).to_string(index=False)
)