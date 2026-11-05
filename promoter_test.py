import pandas as pd
from folding import fold_dna

df = pd.read_parquet("testdata/test.parquet")

# 20 sequences per label, same ones every run
sample = df.groupby("label").sample(20, random_state=0).copy()

# fold each one and keep the MFE
sample["mfe"] = [fold_dna(s)[1] for s in sample["sequence"]]
sample["gc"] = sample["sequence"].apply(lambda s: (s.count("G") + s.count("C")) / len(s))

print(sample.groupby("label")[["mfe", "gc"]].describe().round(2).T)
print("\nOne example:")
row = sample.iloc[0]
print(row["name"], "label", row["label"], "MFE", round(row["mfe"], 2))
print(row["sequence"])
print(fold_dna(row["sequence"])[0])