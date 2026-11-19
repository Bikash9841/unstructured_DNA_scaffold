import random

import pandas as pd

from reward import terms, total_multiplicative

PARQUET = "testdata/test.parquet"
PER_GROUP = 100
SEED = 0


# random_dna
# Input: n, how many letters. seed, which random sequence to build. alphabet,
# which letters to draw from ("ACGT" for normal DNA, "AC" for the exploit).
# Output: a DNA string of length n.
# Purpose: builds the control sequences we compare the real promoters against.
def random_dna(n, seed, alphabet="ACGT"):
    rng = random.Random(seed)
    return "".join(rng.choice(alphabet) for _ in range(n))


# build_groups
# Input: df, the promoter table loaded from the parquet file. n, how many
# sequences to take from each group. seed, for the random controls.
# Output: a dict mapping a group name to its list of DNA strings.
# Purpose: puts the real promoter sequences next to the controls we need, so one
# table shows natural DNA, random DNA and the two known exploits side by side.
def build_groups(df, n, seed):
    length = int(df["sequence"].str.len().median())
    groups = {}
    for label, part in df.groupby("label"):
        groups[f"natural label {label}"] = part.sample(n, random_state=seed)["sequence"].tolist()
    groups["random DNA"] = [random_dna(length, seed + i) for i in range(n)]
    groups["A/C-only (exploit)"] = [random_dna(length, seed + i, "AC") for i in range(n)]
    groups["poly-A (exploit)"] = ["A" * length]
    return groups


# score_groups
# Input: groups, the dict from build_groups.
# Output: a DataFrame with one row per sequence, holding the group name, the four
# term scores and the final reward.
# Purpose: runs the real reward function over every sequence. This is the slow
# step, because each sequence gets folded once by ViennaRNA.
def score_groups(groups):
    rows = []
    for name, seqs in groups.items():
        for seq in seqs:
            folding, complexity, composition, repeat = terms(seq)
            rows.append({
                "group": name,
                "length": len(seq),
                "folding": folding,
                "complexity": complexity,
                "composition": composition,
                "repeat": repeat,
                "reward": total_multiplicative(seq),
            })
    return pd.DataFrame(rows)


# summarize
# Input: scored, the DataFrame from score_groups.
# Output: a DataFrame with one row per group, holding the mean of each term, the
# mean reward and the lowest and highest reward in that group.
# Purpose: the table to read. It shows which term pulls each group's reward down.
def summarize(scored):
    g = scored.groupby("group", sort=False)
    out = g[["folding", "complexity", "composition", "repeat", "reward"]].mean()
    out["reward min"] = g["reward"].min()
    out["reward max"] = g["reward"].max()
    return out.round(3)


# main
# Input: nothing. Reads the parquet file named at the top of this script.
# Output: prints the summary table and writes reward_evaluation.csv.
# Purpose: the whole evaluation in one command: python evaluate_reward.py
def main():
    df = pd.read_parquet(PARQUET)
    groups = build_groups(df, PER_GROUP, SEED)
    scored = score_groups(groups)
    scored.to_csv("reward_evaluation.csv", index=False)
    print(summarize(scored).to_string())
    print("\nwrote reward_evaluation.csv")


if __name__ == "__main__":
    main()
