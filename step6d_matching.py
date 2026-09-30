import pandas as pd
import os

BASE = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))

S1_PATH = BASE + r"\dataset\test\test_source1.tsv"
SCORED_PATH = BASE + r"\output\scored_candidates.tsv"
OUTPUT_PATH = BASE + r"\output\matching_results.tsv"

print("=" * 70)
print("STEP 6D - GENERATING MATCHING RESULTS")
print("=" * 70)

print("\nLoading S1...")
s1 = pd.read_csv(
    S1_PATH,
    sep="\t",
    dtype=str,
    usecols=["entity_id"]
)

print("S1 rows:", f"{len(s1):,}")

print("\nLoading scored candidates...")

scored = pd.read_csv(
    SCORED_PATH,
    sep="\t",
    dtype={
        "source1_entity_id": str,
        "target_entity_id": str,
        "match_probability": float
    }
)

print(
    "Scored rows:",
    f"{len(scored):,}"
)

# Remove duplicate S1-target pairs
scored = scored.drop_duplicates(
    subset=[
        "source1_entity_id",
        "target_entity_id"
    ]
)

# Sort matches by probability
scored = scored.sort_values(
    [
        "source1_entity_id",
        "match_probability"
    ],
    ascending=[
        True,
        False
    ]
)

print(
    "Unique scored pairs:",
    f"{len(scored):,}"
)

# Group target IDs for each S1
grouped = (
    scored
    .groupby("source1_entity_id")["target_entity_id"]
    .apply(
        lambda x: ",".join(
            dict.fromkeys(x.tolist())
        )
    )
)

# Create final result for EVERY S1 entity
result = pd.DataFrame({
    "source1_entity_id": s1["entity_id"]
})

result["matched_entity_ids"] = (
    result["source1_entity_id"]
    .map(grouped)
    .fillna("")
)

# Safety: remove duplicate S1 rows
result = result.drop_duplicates(
    subset=["source1_entity_id"]
)

# Write output
result.to_csv(
    OUTPUT_PATH,
    sep="\t",
    index=False
)

print("\n" + "=" * 70)
print("STEP 6D COMPLETE")
print("=" * 70)

print(
    "S1 rows:",
    f"{len(result):,}"
)

print(
    "Rows with matches:",
    f"{(result['matched_entity_ids'] != '').sum():,}"
)

print(
    "Rows with no matches:",
    f"{(result['matched_entity_ids'] == '').sum():,}"
)

print(
    "Output:",
    OUTPUT_PATH
)

print("=" * 70)
