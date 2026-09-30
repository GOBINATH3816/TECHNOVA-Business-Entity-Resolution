import pandas as pd
import random
import os

# ============================================================
# CONFIG
# ============================================================

BASE = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))

S1_FILE = BASE + r"\dataset\train\train_source1.tsv"
S2_FILE = BASE + r"\dataset\train\train_source2.tsv"
S3_FILE = BASE + r"\dataset\train\train_source3.tsv"
GT_FILE = BASE + r"\dataset\train\train_ground_truth.tsv"

OUTPUT_DIR = BASE + r"\output"
os.makedirs(OUTPUT_DIR, exist_ok=True)

OUTPUT_FILE = OUTPUT_DIR + r"\training_pairs.tsv"

MAX_S1 = 10000
MAX_POSITIVE = 20000
NEGATIVE_PER_POSITIVE = 2
RANDOM_SEED = 42

random.seed(RANDOM_SEED)

print("=" * 70)
print("STEP 5B - CREATE TRAINING PAIRS (OPTIMIZED)")
print("=" * 70)


# ============================================================
# LOAD GROUND TRUTH
# ============================================================

print("\nLoading ground truth...")

gt = pd.read_csv(
    GT_FILE,
    sep="\t",
    dtype=str
)

gt["matched_entity_ids"] = gt["matched_entity_ids"].fillna("")

gt["match_count"] = gt["matched_entity_ids"].apply(
    lambda x: len([v for v in str(x).split(",") if v.strip()])
)

matched_gt = gt[gt["match_count"] > 0].copy()

print("Matched S1 entities:", len(matched_gt))


# ============================================================
# SAMPLE S1
# ============================================================

if len(matched_gt) > MAX_S1:
    matched_gt = matched_gt.sample(
        n=MAX_S1,
        random_state=RANDOM_SEED
    )

print("Selected S1:", len(matched_gt))


# ============================================================
# CREATE POSITIVE PAIRS
# ============================================================

positive_records = []

for _, row in matched_gt.iterrows():

    s1_id = str(row["source1_entity_id"])

    ids = [
        x.strip()
        for x in str(row["matched_entity_ids"]).split(",")
        if x.strip()
    ]

    for target_id in ids:

        positive_records.append(
            (s1_id, target_id, 1)
        )

        if len(positive_records) >= MAX_POSITIVE:
            break

    if len(positive_records) >= MAX_POSITIVE:
        break


print("Positive pairs:", len(positive_records))


# ============================================================
# S1 COUNTRY LOOKUP
# ============================================================

print("\nLoading S1 countries...")

s1_ids = set(
    matched_gt["source1_entity_id"].astype(str)
)

s1_country = {}

for chunk in pd.read_csv(
    S1_FILE,
    sep="\t",
    dtype=str,
    usecols=["entity_id", "country"],
    chunksize=300000
):

    chunk["entity_id"] = chunk["entity_id"].astype(str)

    selected = chunk[
        chunk["entity_id"].isin(s1_ids)
    ]

    for _, row in selected.iterrows():

        entity_id = str(row["entity_id"])

        country = str(
            row["country"]
        ).strip().lower()

        if country == "nan":
            country = ""

        s1_country[entity_id] = country


print("S1 countries loaded:", len(s1_country))


# ============================================================
# BUILD COUNTRY BUCKETS
# ============================================================

print()
print("=" * 70)
print("BUILDING TARGET COUNTRY BUCKETS")
print("=" * 70)

# country -> list of entity IDs
country_buckets = {}

# All target IDs
all_target_ids = set()

for file in [S2_FILE, S3_FILE]:

    print("\nReading:", os.path.basename(file))

    for chunk in pd.read_csv(
        file,
        sep="\t",
        dtype=str,
        usecols=["entity_id", "country"],
        chunksize=300000
    ):

        chunk["entity_id"] = chunk["entity_id"].astype(str)

        chunk["country"] = (
            chunk["country"]
            .fillna("")
            .astype(str)
            .str.strip()
            .str.lower()
        )

        for entity_id, country in zip(
            chunk["entity_id"],
            chunk["country"]
        ):

            all_target_ids.add(entity_id)

            if country not in country_buckets:
                country_buckets[country] = []

            country_buckets[country].append(entity_id)

    print("Finished:", os.path.basename(file))


print()
print("Total target IDs:", len(all_target_ids))
print("Country buckets:", len(country_buckets))


# ============================================================
# POSITIVE TARGET SET
# ============================================================

positive_target_ids = set(
    target_id
    for _, target_id, _ in positive_records
)


# ============================================================
# CREATE NEGATIVES
# ============================================================

print()
print("=" * 70)
print("CREATING NEGATIVE PAIRS")
print("=" * 70)

negative_records = []

for index, (s1_id, positive_id, label) in enumerate(
    positive_records,
    start=1
):

    country = s1_country.get(
        s1_id,
        ""
    )

    # Same-country candidates
    candidates = country_buckets.get(
        country,
        []
    )

    # Remove known positive target IDs
    # We only need a small random sample,
    # so shuffle a small portion instead of
    # scanning the entire list.
    selected = []

    attempts = 0
    max_attempts = 30

    while (
        len(selected) < NEGATIVE_PER_POSITIVE
        and attempts < max_attempts
    ):

        if not candidates:
            break

        random_id = random.choice(candidates)

        if (
            random_id != positive_id
            and random_id not in positive_target_ids
            and random_id not in selected
        ):
            selected.append(random_id)

        attempts += 1

    # Fallback: random global target
    while len(selected) < NEGATIVE_PER_POSITIVE:

        random_id = random.choice(
            tuple(all_target_ids)
        )

        if (
            random_id != positive_id
            and random_id not in positive_target_ids
            and random_id not in selected
        ):
            selected.append(random_id)

    for negative_id in selected:

        negative_records.append(
            (
                s1_id,
                negative_id,
                0
            )
        )

    if index % 2000 == 0:
        print(
            "Processed positives:",
            index,
            "/",
            len(positive_records)
        )


# ============================================================
# COMBINE
# ============================================================

training_pairs = pd.DataFrame(
    positive_records + negative_records,
    columns=[
        "source1_entity_id",
        "target_entity_id",
        "label"
    ]
)


# ============================================================
# REMOVE DUPLICATES
# ============================================================

training_pairs = training_pairs.drop_duplicates(
    subset=[
        "source1_entity_id",
        "target_entity_id"
    ]
)


# ============================================================
# SHUFFLE
# ============================================================

training_pairs = training_pairs.sample(
    frac=1,
    random_state=RANDOM_SEED
).reset_index(drop=True)


# ============================================================
# SAVE
# ============================================================

training_pairs.to_csv(
    OUTPUT_FILE,
    sep="\t",
    index=False
)


# ============================================================
# SUMMARY
# ============================================================

print()
print("=" * 70)
print("TRAINING PAIRS CREATED")
print("=" * 70)

print("Total pairs:", len(training_pairs))

print(
    "Positive:",
    int((training_pairs["label"] == 1).sum())
)

print(
    "Negative:",
    int((training_pairs["label"] == 0).sum())
)

print()
print("Output:")
print(OUTPUT_FILE)

print()
print("First 10 rows:")
print(
    training_pairs.head(10).to_string(
        index=False
    )
)

print()
print("=" * 70)
print("STEP 5B COMPLETE")
print("=" * 70)
