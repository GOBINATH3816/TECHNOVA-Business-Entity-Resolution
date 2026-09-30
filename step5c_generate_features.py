import pandas as pd
import re
import os
from rapidfuzz import fuzz

# ============================================================
# CONFIG
# ============================================================

BASE = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))

S1_FILE = BASE + r"\dataset\train\train_source1.tsv"
S2_FILE = BASE + r"\dataset\train\train_source2.tsv"
S3_FILE = BASE + r"\dataset\train\train_source3.tsv"

PAIR_FILE = BASE + r"\output\training_pairs.tsv"

OUTPUT_FILE = BASE + r"\output\training_features.tsv"


# ============================================================
# NORMALIZATION
# ============================================================

def normalize_text(value):
    if pd.isna(value):
        return ""

    text = str(value).lower().strip()

    text = text.replace("&", " and ")

    chars = []

    for ch in text:
        if ch.isalnum():
            chars.append(ch)
        else:
            chars.append(" ")

    text = "".join(chars)

    text = re.sub(r"\s+", " ", text).strip()

    return text


def compact_text(value):
    return normalize_text(value).replace(" ", "")


def get_numbers(value):
    if pd.isna(value):
        return set()

    return set(
        re.findall(r"\d+", str(value))
    )


def token_jaccard(a, b):
    if not a or not b:
        return 0.0

    a_set = set(a.split())
    b_set = set(b.split())

    if not a_set or not b_set:
        return 0.0

    intersection = len(a_set & b_set)
    union = len(a_set | b_set)

    return intersection / union


# ============================================================
# LOAD TRAINING PAIRS
# ============================================================

print("=" * 70)
print("STEP 5C - FEATURE GENERATION")
print("=" * 70)

print("\nLoading training pairs...")

pairs = pd.read_csv(
    PAIR_FILE,
    sep="\t",
    dtype=str
)

print("Training pairs:", len(pairs))


# ============================================================
# COLLECT REQUIRED IDS
# ============================================================

s1_ids = set(
    pairs["source1_entity_id"].astype(str)
)

target_ids = set(
    pairs["target_entity_id"].astype(str)
)

print("Required S1 IDs:", len(s1_ids))
print("Required target IDs:", len(target_ids))


# ============================================================
# LOAD REQUIRED S1 RECORDS
# ============================================================

print()
print("Loading required Source 1 records...")

s1_parts = []

for chunk in pd.read_csv(
    S1_FILE,
    sep="\t",
    dtype=str,
    chunksize=300000
):

    chunk["entity_id"] = chunk["entity_id"].astype(str)

    selected = chunk[
        chunk["entity_id"].isin(s1_ids)
    ]

    if len(selected) > 0:
        s1_parts.append(selected)

s1_data = pd.concat(
    s1_parts,
    ignore_index=True
)

print("S1 records loaded:", len(s1_data))


# ============================================================
# LOAD REQUIRED TARGET RECORDS
# ============================================================

print()
print("Loading required target records...")

target_parts = []

for file in [S2_FILE, S3_FILE]:

    print("Scanning:", os.path.basename(file))

    for chunk in pd.read_csv(
        file,
        sep="\t",
        dtype=str,
        chunksize=300000
    ):

        chunk["entity_id"] = chunk["entity_id"].astype(str)

        selected = chunk[
            chunk["entity_id"].isin(target_ids)
        ]

        if len(selected) > 0:
            target_parts.append(selected)

target_data = pd.concat(
    target_parts,
    ignore_index=True
)

print("Target records loaded:", len(target_data))


# ============================================================
# CREATE LOOKUPS
# ============================================================

print()
print("Creating lookups...")

s1_lookup = {}

for _, row in s1_data.iterrows():

    entity_id = str(row["entity_id"])

    s1_lookup[entity_id] = {
        "name": row.get("business_name", ""),
        "address": row.get("business_address", ""),
        "country": row.get("country", "")
    }


target_lookup = {}

for _, row in target_data.iterrows():

    entity_id = str(row["entity_id"])

    target_lookup[entity_id] = {
        "name": row.get("business_name", ""),
        "address": row.get("business_address", ""),
        "country": row.get("country", "")
    }


print("S1 lookup:", len(s1_lookup))
print("Target lookup:", len(target_lookup))


# ============================================================
# FEATURE GENERATION
# ============================================================

print()
print("=" * 70)
print("CALCULATING FEATURES")
print("=" * 70)

feature_rows = []

total = len(pairs)

for index, row in pairs.iterrows():

    s1_id = str(row["source1_entity_id"])
    target_id = str(row["target_entity_id"])
    label = int(row["label"])

    s1_record = s1_lookup.get(
        s1_id,
        {}
    )

    target_record = target_lookup.get(
        target_id,
        {}
    )

    # --------------------------------------------------------
    # Raw values
    # --------------------------------------------------------

    name1 = str(
        s1_record.get("name", "")
        if s1_record.get("name", "") is not None
        else ""
    )

    name2 = str(
        target_record.get("name", "")
        if target_record.get("name", "") is not None
        else ""
    )

    address1 = str(
        s1_record.get("address", "")
        if s1_record.get("address", "") is not None
        else ""
    )

    address2 = str(
        target_record.get("address", "")
        if target_record.get("address", "") is not None
        else ""
    )

    country1 = normalize_text(
        s1_record.get("country", "")
    )

    country2 = normalize_text(
        target_record.get("country", "")
    )

    # --------------------------------------------------------
    # Normalized values
    # --------------------------------------------------------

    n_name1 = normalize_text(name1)
    n_name2 = normalize_text(name2)

    n_address1 = normalize_text(address1)
    n_address2 = normalize_text(address2)

    c_name1 = compact_text(name1)
    c_name2 = compact_text(name2)

    # --------------------------------------------------------
    # NAME FEATURES
    # --------------------------------------------------------

    name_ratio = fuzz.ratio(
        n_name1,
        n_name2
    ) if n_name1 and n_name2 else 0

    name_token_sort = fuzz.token_sort_ratio(
        n_name1,
        n_name2
    ) if n_name1 and n_name2 else 0

    name_token_set = fuzz.token_set_ratio(
        n_name1,
        n_name2
    ) if n_name1 and n_name2 else 0

    name_partial = fuzz.partial_ratio(
        n_name1,
        n_name2
    ) if n_name1 and n_name2 else 0

    # --------------------------------------------------------
    # ADDRESS FEATURES
    # --------------------------------------------------------

    address_ratio = fuzz.ratio(
        n_address1,
        n_address2
    ) if n_address1 and n_address2 else 0

    address_token_sort = fuzz.token_sort_ratio(
        n_address1,
        n_address2
    ) if n_address1 and n_address2 else 0

    address_token_set = fuzz.token_set_ratio(
        n_address1,
        n_address2
    ) if n_address1 and n_address2 else 0

    address_partial = fuzz.partial_ratio(
        n_address1,
        n_address2
    ) if n_address1 and n_address2 else 0

    # --------------------------------------------------------
    # TOKEN JACCARD
    # --------------------------------------------------------

    name_jaccard = token_jaccard(
        n_name1,
        n_name2
    )

    address_jaccard = token_jaccard(
        n_address1,
        n_address2
    )

    # --------------------------------------------------------
    # EXACT FEATURES
    # --------------------------------------------------------

    exact_name = int(
        bool(n_name1)
        and bool(n_name2)
        and n_name1 == n_name2
    )

    exact_compact_name = int(
        bool(c_name1)
        and bool(c_name2)
        and c_name1 == c_name2
    )

    exact_address = int(
        bool(n_address1)
        and bool(n_address2)
        and n_address1 == n_address2
    )

    # --------------------------------------------------------
    # ADDRESS NUMBER FEATURES
    # --------------------------------------------------------

    numbers1 = get_numbers(address1)
    numbers2 = get_numbers(address2)

    number_intersection = len(
        numbers1 & numbers2
    )

    number_match = int(
        len(numbers1) > 0
        and len(numbers2) > 0
        and bool(numbers1 & numbers2)
    )

    # --------------------------------------------------------
    # COUNTRY
    # --------------------------------------------------------

    country_match = int(
        bool(country1)
        and bool(country2)
        and country1 == country2
    )

    # --------------------------------------------------------
    # MISSING DATA
    # --------------------------------------------------------

    name_missing_1 = int(not bool(n_name1))
    name_missing_2 = int(not bool(n_name2))

    address_missing_1 = int(not bool(n_address1))
    address_missing_2 = int(not bool(n_address2))

    # --------------------------------------------------------
    # LENGTH FEATURES
    # --------------------------------------------------------

    name_length_diff = abs(
        len(n_name1) - len(n_name2)
    )

    address_length_diff = abs(
        len(n_address1) - len(n_address2)
    )

    # --------------------------------------------------------
    # COMBINED FEATURES
    # --------------------------------------------------------

    best_name_score = max(
        name_ratio,
        name_token_sort,
        name_token_set
    )

    best_address_score = max(
        address_ratio,
        address_token_sort,
        address_token_set
    )

    combined_score = (
        0.55 * best_name_score
        + 0.45 * best_address_score
    )

    # --------------------------------------------------------
    # SOURCE
    # --------------------------------------------------------

    target_source = (
        "S2"
        if target_id.startswith("S2-")
        else "S3"
    )

    # --------------------------------------------------------
    # STORE
    # --------------------------------------------------------

    feature_rows.append({

        "source1_entity_id": s1_id,
        "target_entity_id": target_id,

        "label": label,

        "target_source": target_source,

        # Name
        "name_ratio": name_ratio,
        "name_token_sort": name_token_sort,
        "name_token_set": name_token_set,
        "name_partial": name_partial,
        "name_jaccard": name_jaccard,

        # Address
        "address_ratio": address_ratio,
        "address_token_sort": address_token_sort,
        "address_token_set": address_token_set,
        "address_partial": address_partial,
        "address_jaccard": address_jaccard,

        # Exact
        "exact_name": exact_name,
        "exact_compact_name": exact_compact_name,
        "exact_address": exact_address,

        # Address numbers
        "number_match": number_match,
        "number_intersection": number_intersection,

        # Country
        "country_match": country_match,

        # Missing
        "name_missing_1": name_missing_1,
        "name_missing_2": name_missing_2,
        "address_missing_1": address_missing_1,
        "address_missing_2": address_missing_2,

        # Length
        "name_length_diff": name_length_diff,
        "address_length_diff": address_length_diff,

        # Combined
        "best_name_score": best_name_score,
        "best_address_score": best_address_score,
        "combined_score": combined_score
    })

    if (index + 1) % 5000 == 0:
        print(
            "Processed:",
            index + 1,
            "/",
            total
        )


# ============================================================
# SAVE
# ============================================================

features = pd.DataFrame(feature_rows)

features.to_csv(
    OUTPUT_FILE,
    sep="\t",
    index=False
)


# ============================================================
# SUMMARY
# ============================================================

print()
print("=" * 70)
print("STEP 5C COMPLETE")
print("=" * 70)

print("Feature rows:", len(features))
print("Feature columns:", len(features.columns))

print()
print("Positive examples:")
print(
    int((features["label"] == 1).sum())
)

print("Negative examples:")
print(
    int((features["label"] == 0).sum())
)

print()
print("Saved to:")
print(OUTPUT_FILE)

print()
print("Feature columns:")
print(
    list(features.columns)
)

print()
print(features.head(5).to_string(index=False))

print()
print("=" * 70)
print("DONE")
print("=" * 70)
