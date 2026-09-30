import os
import re
import pandas as pd
import lightgbm as lgb
from rapidfuzz import fuzz

BASE = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))

S1_PATH = BASE + r"\dataset\test\test_source1.tsv"
S2_PATH = BASE + r"\dataset\test\test_source2.tsv"
S3_PATH = BASE + r"\dataset\test\test_source3.tsv"

CANDIDATE_PATH = BASE + r"\output\candidate_pairs.tsv"
MODEL_PATH = BASE + r"\code\business_entity_resolution\model\entity_matching_model.txt"
FEATURE_PATH = BASE + r"\code\business_entity_resolution\model\model_features.txt"
OUTPUT_PATH = BASE + r"\output\scored_candidates.tsv"

THRESHOLD = 0.75
CHUNK_SIZE = 50000


def normalize(text):
    if pd.isna(text):
        return ""
    text = str(text).lower()
    text = text.replace("&", " and ")
    text = re.sub(r"[^\w\s]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def compact(text):
    return normalize(text).replace(" ", "")


def tokens(text):
    return set(normalize(text).split())


def numbers(text):
    return set(re.findall(r"\d+", str(text)))


def safe_ratio(a, b):
    if not a or not b:
        return 0.0
    return fuzz.ratio(a, b) / 100.0


def make_features(
    s1_name,
    s1_addr,
    s1_country,
    t_name,
    t_addr,
    t_country,
    target_source
):

    n1 = normalize(s1_name)
    n2 = normalize(t_name)

    a1 = normalize(s1_addr)
    a2 = normalize(t_addr)

    t1 = tokens(n1)
    t2 = tokens(n2)

    at1 = tokens(a1)
    at2 = tokens(a2)

    num1 = numbers(a1)
    num2 = numbers(a2)

    name_ratio = safe_ratio(n1, n2)

    name_token_sort = (
        fuzz.token_sort_ratio(n1, n2) / 100.0
        if n1 and n2 else 0.0
    )

    name_token_set = (
        fuzz.token_set_ratio(n1, n2) / 100.0
        if n1 and n2 else 0.0
    )

    name_partial = (
        fuzz.partial_ratio(n1, n2) / 100.0
        if n1 and n2 else 0.0
    )

    name_jaccard = (
        len(t1 & t2) / len(t1 | t2)
        if t1 | t2 else 0.0
    )

    address_ratio = safe_ratio(a1, a2)

    address_token_sort = (
        fuzz.token_sort_ratio(a1, a2) / 100.0
        if a1 and a2 else 0.0
    )

    address_token_set = (
        fuzz.token_set_ratio(a1, a2) / 100.0
        if a1 and a2 else 0.0
    )

    address_partial = (
        fuzz.partial_ratio(a1, a2) / 100.0
        if a1 and a2 else 0.0
    )

    address_jaccard = (
        len(at1 & at2) / len(at1 | at2)
        if at1 | at2 else 0.0
    )

    exact_name = int(
        n1 != "" and n1 == n2
    )

    exact_compact_name = int(
        compact(s1_name) != "" and
        compact(s1_name) == compact(t_name)
    )

    exact_address = int(
        a1 != "" and a1 == a2
    )

    number_match = int(
        bool(num1 & num2)
    )

    number_intersection = len(
        num1 & num2
    )

    country_match = int(
        normalize(s1_country) != "" and
        normalize(s1_country) == normalize(t_country)
    )

    name_missing_1 = int(n1 == "")
    name_missing_2 = int(n2 == "")

    address_missing_1 = int(a1 == "")
    address_missing_2 = int(a2 == "")

    name_length_diff = abs(
        len(n1) - len(n2)
    )

    address_length_diff = abs(
        len(a1) - len(a2)
    )

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
        best_name_score +
        best_address_score
    ) / 2.0

    # EXACT ORDER FROM model_features.txt
    return [
        0 if target_source == "S2" else 1,
        name_ratio,
        name_token_sort,
        name_token_set,
        name_partial,
        name_jaccard,
        address_ratio,
        address_token_sort,
        address_token_set,
        address_partial,
        address_jaccard,
        exact_name,
        exact_compact_name,
        exact_address,
        number_match,
        number_intersection,
        country_match,
        name_missing_1,
        name_missing_2,
        address_missing_1,
        address_missing_2,
        name_length_diff,
        address_length_diff,
        best_name_score,
        best_address_score,
        combined_score
    ]


print("=" * 70)
print("STEP 6C - ML CANDIDATE SCORING")
print("=" * 70)

# ---------------------------------------------------------
# LOAD MODEL
# ---------------------------------------------------------

print("\nLoading model...")

model = lgb.Booster(
    model_file=MODEL_PATH
)

print("Model loaded.")


# ---------------------------------------------------------
# LOAD FEATURE NAMES
# ---------------------------------------------------------

print("\nLoading feature names...")

with open(
    FEATURE_PATH,
    "r",
    encoding="utf-8"
) as f:

    feature_names = [
        x.strip()
        for x in f.read().splitlines()
        if x.strip()
    ]

print("Features:", len(feature_names))

if len(feature_names) != 26:
    raise ValueError(
        f"Expected 26 features, found {len(feature_names)}"
    )


# ---------------------------------------------------------
# LOAD S1
# ---------------------------------------------------------

print("\nLoading S1...")

s1 = pd.read_csv(
    S1_PATH,
    sep="\t",
    dtype=str,
    usecols=[
        "entity_id",
        "business_name",
        "business_address",
        "country"
    ]
)

s1 = s1.set_index("entity_id")

print(
    "S1 loaded:",
    len(s1)
)


# ---------------------------------------------------------
# LOAD S2
# ---------------------------------------------------------

print("\nLoading S2...")

s2 = pd.read_csv(
    S2_PATH,
    sep="\t",
    dtype=str,
    usecols=[
        "entity_id",
        "business_name",
        "business_address",
        "country"
    ]
)

s2 = s2.set_index("entity_id")

print(
    "S2 loaded:",
    len(s2)
)


# ---------------------------------------------------------
# LOAD S3
# ---------------------------------------------------------

print("\nLoading S3...")

s3 = pd.read_csv(
    S3_PATH,
    sep="\t",
    dtype=str,
    usecols=[
        "entity_id",
        "business_name",
        "business_address",
        "country"
    ]
)

s3 = s3.set_index("entity_id")

print(
    "S3 loaded:",
    len(s3)
)


# ---------------------------------------------------------
# COMBINE TARGETS
# ---------------------------------------------------------

target_lookup = pd.concat(
    [s2, s3]
)

del s2
del s3

print(
    "Total target entities:",
    len(target_lookup)
)


# ---------------------------------------------------------
# REMOVE OLD OUTPUT IF EXISTS
# ---------------------------------------------------------

if os.path.exists(OUTPUT_PATH):
    os.remove(OUTPUT_PATH)

print("\nStarting candidate scoring...")

first = True

total_candidates = 0
total_matches = 0

chunk_no = 0


# ---------------------------------------------------------
# PROCESS CANDIDATES IN CHUNKS
# ---------------------------------------------------------

for chunk in pd.read_csv(
    CANDIDATE_PATH,
    sep="\t",
    dtype=str,
    chunksize=CHUNK_SIZE
):

    chunk_no += 1

    rows = []

    for _, row in chunk.iterrows():

        s1_id = row["source1_entity_id"]

        candidate_text = row[
            "candidate_entity_ids"
        ]

        if (
            pd.isna(candidate_text)
            or not candidate_text
        ):
            continue

        if s1_id not in s1.index:
            continue

        s1row = s1.loc[s1_id]

        candidate_ids = candidate_text.split(",")

        for target_id in candidate_ids:

            target_id = str(
                target_id
            ).strip()

            if not target_id:
                continue

            if target_id not in target_lookup.index:
                continue

            trow = target_lookup.loc[
                target_id
            ]

            if target_id.startswith("S2-"):
                target_source = "S2"
            else:
                target_source = "S3"

            feats = make_features(
                s1row["business_name"],
                s1row["business_address"],
                s1row["country"],

                trow["business_name"],
                trow["business_address"],
                trow["country"],

                target_source
            )

            rows.append(
                [
                    s1_id,
                    target_id
                ] + feats
            )


    # -----------------------------------------------------
    # PREDICT
    # -----------------------------------------------------

    if rows:

        feature_df = pd.DataFrame(
            [r[2:] for r in rows],
            columns=feature_names
        )

        predictions = model.predict(
            feature_df
        )

        output_rows = []

        for r, score in zip(
            rows,
            predictions
        ):

            total_candidates += 1

            if float(score) >= THRESHOLD:

                total_matches += 1

                output_rows.append(
                    [
                        r[0],
                        r[1],
                        float(score)
                    ]
                )


        # -------------------------------------------------
        # WRITE ACCEPTED MATCHES
        # -------------------------------------------------

        if output_rows:

            out_df = pd.DataFrame(
                output_rows,
                columns=[
                    "source1_entity_id",
                    "target_entity_id",
                    "match_probability"
                ]
            )

            out_df.to_csv(
                OUTPUT_PATH,
                sep="\t",
                index=False,
                mode="w" if first else "a",
                header=first
            )

            first = False


    # -----------------------------------------------------
    # PROGRESS
    # -----------------------------------------------------

    if chunk_no % 5 == 0:

        print(
            f"Processed candidates: "
            f"{total_candidates:,} | "
            f"Accepted matches: "
            f"{total_matches:,}"
        )


# ---------------------------------------------------------
# COMPLETE
# ---------------------------------------------------------

print("\n" + "=" * 70)
print("STEP 6C COMPLETE")
print("=" * 70)

print(
    "Candidates scored:",
    f"{total_candidates:,}"
)

print(
    "Matches >= threshold:",
    f"{total_matches:,}"
)

print(
    "Threshold:",
    THRESHOLD
)

print(
    "Output:",
    OUTPUT_PATH
)

print("=" * 70)

