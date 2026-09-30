import pandas as pd
import re
import os
import time
from collections import defaultdict

BASE = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))

S1_PATH = BASE + r"\dataset\test\test_source1.tsv"
S2_PATH = BASE + r"\dataset\test\test_source2.tsv"
S3_PATH = BASE + r"\dataset\test\test_source3.tsv"

OUTPUT_PATH = BASE + r"\output\candidate_pairs.tsv"

CHUNK_SIZE = 200000
TOP_K = 100


def normalize(text):
    if pd.isna(text):
        return ""

    text = str(text).lower()
    text = text.replace("&", " and ")

    text = re.sub(r"[^\w\s]", " ", text)
    text = re.sub(r"\s+", " ", text)

    return text.strip()


def build_index(path, source, index):

    print("Indexing", source)

    total = 0

    for chunk in pd.read_csv(
        path,
        sep="\t",
        dtype=str,
        chunksize=CHUNK_SIZE
    ):

        for _, row in chunk.iterrows():

            entity_id = row["entity_id"]

            name = normalize(
                row.get("business_name", "")
            )

            if name:
                index[name].append(entity_id)

            total += 1

        print(
            source,
            "processed:",
            f"{total:,}"
        )

    return total


# =========================================================
# BUILD TARGET NAME INDEX
# =========================================================

print("=" * 60)
print("FAST EXACT-NAME CANDIDATE GENERATION")
print("=" * 60)

start = time.time()

name_index = defaultdict(list)

build_index(
    S2_PATH,
    "S2",
    name_index
)

build_index(
    S3_PATH,
    "S3",
    name_index
)

print(
    "\nUnique normalized names:",
    f"{len(name_index):,}"
)

print(
    "Indexing time:",
    round(time.time() - start, 1),
    "seconds"
)


# =========================================================
# PROCESS S1
# =========================================================

print("\nLoading S1...")

s1 = pd.read_csv(
    S1_PATH,
    sep="\t",
    dtype=str
)

print(
    "S1 rows:",
    f"{len(s1):,}"
)


os.makedirs(
    os.path.dirname(OUTPUT_PATH),
    exist_ok=True
)

out = open(
    OUTPUT_PATH,
    "w",
    encoding="utf-8"
)

out.write(
    "source1_entity_id\tcandidate_entity_ids\n"
)


total_candidates = 0
zero_candidates = 0

start = time.time()

for i, row in s1.iterrows():

    s1_id = row["entity_id"]

    name = normalize(
        row["business_name"]
    )

    candidates = name_index.get(
        name,
        []
    )

    candidates = candidates[:TOP_K]

    if not candidates:
        zero_candidates += 1

    total_candidates += len(candidates)

    out.write(
        str(s1_id)
        + "\t"
        + ",".join(candidates)
        + "\n"
    )

    if (i + 1) % 100000 == 0:

        elapsed = time.time() - start

        print(
            f"Completed {i + 1:,}/{len(s1):,} | "
            f"Time: {elapsed:.1f}s"
        )


out.close()

elapsed = time.time() - start

print("\n" + "=" * 60)
print("STEP 6B COMPLETE")
print("=" * 60)

print(
    "S1 rows:",
    f"{len(s1):,}"
)

print(
    "Total candidates:",
    f"{total_candidates:,}"
)

print(
    "Average candidates:",
    round(
        total_candidates / len(s1),
        2
    )
)

print(
    "Zero candidates:",
    f"{zero_candidates:,}"
)

print(
    "Candidate generation time:",
    round(elapsed, 1),
    "seconds"
)

print(
    "Output:",
    OUTPUT_PATH
)

print("=" * 60)
