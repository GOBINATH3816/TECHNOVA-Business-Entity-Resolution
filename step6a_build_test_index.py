import pandas as pd
import re
import os
import sqlite3

# ============================================================
# CONFIG
# ============================================================

BASE = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))

S2_FILE = BASE + r"\dataset\test\test_source2.tsv"
S3_FILE = BASE + r"\dataset\test\test_source3.tsv"

DB_FILE = BASE + r"\output\test_candidate_index.db"

CHUNK_SIZE = 200000


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


def get_pairs(text):
    tokens = text.split()

    if len(tokens) < 2:
        return []

    return [
        tokens[i] + "|" + tokens[i + 1]
        for i in range(len(tokens) - 1)
    ]


# ============================================================
# SETUP
# ============================================================

print("=" * 70)
print("STEP 6A - BUILD TEST CANDIDATE INDEX")
print("=" * 70)

os.makedirs(
    os.path.dirname(DB_FILE),
    exist_ok=True
)

if os.path.exists(DB_FILE):
    print("\nExisting database found.")
    print("Removing old database...")
    os.remove(DB_FILE)


conn = sqlite3.connect(DB_FILE)

cursor = conn.cursor()


# ============================================================
# CREATE TABLES
# ============================================================

cursor.execute("""
CREATE TABLE candidate_index (
    pair TEXT,
    entity_id TEXT,
    source TEXT,
    field TEXT
)
""")

conn.commit()


# ============================================================
# PROCESS SOURCE
# ============================================================

def process_file(file_path, source):

    print()
    print("=" * 70)
    print("Processing", source)
    print("=" * 70)

    total = 0
    insert_buffer = []

    for chunk in pd.read_csv(
        file_path,
        sep="\t",
        dtype=str,
        chunksize=CHUNK_SIZE
    ):

        for _, row in chunk.iterrows():

            entity_id = str(
                row["entity_id"]
            )

            # -----------------------------
            # NAME
            # -----------------------------

            name = normalize_text(
                row.get("business_name", "")
            )

            name_pairs = set(
                get_pairs(name)
            )

            for pair in name_pairs:

                insert_buffer.append(
                    (
                        pair,
                        entity_id,
                        source,
                        "name"
                    )
                )

            # -----------------------------
            # ADDRESS
            # -----------------------------

            address = normalize_text(
                row.get("business_address", "")
            )

            address_pairs = set(
                get_pairs(address)
            )

            for pair in address_pairs:

                insert_buffer.append(
                    (
                        pair,
                        entity_id,
                        source,
                        "address"
                    )
                )

            total += 1

            # -----------------------------
            # BULK INSERT
            # -----------------------------

            if len(insert_buffer) >= 100000:

                cursor.executemany(
                    """
                    INSERT INTO candidate_index
                    (pair, entity_id, source, field)
                    VALUES (?, ?, ?, ?)
                    """,
                    insert_buffer
                )

                conn.commit()

                insert_buffer.clear()

        print(
            source,
            "processed:",
            total
        )

    # Remaining rows

    if insert_buffer:

        cursor.executemany(
            """
            INSERT INTO candidate_index
            (pair, entity_id, source, field)
            VALUES (?, ?, ?, ?)
            """,
            insert_buffer
        )

        conn.commit()

        insert_buffer.clear()

    print(
        source,
        "complete:",
        total
    )


# ============================================================
# PROCESS S2
# ============================================================

process_file(
    S2_FILE,
    "S2"
)


# ============================================================
# PROCESS S3
# ============================================================

process_file(
    S3_FILE,
    "S3"
)


# ============================================================
# CREATE INDEX
# ============================================================

print()
print("=" * 70)
print("CREATING SQLITE INDEX")
print("=" * 70)

cursor.execute("""
CREATE INDEX idx_pair
ON candidate_index(pair)
""")

cursor.execute("""
CREATE INDEX idx_pair_source
ON candidate_index(pair, source)
""")

conn.commit()


# ============================================================
# STATISTICS
# ============================================================

cursor.execute(
    "SELECT COUNT(*) FROM candidate_index"
)

total_rows = cursor.fetchone()[0]

cursor.execute(
    "SELECT COUNT(DISTINCT pair) FROM candidate_index"
)

unique_pairs = cursor.fetchone()[0]

cursor.execute(
    "SELECT COUNT(DISTINCT entity_id) FROM candidate_index"
)

unique_entities = cursor.fetchone()[0]


print()
print("=" * 70)
print("INDEX COMPLETE")
print("=" * 70)

print(
    "Index rows:",
    total_rows
)

print(
    "Unique token pairs:",
    unique_pairs
)

print(
    "Unique entities:",
    unique_entities
)

print()
print("Database:")
print(DB_FILE)

conn.close()

print()
print("=" * 70)
print("STEP 6A COMPLETE")
print("=" * 70)
