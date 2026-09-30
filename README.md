# Business Entity Resolution - ML Challenge 2026

## Overview

This solution performs large-scale business entity resolution between Source 1 and Sources 2 and 3.

The pipeline uses:

1. Text normalization
2. Candidate generation using exact normalized business-name blocking
3. Feature engineering using name and address similarity features
4. LightGBM classification
5. Threshold-based matching
6. Final submission generation

## Pipeline

The main source files are located under src/.

- step5c_generate_features.py - training feature generation
- step5d_train_model.py - LightGBM model training
- step6a_build_test_index.py - candidate indexing
- step6b_exact.py - test candidate generation
- step6c_score.py - ML candidate scoring
- step6d_matching.py - final matching result generation

## Dependencies

Install the required Python packages using:

pip install -r requirements.txt

## Final Output

The final submission files are generated under output/:

- candidate_pairs.tsv
- matching_results.tsv

candidate_pairs.tsv contains the candidate entity IDs considered for each Source 1 entity.

matching_results.tsv contains the final predicted matches for each Source 1 entity.

## Validation

The final output can be validated using:

python utils/validate_submission.py --matching output/matching_results.tsv --candidate output/candidate_pairs.tsv --test-dir dataset/test --check-ids

## Methodology

The candidate generation stage uses normalized business names to efficiently reduce the search space.

The matching stage uses a LightGBM classification model with name similarity, address similarity, exact-match, country, and other engineered features.

A probability threshold is applied to determine final matches.

## Reproducibility

The source scripts used for training, candidate generation, scoring, and final matching are included in src/.

The generated final files are included in the output/ directory.
