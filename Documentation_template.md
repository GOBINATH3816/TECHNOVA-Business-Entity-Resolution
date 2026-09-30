# ML Challenge 2026: Business Entity Resolution Solution

**Team Name:** TECHNOVA  
**Team Members:** ANITHA AR, GOBINATH M, NITHYA V  
**Submission Date:** 27 September 2026

---

## 1. Executive Summary

This solution performs large-scale business entity resolution between Source 1 and Sources 2 and 3 using only the provided challenge data. The approach combines text normalization, exact normalized-name candidate generation, engineered name/address similarity features, and a LightGBM classifier to identify likely entity matches.

The final pipeline generates a candidate set first and then applies the trained matching model to score the candidates and produce the required submission files.

---

## 2. Methodology

### 2.1 Problem Analysis

The task contains business records from multiple sources with variations in business names and addresses. The data can contain differences in punctuation, capitalization, spacing, abbreviations, word ordering, missing fields, and other textual variations.

Because Source 1 contains a large number of records and Sources 2 and 3 contain millions of records, directly comparing every Source 1 record with every Source 2 and Source 3 record would be computationally expensive.

Therefore, the solution separates the task into two stages:

1. Candidate generation to reduce the search space.
2. Machine-learning-based matching to score the generated candidates.

The challenge evaluation uses a precision-heavy F_0.5 metric, so the final matching stage uses a probability threshold to control false positive matches.

### 2.2 Solution Strategy

**Approach Type:** Blocking + Classifier

**Core Approach:** The pipeline first normalizes business names and generates a restricted candidate set using exact normalized business-name blocking. Candidate pairs are then represented using multiple name and address similarity features and classified using LightGBM.

The overall workflow is:

```text
Raw Data
   ↓
Text Normalization
   ↓
Candidate Generation
   ↓
Feature Engineering
   ↓
LightGBM Classification
   ↓
Probability Threshold
   ↓
Final Matching Results