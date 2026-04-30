# ml-algorithms
R analysis of the 2016 US Republican primary. Merges county-level demographic and economic census data to predict Trump's vote share using classification models (Logistic, RF, SVM, KNN, LDA/QDA, Naive Bayes, Decision Tree) and clustering methods (Hierarchical, GMM, DBSCAN).


# 2016 US Republican Primary: County-Level Analysis

R analysis of the 2016 US Republican primary. Merges county-level demographic and economic census data to predict Trump's vote share using classification models (Logistic, RF, SVM, KNN, LDA/QDA, Naive Bayes, Decision Tree) and clustering methods (Hierarchical, GMM, DBSCAN).

---

## Table of Contents

- [Project Overview](#project-overview)
- [Data Sources](#data-sources)
- [Repository Structure](#repository-structure)
- [Requirements](#requirements)
- [Methodology](#methodology)
  - [Data Engineering](#data-engineering)
  - [Part 1: Supervised Classification](#part-1-supervised-classification)
  - [Part 2: Unsupervised Clustering](#part-2-unsupervised-clustering)
- [Results Summary](#results-summary)
- [Notes & Limitations](#notes--limitations)
- [Author](#author)

---

## Project Overview

This project investigates what demographic and economic characteristics of US counties predict whether Donald Trump exceeded 50% of the Republican primary vote in 2016. It is split into two parts:

**Part 1** frames the problem as binary classification. The target variable `trump_majority` indicates whether Trump received more than 50% of the vote in a given county (`gt50`) or not (`lte50`). Nine models are trained, evaluated, and compared.

**Part 2** uses demographic variables to cluster counties into groups, then describes those groups using economic indicators. Three clustering approaches are applied and compared.

---

## Data Sources

The analysis merges two datasets:

| Dataset | Dimensions | Description |
|---|---|---|
| `df1` | 3,195 × 54 | County-level demographic and economic census data |
| `df2` | 24,611 × 8 | 2016 Republican primary results by county |

The two datasets are joined on `fips` code and `state_abbreviation` (inner join), resulting in **17,479 rows × 59 columns** before filtering.

---

## Repository Structure

```
.
├── ml-code.R     # Main analysis script
├── datamerged.RData      # Merged dataset (generated at runtime)
├── trumpdata.RData       # Trump-filtered subset (generated at runtime)
└── README.md
```

---

## Requirements

The following R packages are required:

```r
require(readxl)
require(caret)
require(randomForest)
require(glmnet)
require(class)
require(naivebayes)
require(rpart)
require(rpart.plot)
require(gridExtra)
require(pROC)
require(vip)
require(cluster)
require(clustvarsel)
require(mclust)
require(ggcorrplot)
require(klaR)
require(vscc)
require(pgmm)
require(NbClust)
require(dbscan)
require(ClustOfVar)
require(corrgram)
```

Install any missing packages with `install.packages("package_name")`.

---

## Methodology

### Data Engineering

Raw data required several preprocessing steps before analysis:

- Removed the aggregate "United States" row from `df1`
- Built a `state` column from a manual `state_abbreviation` lookup table
- Extracted `county` names by stripping the geographic type suffix
- **Excluded Alaska** due to incompatible geographic subdivisions between the two datasets (df1 uses Boroughs/Census Areas; df2 uses State House Districts — no 1:1 mapping exists)
- **Excluded 8 additional states** (CT, KS, MA, ME, MN, ND, NH, RI, VT) that reported primary results at the congressional or legislative district level rather than the county level, making a county-level merge impossible
- Final merged dataset covers **40 states**

### Part 1: Supervised Classification

**Target variable:** `trump_majority` — binary factor indicating whether Trump's `fraction_votes > 0.50` in a given county.

> Note: In a multi-candidate primary, winning a county with a plurality (not a majority) is completely normal. Trump won many counties without cracking 50% because the vote was split across Cruz, Rubio, Kasich, Carson, and others. The class imbalance (lte50: 1,714 | gt50: 997) reflects this reality.

**Train/Test split:** 70/30 stratified split using `createDataPartition`.

**Cross-validation:** 10-fold CV with `twoClassSummary` (ROC as the optimization metric).

The following models were trained and compared:

| Model | Key Hyperparameters Tuned |
|---|---|
| Logistic Regression (Elastic Net) | `alpha`, `lambda` via `glmnet` |
| Linear Regression (baseline) | None |
| K-Nearest Neighbours | Number of neighbours `k` |
| Naive Bayes | `usekernel`, `laplace`, `adjust` |
| Decision Tree | Complexity parameter `cp` via `rpart` |
| Random Forest | `mtry` |
| LDA | None |
| QDA | None |
| SVM (Radial Kernel) | Cost `C`, `sigma` |

All models were evaluated on accuracy, and ROC curves were plotted together for visual comparison. Variable importance was extracted for each compatible model.

### Part 2: Unsupervised Clustering

**Goal:** Cluster US counties using demographic variables, then interpret clusters using economic variables.

**Variable selection:** `hclustvar` from the `ClustOfVar` package was used to group correlated demographic variables into 7 clusters. One representative variable per cluster was selected based on the highest squared loading, yielding 7 features for clustering:

```
POP010210, EDU685213, AGE295214, SEX255214, RHI125214, RHI625214, POP815213
```

All features were standardised with `scale()` before clustering.

Three clustering algorithms were applied:

**Hierarchical Clustering (Ward's Linkage)**
- Distance matrix computed with Euclidean distance
- Optimal number of clusters assessed with `NbClust` (30 indices), height plots, and silhouette plots
- Agglomerative coefficient: 0.9914 (excellent structure)

**Model-Based Clustering — Gaussian Mixture Model (GMM)**
- Model and cluster count selected via ICL criterion using `mclustICL`
- Best model: VVV, k=5 (ICL = -33,337)
- Final solution evaluated with silhouette plots

**Density-Based Clustering — DBSCAN**
- `minPts = 8`, optimal `eps` explored via k-NN distance plot
- Result: one dominant cluster containing ~95% of data points with minimal noise
- DBSCAN was deemed unsuitable for this dataset due to the absence of meaningful density separation; results are not reported

---

## Results Summary

### Classification Accuracy

| Model | Accuracy |
|---|---|
| Random Forest | 0.7478 |
| Logistic Regression | 0.7159 |
| LDA | 0.7122 |
| Linear Regression | 0.7085 |
| SVM | 0.7306 |
| Decision Tree | 0.6900 |
| KNN | 0.6765 |
| QDA | 0.6458 |
| Naive Bayes | 0.6531 |

**Random Forest** achieved the highest accuracy (74.78%) and ROC-AUC.

### Clustering

Hierarchical clustering with Ward's linkage and GMM both converged on **2 meaningful clusters**, interpretable through economic indicators such as income, housing, poverty rate, and business ownership.

---

## Notes & Limitations

- The 2016 Republican primary uses **plurality voting** across multiple candidates, so `trump_majority = lte50` does not mean Trump lost that county, but it means he won without an outright majority, which was the norm.
- Nine states were excluded due to geographic incompatibilities or non-county-level reporting. The analysis covers 40 states.
- DBSCAN was explored but ultimately not suitable for this dataset's structure.
- All random processes use `set.seed(222)` for reproducibility.

---

## Author

**Nefeli Boumpari**
