#!/usr/bin/env python

# # CFReT-Screen analysis
#
# In this notebook, we will be applying `buscar` to the CFReT initial screen.
#
# The resource for this dataset can be found [here](https://github.com/WayScience/targeted_fibrosis_drug_screen/tree/main/3.preprocessing_features)
#

# In[ ]:


import json
import pathlib
import sys

import matplotlib.pyplot as plt
import numpy as np
import polars as pl
import seaborn as sns

sys.path.append("../../")
# from utils.metrics import measure_phenotypic_activity
from utils.data_utils import split_meta_and_features
from utils.heterogeneity import optimized_clustering
from utils.identify_hits import identify_compound_hit
from utils.io_utils import load_profiles
from utils.metrics import measure_phenotypic_activity
from utils.signatures import get_signatures

# In[ ]:


# setting parameters
treatment_col = "Metadata_treatment"
treatment_heart_col = "Metadata_treatment_and_heart"

# parameters used for clustering optimization
cfret_cluster_param_grid = {
    # Clustering resolution: how granular the clusters should be
    "cluster_resolution": {"type": "float", "low": 0.1, "high": 2.5},
    # Number of neighbors for graph construction
    "n_neighbors": {"type": "int", "low": 10, "high": 100},
    # Clustering algorithm
    "cluster_method": {"type": "categorical", "choices": ["leiden", "louvain"]},
    # Distance metric for neighbor computation
    "neighbor_distance_metric": {
        "type": "categorical",
        "choices": ["euclidean", "cosine", "manhattan"],
    },
    # Dimensionality reduction approach
    "dim_reduction": {"type": "categorical", "choices": ["PCA", "raw"]},
}


# setting paths

# In[ ]:


# load in raw data from
cfret_data_dir = pathlib.Path(
    "../0.download-data/data/sc-profiles/cfret-screen"
).resolve(strict=True)
cfret_profiles_path = (cfret_data_dir / "cfret_screen_concat_profiles.parquet").resolve(
    strict=True
)

# make results dir
results_dir = pathlib.Path("./results/cfret-screen").resolve()
results_dir.mkdir(parents=True, exist_ok=True)


# In[ ]:


# loading profiles
cfret_df = load_profiles(cfret_profiles_path)
cfret_screen_meta, cfret_screen_feats = split_meta_and_features(cfret_df)

# updating the treatment name to reflect the heart source for DMSO in healthy cells
# this is our reference for healthy cells when measuring phenotypic activity
cfret_df = cfret_df.with_columns(
    pl.when(
        (pl.col("Metadata_treatment") == "DMSO")
        & (pl.col("Metadata_cell_type") == "healthy")
    )
    .then(pl.lit("DMSO_heart_11"))
    .otherwise(pl.col("Metadata_treatment"))
    .alias("Metadata_treatment")
)

# Display data
cfret_df.head()


# ## Preprocessing

# Filtering Treatments with Low Cell Counts:
#
# Treatments with low cell counts were removed from the analysis. This reduction in cell numbers is typically caused by cellular toxicity, which leads to cell death and consequently results in insufficient cell representation for downstream analysis.
#
# Low cell count treatments also pose challenges when assessing heterogeneity, as there are not enough data points to form meaningful clusters. To address this, highly toxic compounds with very few surviving cells were excluded from the BUSCAR analysis.
#
# A threshold of 10% was applied based on Scanpy documentation, which recommends having at least 15–100 data points to compute a reliable neighborhood graph. To validate this threshold, we generated a histogram of cell counts and marked the 10th percentile with a red line. Treatments falling below this threshold were removed and excluded from the BUSCAR pipeline.

# In[ ]:


# count number of cells per Metadata_treatment and ensure 'count' is Int64
counts = cfret_df["Metadata_treatment"].value_counts()
counts = counts.with_columns(pl.col("count").cast(pl.Int64))
counts = counts.sort("count", descending=True)
counts = counts.to_pandas()


# In[ ]:


# using numpy to calculate 10th percentile
tenth_percentile = np.round(np.percentile(counts["count"], 10), 3)
print(f"10th percentile of cell counts: {tenth_percentile} cells")


# Plotting cell count distribution

# In[ ]:


# setting seaborn style and figure size
sns.set(style="whitegrid")
plt.figure(figsize=(12, 6), dpi=200)

# plot histogram with seaborn
ax = sns.histplot(data=counts, x="count", bins=100, color="skyblue", kde=True)

# add 10th percentile vertical line and annotation (tenth_percentile already defined)
ax.axvline(
    x=tenth_percentile,
    color="red",
    linestyle="--",
    linewidth=2,
    label=f"10th percentile ({int(tenth_percentile)} cells)",
)
ymin, ymax = ax.get_ylim()
ax.text(
    tenth_percentile,
    ymax * 0.9,
    f"10th pct = {tenth_percentile:.0f}",
    color="red",
    rotation=90,
    va="top",
    ha="right",
    backgroundcolor="white",
)

# labeling the plot
ax.set_xlabel("Number of Cells")
ax.set_ylabel("Metadata_treatment")
ax.set_title("Cell Count per treeatment in CFRET screen")

# adding legend
ax.legend()

# adjust layout
plt.tight_layout()

# save the plot
plt.savefig(results_dir / "cell_count_per_treatment_cfret_screen.png", dpi=500)

# display plot
plt.show()


# Removing cells under those specific treatments

# In[ ]:


# remove treatments with cell counts below the 10th percentile
kept_treatments = counts[counts["count"] >= tenth_percentile][
    "Metadata_treatment"
].tolist()
cfret_df = cfret_df.filter(pl.col("Metadata_treatment").is_in(kept_treatments))

# print the treatments that were removed
removed_treatments = counts[counts["count"] < tenth_percentile][
    "Metadata_treatment"
].tolist()
print(
    "Removed treatments due to low cell counts (below 10th percentile):",
    removed_treatments,
)

cfret_df.head()


# ## Buscar pipeline

# Get on and off signatures

# In[ ]:


# once the data is loaded, separate the controls
# here we want the healthy DMSO cells to be the target since the screen consists
# of failing cells treated with compounds
ref_df = cfret_df.filter(
    pl.col("Metadata_treatment") == "DMSO", pl.col("Metadata_cell_type") == "failing"
)
target_df = cfret_df.filter(pl.col("Metadata_treatment") == "DMSO_heart_11")

# creating signatures
on_sigs, off_sigs, _ = get_signatures(
    ref_profiles=ref_df,
    exp_profiles=target_df,
    morph_feats=cfret_screen_feats,
    test_method="mann_whitney_u",
)

print("length of on and off signatures:", len(on_sigs), len(off_sigs))


# Assess heterogeneity

# In[ ]:


# setting best params outputs
cfret_screen_treatment_best_params_outpath = (
    results_dir / "cfret_screen_treatment_clustering_params.json"
).resolve()
cfret_screen_treatment_cluster_df_outpath = (
    results_dir / "cfret_screen_treatment_clustered.parquet"
).resolve()

# here we are clustering each treatment-heart combination
# this will allow us to see how each heart responds to each treatment
cfret_screen_treatment_clustered_df, cfret_screen_treatment_clustered_best_params = (
    optimized_clustering(
        profiles=cfret_df,
        meta_features=cfret_screen_meta,
        morph_features=cfret_screen_feats,
        treatment_col="Metadata_treatment",
        param_grid=cfret_cluster_param_grid,
        n_trials=200,
        n_jobs=1,
    )
)

# save best params as json and dataframe as parquet
cfret_screen_treatment_clustered_df.write_parquet(
    cfret_screen_treatment_cluster_df_outpath
)
with open(cfret_screen_treatment_best_params_outpath, "w") as f:
    json.dump(
        cfret_screen_treatment_clustered_best_params,
        f,
        indent=4,
    )


# In[ ]:


treatment_phenotypic_dist_scores = measure_phenotypic_activity(
    profiles=cfret_screen_treatment_clustered_df,
    on_signature=on_sigs,
    off_signature=off_sigs,
    ref_treatment="DMSO_heart_11",
    cluster_col="Metadata_cluster_id",
)

# save those as csv files
treatment_phenotypic_dist_scores.write_csv(
    results_dir / "cfret_screen_treatment_phenotypic_dist_scores.csv"
)


# In[ ]:


treatment_rankings = identify_compound_hit(
    distance_df=treatment_phenotypic_dist_scores, method="weighted_sum"
)

# save as csv files
treatment_rankings.write_csv(results_dir / "cfret_screen_treatment_rankings.csv")
