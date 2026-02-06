#!/usr/bin/env python

# # Assessing Heterogeneity
#
# In this notebook, we assess cellular heterogeneity using community-based clustering. We utilize the `optimized_clustering` function from the `heterogeneity` module, which leverages Optuna to perform hyperparameter optimization for clustering. The optimization process uses the silhouette score to identify the most suitable clustering parameters.
#
# Clustering is performed on a per-treatment basis, meaning that single cells are clustered separately for each treatment group. This approach allows us to capture the diversity within cell populations exposed to different treatments, rather than applying a global clustering across all cells.

# In[ ]:


import json
import logging
import pathlib
import sys

import polars as pl

sys.path.append("../../")
from utils.data_utils import split_meta_and_features
from utils.heterogeneity import optimized_clustering
from utils.io_utils import load_profiles
from utils.preprocess import apply_pca

# Setting parametes for the notebook

# In[ ]:


# parameters used for clustering optimization
cfret_screen_cluster_param_grid = {
    # Clustering resolution: how granular the clusters should be
    "cluster_resolution": {"type": "float", "low": 0.1, "high": 2.2},
    # Number of neighbors for graph construction
    "n_neighbors": {"type": "int", "low": 5, "high": 100},
    # Clustering algorithm
    "cluster_method": {"type": "categorical", "choices": ["leiden"]},
    # Distance metric for neighbor computation
    "neighbor_distance_metric": {
        "type": "categorical",
        "choices": ["euclidean", "cosine", "manhattan"],
    },
}


# Setting input and output paths

# In[ ]:


# setting data directory
data_dir = pathlib.Path("../0.download-data/data/sc-profiles/").resolve(strict=True)
"../0.download-data/data/sc-profiles/cpjump1/cp"

# setting CPJUMP1 profiles path
cpjump1_profiles_path = (
    data_dir / "cpjump1/cpjump1_compound_concat_profiles.parquet"
).resolve(strict=True)

# setting cpjump1 experimental data
cpjump1_experimental_data_path = (
    data_dir / "cpjump1/CPJUMP1-experimental-metadata.csv"
).resolve(strict=True)


# setting output paths
results_dir = pathlib.Path("./results").resolve()
results_dir.mkdir(parents=True, exist_ok=True)

# setting cluster output directory
cluster_results_dir = (results_dir / "clusters").resolve()
cluster_results_dir.mkdir(parents=True, exist_ok=True)

# setting pca output results
pca_results_dir = (results_dir / "pca").resolve()
pca_results_dir.mkdir(parents=True, exist_ok=True)


# Load the experimental metadata and select only the plates of interest. Here, we focus on plates incubated for 144 hours and filter by cell type. We then extract the plate barcodes corresponding to these conditions. This allows us to analyze only the relevant subset of the dataset for downstream analysis.

# In[ ]:


# load in the experimental data
cpjump1_experimental_data = pl.read_csv(cpjump1_experimental_data_path)

# Split the dataset by cell type and treatment duration
# Filter U2OS cells (all records)
cpjump1_u2os_exp_metadata = cpjump1_experimental_data.filter(
    pl.col("Cell_type") == "U2OS"
)

# Filter A549 cells with density of 100 for consistency
cpjump1_a549_exp_metadata = cpjump1_experimental_data.filter(
    (pl.col("Cell_type") == "A549") & (pl.col("Density") == 100)
)

# get the plates for each cell type
u20s_plates = cpjump1_u2os_exp_metadata["Assay_Plate_Barcode"].unique().to_list()
a549_plates = cpjump1_a549_exp_metadata["Assay_Plate_Barcode"].unique().to_list()

# print the plates
print("U2OS plates:", u20s_plates)
print("A549 plates:", a549_plates)


# Next we load in the compound cpjump1 single-cell profiles

# In[ ]:


# load profiles
cpjump1_df = load_profiles(cpjump1_profiles_path)

# separete metadata and feature columns
cpjump1_meta, cpjump1_feats = split_meta_and_features(cpjump1_df)

# display
print("shape: ", cpjump1_df.shape)
cpjump1_df.head()


# Convert the single-cell spce into PCA componenets that explains 95% of the variance

# In[ ]:


cpjump1_u2os_df = apply_pca(
    cpjump1_df.filter(pl.col("Metadata_Plate").is_in(u20s_plates)),
    meta_features=cpjump1_meta,
    morph_features=cpjump1_feats,
    var_explained=0.95,
)
cpjump1_a549_df = apply_pca(
    cpjump1_df.filter(pl.col("Metadata_Plate").is_in(a549_plates)),
    meta_features=cpjump1_meta,
    morph_features=cpjump1_feats,
    var_explained=0.95,
)

# now get pca_feature names
cpjump1_a549_pca_features = cpjump1_a549_df.drop(cpjump1_meta).columns
cpjump1_u2os_pca_features = cpjump1_u2os_df.drop(cpjump1_meta).columns

# save pca profiles
cpjump1_u2os_df.write_parquet(pca_results_dir / "cpjump1_u2os_pca_profiles.parquet")
cpjump1_a549_df.write_parquet(pca_results_dir / "cpjump1_a549_pca_profiles.parquet")

# print shape of the pca dataframes
print("U2OS PCA shape: ", cpjump1_u2os_df.shape)
print("A549 PCA shape: ", cpjump1_a549_df.shape)


# Execute optimized clustering for both U2OS cells and A549 cells

# In[ ]:


# setting log (delete later)
logging.basicConfig(
    level=logging.INFO,
    filename="_heterogeneity_clustering.log",
    format="%(asctime)s - %(levelname)s - %(message)s",
)

try:
    # Your clustering code here
    logging.info("Starting U2OS clustering...")

    # U2OS clustering optimization
    u2os_clusters, u2os_params_summary = optimized_clustering(
        cpjump1_u2os_df,
        meta_features=cpjump1_meta,
        morph_features=cpjump1_u2os_pca_features,
        treatment_col="Metadata_pert_iname",
        param_grid=cfret_screen_cluster_param_grid,
        n_trials=60,
        n_jobs=45,
        seed=0,
        study_name="cpjump1_u2os_clustering_optimization",
    )

    # save cluster labels and parameters summary
    u2os_clusters.write_parquet(results_dir / "cpjump1_u2os_clusters.parquet")
    with open(
        cluster_results_dir / "cpjump1_u2os_clustering_optimization_study_summary.json",
        "w",
    ) as f:
        json.dump(u2os_params_summary, f, indent=4)
    logging.info("U2OS clustering complete!")

    # A549 clustering optimization
    logging.info("Starting A549 clustering...")
    a549_clusters, a549_params_summary = optimized_clustering(
        cpjump1_a549_df,
        meta_features=cpjump1_meta,
        morph_features=cpjump1_a549_pca_features,
        treatment_col="Metadata_pert_iname",
        param_grid=cfret_screen_cluster_param_grid,
        n_trials=60,
        n_jobs=45,
        seed=0,
        study_name="cpjump1_a549_clustering_optimization",
    )

    # save cluster labels and parameters summary
    a549_clusters.write_parquet(results_dir / "cpjump1_a549_clusters.parquet")
    with open(
        cluster_results_dir / "cpjump1_a549_clustering_optimization_study_summary.json",
        "w",
    ) as f:
        json.dump(a549_params_summary, f, indent=4)
    logging.info("A549 clustering complete!")

except Exception as e:
    logging.error(f"Error during clustering: {e}", exc_info=True)
    raise
