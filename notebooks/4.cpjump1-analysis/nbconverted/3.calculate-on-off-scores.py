#!/usr/bin/env python

# ## Generating On and Off Scores
#
# With the cluster labels and consensus on/off signatures, we can begin calculating phenotypic activity scores.
#
# This is achieved by measuring the Earth Mover’s Distance (EMD) between the morphological features of each cluster and the on/off signatures. Specifically, the on-score is calculated using only the features relevant to the on-signature, while the off-score uses features from the off-signature.
#
# These scores are then combined into a weighted sum (as implemented in the `heterogeneity.py` module) and sorted from lowest to highest. Lower scores indicate that the sample is more similar to the reference, making them top candidates for further analysis.

# In[ ]:


import pathlib
import sys

import polars as pl
import tqdm

sys.path.append("../../")
from utils.data_utils import split_meta_and_features
from utils.io_utils import load_configs, load_profiles
from utils.metrics import measure_phenotypic_activity

# In[ ]:


# setting paths
data_dir = pathlib.Path("../0.download-data/data/sc-profiles")
results_dir = pathlib.Path("./results/").resolve(strict=True)
signatures_result_dir = (results_dir / "signatures").resolve(strict=True)
cluster_results_dir = (results_dir / "clusters").resolve(strict=True)

# set signatures paths
u2os_cpjump1_consensus_signatures_path = (
    signatures_result_dir / "ks_test_u2os_cpjump1_consensus_signatures_per_poscon.json"
).resolve(strict=True)
a549_cpjump1_consensus_signatures_path = (
    signatures_result_dir / "ks_test_a549_cpjump1_consensus_signatures_per_poscon.json"
).resolve(strict=True)

# set cluster labels paths
u2os_cluster_labels_path = (
    cluster_results_dir / "cpjump1_u2os_144hrs_clusters.parquet"
).resolve(strict=True)
a549_cluster_labels_path = (
    cluster_results_dir / "cpjump1_a549_144hrs_clusters.parquet"
).resolve(strict=True)

# setting cpjump1 experimental data
cpjump1_experimental_data_path = (
    data_dir / "cpjump1/CPJUMP1-experimental-metadata.csv"
).resolve(strict=True)

# setting CPJUMP1 profiles path
cpjump1_profiles_path = (
    data_dir / "cpjump1/cpjump1_crispr_concat_profiles.parquet"
).resolve(strict=True)

# output filepath
output_dir = (results_dir / "scores").resolve()
output_dir.mkdir(exist_ok=True)


# Load experimental data and find the plates that make up the two cell types that have been incubated for 144hrs

# In[ ]:


# loading CPJUMP1 experimental data
cpjump1_experimental_data = pl.read_csv(cpjump1_experimental_data_path)

# select interested time and batch (144 hrs, batch 2020_11_04_CPJUMP1)
cpjump1_experimental_data = cpjump1_experimental_data.filter(
    (pl.col("Time") == 144) & (pl.col("Batch") == "2020_11_04_CPJUMP1")
)

# now generate groups based on batch, cell_type and time
cpjump1_u2os_exp_metadata = cpjump1_experimental_data.filter(
    pl.col("Cell_type") == "U2OS"
)

cpjump1_a549_exp_metadata = cpjump1_experimental_data.filter(
    (pl.col("Cell_type") == "A549") & (pl.col("Antibiotics") == "absent")
)

# get the plates for each cell type
u2os_plates = cpjump1_u2os_exp_metadata["Assay_Plate_Barcode"].unique().to_list()
a549_plates = cpjump1_a549_exp_metadata["Assay_Plate_Barcode"].unique().to_list()

# print the plates
print(f"U2OS plates: {u2os_plates}")
print(f"A549 plates: {a549_plates}")


# Load cpjump1 dataset and filter it to those plates only

# In[ ]:


# load in the cluster labels
u2os_cluster_labels_df = pl.read_parquet(u2os_cluster_labels_path)
a549_cluster_labels_df = pl.read_parquet(a549_cluster_labels_path)

# load cpjump datasets and filtered by the selected paltes
cpjump1_df = load_profiles(cpjump1_profiles_path).filter(
    pl.col("Metadata_Plate").is_in(u2os_plates + a549_plates)
)

# update Metadata_gene column: if Metadata_gene is null and Metadata_control_type is "negcon", replace with "DMSO"
cpjump1_df = cpjump1_df.with_columns(
    pl.when(
        pl.col("Metadata_gene").is_null()
        & (pl.col("Metadata_control_type") == "negcon")
    )
    .then(pl.lit("DMSO"))
    .otherwise(pl.col("Metadata_gene"))
    .alias("Metadata_gene")
)

# split the feature space
cpjump1_meta, cpjump1_features = split_meta_and_features(cpjump1_df)

# split the datasets based on cell type and merge cluster labels based on Metadata_cell_id
u2os_cpjump1_df = cpjump1_df.filter(pl.col("Metadata_Plate").is_in(u2os_plates)).join(
    u2os_cluster_labels_df, on="Metadata_cell_id"
)
a549_cpjump1_df = cpjump1_df.filter(pl.col("Metadata_Plate").is_in(a549_plates)).join(
    a549_cluster_labels_df, on="Metadata_cell_id"
)


# Load the on and off signatures.
#
# This contains multiple consensus on and off signatures since there were multiple positive controls in the dataset. Therefore, this required generating on and off signatures for each positive control. When measuring phenotypic activity, we iterate through each positive control and use its corresponding signatures.

# In[ ]:


# load consensus signatures
u2os_signatures = load_configs(u2os_cpjump1_consensus_signatures_path)
a549_signatures = load_configs(a549_cpjump1_consensus_signatures_path)


# ## Measuring phenotypic activity
#

# In[ ]:


# measuring phenotypic activity for u2os cells
for poscon_gen, signatures in tqdm.tqdm(u2os_signatures.items()):
    # start message
    print(
        f"Starting phenotypic activity analysis for U2OS with positive control: {poscon_gen}"
    )

    # measuring phenotypic activity with given poscon gene
    treatment_phenotypic_dist_scores = measure_phenotypic_activity(
        profiles=u2os_cpjump1_df,
        on_signature=signatures["on"],
        off_signature=signatures["off"],
        ref_treatment="DMSO",
        cluster_col="Metadata_cluster_id",
        treatment_col="Metadata_gene",
    )

    # message
    print(f"Writing parquet file for U2OS with positive control: {poscon_gen}")

    # save as parquet file
    treatment_phenotypic_dist_scores.write_parquet(
        output_dir / f"u2os_{poscon_gen}_phenotypic_dist_scores.parquet"
    )

    print(
        f"Completed phenotypic activity analysis for U2OS with positive control: {poscon_gen}"
    )


# In[ ]:


# measuring phenotypic activity for a549 cells
for poscon_gen, signatures in tqdm.tqdm(a549_signatures.items()):
    # start message
    print(
        f"Starting phenotypic activity analysis for A549 with positive control: {poscon_gen}"
    )

    # measuring phenotypic activity with given poscon gene
    treatment_phenotypic_dist_scores = measure_phenotypic_activity(
        profiles=a549_cpjump1_df,
        on_signature=signatures["on"],
        off_signature=signatures["off"],
        ref_treatment="DMSO",
        cluster_col="Metadata_cluster_id",
        treatment_col="Metadata_gene",
    )

    # message
    print(f"Writing parquet file for A549 with positive control: {poscon_gen}")

    # save as parquet file
    treatment_phenotypic_dist_scores.write_parquet(
        output_dir / f"a549_{poscon_gen}_phenotypic_dist_scores.parquet"
    )
