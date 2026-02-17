#!/usr/bin/env python

# In[ ]:


import json
import logging
import pathlib
import sys
from pprint import pprint

import polars as pl
from tqdm import tqdm

sys.path.append("../../")
from buscar.metrics import measure_phenotypic_activity
from buscar.signatures import get_signatures
from utils.data_utils import shuffle_feature_profiles, split_meta_and_features
from utils.io_utils import load_configs, load_profiles

# Notebook parameters

# In[ ]:


# set to True for debugging purposes, to run the notebook faster with a subset of the data
subet_data = False
subet_fraction = 0.1


# Setting input and output paths

# In[ ]:


# setting data directory
data_dir = pathlib.Path("../0.download-data/data/sc-profiles/").resolve(strict=True)
results_module_dir = pathlib.Path("./results").resolve(strict=True)

# setting cpjump1_dataset
cpjump1_profiles_path = (
    data_dir / "cpjump1/cpjump1_compound_concat_profiles.parquet"
).resolve(strict=True)

# get experimental dataset
cpjump1_experimental_metadata_path = (
    data_dir / "cpjump1/CPJUMP1-experimental-metadata.csv"
).resolve(strict=True)

# get shared feature space
shared_feature_space = (
    data_dir / "cpjump1/feature_selected_sc_qc_features.json"
).resolve(strict=True)

# moa config
cpjump1_compounds_moa = (data_dir / "cpjump1/cpjump1_compound_moa.tsv").resolve(
    strict=True
)

# create MoA analysis output directory
moa_analysis_output_dir = (results_module_dir / "moa_analysis").resolve()
moa_analysis_output_dir.mkdir(parents=True, exist_ok=True)


# In[ ]:


# loading shared features
shared_features = load_configs(shared_feature_space)["shared-features"]

# loading experimental and moa metadata
cpjump1_moa_df = pl.read_csv(cpjump1_compounds_moa, separator="\t")
cpjump1_experimental_data = pl.read_csv(cpjump1_experimental_metadata_path)

# load profiles
cpjump1_df = load_profiles(cpjump1_profiles_path)
cpjump1_meta, cpjump1_feats = split_meta_and_features(cpjump1_df)

# replace treatments where the MoA is 'null' to 'unknown'
cpjump1_moa_df = cpjump1_moa_df.with_columns(
    pl.when(pl.col("Metadata_moa").is_null())
    .then(pl.lit("unknown"))
    .otherwise(pl.col("Metadata_moa"))
    .alias("Metadata_moa")
)

# displaying dataframe information
print(f"Dataframe shape: {cpjump1_df.shape}")
print(
    "Number of unique treatments",
    cpjump1_df["Metadata_pert_iname"].n_unique(),
)

cpjump1_df.head()


# Identify plates that contains U2OS and A549 cells

# In[ ]:


# Split the dataset by cell type and treatment duration
# Filter U2OS cells (all records)
cpjump1_u2os_exp_metadata = cpjump1_experimental_data.filter(
    pl.col("Cell_type") == "U2OS"
)

# Filter A549 cells with density of 100 for consistency
cpjump1_a549_exp_metadata = cpjump1_experimental_data.filter(
    (pl.col("Cell_type") == "A549") & (pl.col("Density") == 100)
)

# Extract plate identifiers for each cell type
u2os_plates = cpjump1_u2os_exp_metadata["Assay_Plate_Barcode"].unique().to_list()
a549_plates = cpjump1_a549_exp_metadata["Assay_Plate_Barcode"].unique().to_list()

# Display the extracted plates for verification
print(f"U2OS plates: {u2os_plates}")
print(f"A549 plates: {a549_plates}")


# Add the MoA data into the profiles

# In[ ]:


# merge moa data (join on Metadata_pert_iname)
cpjump1_df = cpjump1_df.filter(pl.col("Metadata_Plate").is_in(u2os_plates))
cpjump1_df = cpjump1_df.join(cpjump1_moa_df, how="inner", on="Metadata_pert_iname")


# Verify all required columns exist
required_cols = [
    "Metadata_control_type",
    "Metadata_pert_iname",
    "Metadata_moa",
]
missing_cols = [col for col in required_cols if col not in cpjump1_df.columns]
if missing_cols:
    raise ValueError(f"Missing required columns: {missing_cols}")

# print dataframe information
# displaying dataframe information
print("CPJUMP1 U2OS dataset")
print(f"Dataframe shape: {cpjump1_df.shape}")
print(
    "Number of poscon_cp",
    cpjump1_df.filter(pl.col("Metadata_control_type") == "poscon_cp")[
        "Metadata_pert_iname"
    ].n_unique(),
)
print(
    "Number of unique treatments that are not controls",
    cpjump1_df.filter(pl.col("Metadata_pert_type") == "trt")
    .select("Metadata_pert_iname")
    .n_unique(),
)

cpjump1_df.head()


# Generate a shuffled baseline dataset

# In[ ]:


# create a subsetted dataframe for faster testing (optional)
if subet_data:
    print("Subsetting data for testing purposes...")
    print("subsetting fraction:", subet_fraction)
    print("original dataframe shape:", cpjump1_df.shape)
    cpjump1_df = (
        cpjump1_df.group_by(["Metadata_Plate", "Metadata_Well"])
        .agg(pl.all().sample(fraction=subet_fraction, seed=0))
        .explode(pl.all().exclude(["Metadata_Plate", "Metadata_Well"]))
    )
    print(f"New dataframe shape: {cpjump1_df.shape}")


# In[ ]:


# Parameters
# negcon_sub_sample (int) - fraction of negative controls to sub-sample
# n_same_moa_treatments (int) - minimum number of treatments sharing the same MoA
negcon_sub_sample = 0.25
n_same_moa_treatments = 3
n_iterations = 5


# In[ ]:


# counts number of treatments that have the same MoA
moa_counts = (
    (
        cpjump1_df.group_by("Metadata_moa").agg(
            pl.col("Metadata_pert_iname").n_unique().alias("treatment_count")
        )
    )
    .sort("treatment_count", descending=True)
    .filter(pl.col("treatment_count") >= n_same_moa_treatments)
)

# get all treatments with MoAs that have that passes the threshold of n_same_moa_treatments
selected_treatments_df = (
    cpjump1_df.filter(
        pl.col("Metadata_moa").is_in(moa_counts["Metadata_moa"].implode())
    )
    .select("Metadata_pert_iname")
    .unique()
    .to_series()
    .to_list()
) + ["DMSO"]

# display results
pprint(
    f"Number of MoAs with at least {n_same_moa_treatments} treatments: {moa_counts.height}"
)
pprint(f"The treatments are: {selected_treatments_df}")
print(
    f"total amount of treatments to be analyzed: {moa_counts['treatment_count'].sum()}"
)
moa_counts


# In[ ]:


# reduce the profiles to only the treatments with MoAs that have at least n_same_moa_treatments
cpjump1_df = cpjump1_df.filter(
    pl.col("Metadata_pert_iname").is_in(selected_treatments_df)
)

# displaying dataframe information
print("CPJUMP1 U2OS dataset after filtering treatments by MoA counts")
print("Dataframe shape: {cpjump1_df.shape}")
print(f"Number of treatment: {cpjump1_df['Metadata_pert_iname'].n_unique()}")
cpjump1_df.head()


# In[ ]:


# make an MoA look up dictionary {"treatment_name": "MoA"}
moa_lookup = dict(
    zip(cpjump1_moa_df["Metadata_pert_iname"], cpjump1_moa_df["Metadata_moa"])
)
pprint(moa_lookup)


# In[ ]:


# Set up a logger to track the process below
logger = logging.getLogger("buscar_moa_analysis")
logger.setLevel(logging.INFO)

# Create file handler which logs even debug messages
log_file_path = moa_analysis_output_dir / "buscar_moa_analysis.log"
fh = logging.FileHandler(log_file_path)
fh.setLevel(logging.INFO)

# Create formatter and add it to the handlers
formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
fh.setFormatter(formatter)

# Add the handlers to the logger
if not logger.hasHandlers():
    logger.addHandler(fh)
else:
    # Avoid duplicate handlers in Jupyter
    logger.handlers.clear()
    logger.addHandler(fh)

logger.info("Logger initialized for Buscar MoA analysis.")


# In[ ]:


# Run Buscar analysis for each treatment in both original and shuffled datasets
# store results here with dataset_type as top-level key
scores = {
    "original": {},
    "shuffled": {},
}
for dataset_type in ["original", "shuffled"]:
    logger.info(f"Starting analysis for dataset: {dataset_type}")
    for treatment in (
        pbar := tqdm(selected_treatments_df, desc=f"{dataset_type} treatments")
    ):
        # skip DMSO treatment
        if treatment == "DMSO":
            continue

        # getting current iteration for progress tracking
        current_iter = pbar.n
        logger.info(
            f"Processing treatment: {treatment} in dataset: {dataset_type} "
            f"Progress: {current_iter + 1}/{len(selected_treatments_df)}"
        )

        for n_iter in range(n_iterations):
            logger.info(
                f"Iteration {n_iter} for treatment: {treatment} in dataset: "
                f"{dataset_type}"
            )

            # Sample from negative controls
            negcon_df = cpjump1_df.filter(
                pl.col("Metadata_control_type") == "negcon"
            ).sample(fraction=0.025, seed=n_iter)

            # Make the selected treatment as the positive control
            target_df = cpjump1_df.filter(pl.col("Metadata_pert_iname") == treatment)

            # check the shape of negcon_df and target_df if 0 raise an error
            if negcon_df.height == 0 or target_df.height == 0:
                logger.error(
                    f"Empty dataframe encountered for treatment {treatment} in dataset "
                    f"{dataset_type} at iteration {n_iter}. "
                    f"negcon_df height: {negcon_df.height}, target_df height: "
                    f"{target_df.height}. Skipping iteration."
                )
                raise ValueError("Empty dataframe encountered.")

            logger.debug(
                f"Dataset: {dataset_type} | Treatment: {treatment} | Iteration: {n_iter}"
            )

            # Buscar step 1: identify on and off signatures
            on_signatures, off_signatures, _ = get_signatures(
                ref_profiles=negcon_df,
                exp_profiles=target_df,
                morph_feats=shared_features,
                test_method="ks_test",
                p_threshold=0.05,
                seed=n_iter,
            )

            # Skip if no on or off signatures were found
            if len(on_signatures) == 0 or len(off_signatures) == 0:
                logger.warning(
                    f"No on or off signatures found for treatment {treatment}. Skipping."
                )
                logger.debug(f"on_signatures: {len(on_signatures)}")
                logger.debug(f"off_signatures: {len(off_signatures)}")
                raise ValueError("No on or off signatures found.")

            # Buscar step 2: measure phenotypic activity and rank treatments
            # (lower is better)
            logger.debug("measuring phenotypic activity...")
            profiles = pl.concat(
                [
                    negcon_df,
                    cpjump1_df.filter(pl.col("Metadata_pert_iname") != "DMSO"),
                ]
            )
            if dataset_type == "shuffled":
                logger.debug(
                    "shuffled dataset - using shuffled profiles for phenotypic activity measurement"
                )
                profiles = shuffle_feature_profiles(
                    profiles, shared_features, seed=n_iter
                )

            treatment_rankings = measure_phenotypic_activity(
                profiles=profiles,
                meta_cols=cpjump1_meta,
                on_signature=on_signatures,
                off_signature=off_signatures,
                ref_state="DMSO",
                target_state=treatment,
                treatment_col="Metadata_pert_iname",
                emd_n_threads=-1,  # using all threads for EMD calculation
            )

            # Skip if no treatment rankings were generated
            if treatment_rankings.height == 0:
                logger.warning("No treatment scores calculated.. skipping")
                continue

            # Prepare results for this iteration
            logger.debug("storing results for this iteration...")
            result = {
                "on_scores": dict(
                    zip(treatment_rankings["treatment"], treatment_rankings["on_score"])
                ),
                "off_scores": dict(
                    zip(
                        treatment_rankings["treatment"], treatment_rankings["off_score"]
                    )
                ),
                "ranks": dict(
                    zip(treatment_rankings["treatment"], treatment_rankings["rank"])
                ),
                "moa": moa_lookup[treatment],
            }

            # Store per treatment and per iteration under dataset_type
            if treatment not in scores[dataset_type]:
                scores[dataset_type][treatment] = {}

            iteration_key = f"iteration_{n_iter}"
            scores[dataset_type][treatment][iteration_key] = result

            # Save after each iteration
            with open(
                (moa_analysis_output_dir / "cpjump1_buscar_scores.json").resolve(
                    strict=False
                ),
                "w",
            ) as f:
                json.dump(scores, f, indent=4, default=str)
            logger.info(
                f"Saved results for treatment: {treatment}, iteration: {n_iter}, dataset: {dataset_type}"
            )


# In[ ]:
