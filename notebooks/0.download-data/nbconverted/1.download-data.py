#!/usr/bin/env python

# # Downloading Single-Cell Profiles
#
# This notebook focuses on downloading metadata and single-cell profiles from three key datasets:
#
# 1. **CPJUMP1 Pilot Dataset** ([link](https://github.com/jump-cellpainting/2024_Chandrasekaran_NatureMethods_CPJUMP1)): Metadata is downloaded and processed to identify and organize plates containing wells treated with compound perturbations for downstream analysis.
# 2. **MitoCheck Dataset**: Normalized and feature-selected single-cell profiles are downloaded for further analysis.
# 3. **CFReT Dataset**: Normalized and feature-selected single-cell profiles from the CFReT plate are downloaded for downstream analysis.

# In[1]:


import pathlib
import sys

import polars as pl

sys.path.append("../../")
from utils import io_utils

# ## Downloading data

# Parameters used in this notebook

# In[2]:


# setting perturbation type
# other options are "compound", "orf",
pert_type = "compound"


# setting input and output paths

# In[3]:


# setting config path
config_path = pathlib.Path("../nb-configs.yaml").resolve(strict=True)

# setting results setting a data directory
data_dir = pathlib.Path("./data").resolve()
data_dir.mkdir(exist_ok=True)

# setting a path to save the experimental metadata
exp_metadata_path = (data_dir / "CPJUMP1-experimental-metadata.csv").resolve()

# setting profile directory
profiles_dir = (data_dir / "sc-profiles").resolve()
profiles_dir.mkdir(exist_ok=True)

# create mitocheck directory
mitocheck_dir = (profiles_dir / "mitocheck").resolve()
mitocheck_dir.mkdir(exist_ok=True)

# create cfret directory
cfret_dir = (profiles_dir / "cfret").resolve()
cfret_dir.mkdir(exist_ok=True)


# ## Downloading CPJUMP1 Metadata
#
# In this section, we download the [experimental metadata](https://github.com/carpenter-singh-lab/2024_Chandrasekaran_NatureMethods/blob/main/benchmark/output/experiment-metadata.tsv) for the CPJUMP1 dataset. This metadata provides detailed information about each experimental batch, including plate barcodes, cell lines, perturbation types, and incubation times. Access to this metadata is essential for selecting and organizing the relevant subset of CPJUMP1 data for downstream analysis.
#
# For this notebook, we focus on plates containing both U2OS and A549 parental cell lines that have been treated with compounds for 48 hours. More information about the batch and plate metadata can be found in the [CPJUMP1 documentation](https://github.com/carpenter-singh-lab/2024_Chandrasekaran_NatureMethods/blob/main/README.md#batch-and-plate-metadata).

# In[4]:


# loading config file and setting experimental metadata URL
nb_configs = io_utils.load_configs(config_path)
CPJUMP1_exp_metadata_url = nb_configs["links"]["CPJUMP1-experimental-metadata-source"]

# read in the experimental metadata CSV file and only filter down to plays that
# have an CRISPR perturbation
exp_metadata = pl.read_csv(
    CPJUMP1_exp_metadata_url, separator="\t", has_header=True, encoding="utf-8"
)

# apply a single filter to select only rows matching all criteria
exp_metadata = exp_metadata.filter(
    (
        exp_metadata["Perturbation"].str.contains(pert_type)
    )  # selecting based on pert type
    & (exp_metadata["Time"] == 48)  # time of incubation with compound
    & (
        exp_metadata["Cell_type"].is_in(["U2OS", "A549"])
    )  # selecting based on cell type
    & (exp_metadata["Cell_line"] == "Parental")  # selecting only the parental cell line
    & (pl.col("Batch") == "2020_11_04_CPJUMP1")  # selecting only the specified batch
)

# save the experimental metadata as a csv file
exp_metadata.write_csv(exp_metadata_path)

# display
print(
    "plates that will be downloaded are: ", exp_metadata["Assay_Plate_Barcode"].unique()
)
print("shape: ", exp_metadata.shape)
exp_metadata


# ## Downloading MitoCheck Data
#
# In this section, we download the MitoCheck data generated in [this study](https://pmc.ncbi.nlm.nih.gov/articles/PMC3108885/).
#
# Specifically, we are downloading data that has already been normalized and feature-selected. The normalization and feature selection pipeline is available [here](https://github.com/WayScience/mitocheck_data/tree/main/3.normalize_data).

# In[5]:


# url source for the MitoCheck data
mitocheck_url = nb_configs["links"]["MitoCheck-profiles-source"]
save_path = (mitocheck_dir / "mitocheck_profile.parquet").resolve()

if save_path.exists():
    print(f"File {save_path} already exists. Skipping download.")
else:
    # read and download mitocheck data
    mitocheck_profile = pl.read_csv(mitocheck_url)
    mitocheck_profile.write_parquet(save_path)

    # display
    print("shape: ", mitocheck_profile.shape)
    mitocheck_profile.head()


# ## Downloading CFReT Data
#
# In this section, we download feature-selected single-cell profiles from the CFReT plate `localhost230405150001`. This plate contains three treatments: DMSO (control), drug_x, and TGFRi. The dataset consists of high-content imaging data that has already undergone feature selection, making it suitable for downstream analysis.
#
# **Key Points:**
# - Only the processed single-cell profiles are downloaded [here](https://github.com/WayScience/cellpainting_predicts_cardiac_fibrosis/tree/main/3.process_cfret_features/data/single_cell_profiles)
# - The CFReT dataset was used and published in [this study](https://doi.org/10.1161/CIRCULATIONAHA.124.071956).

# In[6]:


# setting the source for the CFReT data
cfret_source = nb_configs["links"]["CFReT-profiles-source"]

# use the correct filename from the source URL
output_path = (
    cfret_dir / "localhost230405150001_sc_feature_selected.parquet"
).resolve()

# check if it exists
if output_path.exists():
    print(f"File {output_path} already exists. Skipping download.")
else:
    # download cfret data
    cfret_df = pl.read_parquet(cfret_source)
    cfret_df.write_parquet(output_path)

    # display
    print("shape: ", cfret_df.shape)
    cfret_df.head()
