#!/usr/bin/env python

# # Downloading Single-Cell Profiles
#
# This notebook focuses on downloading metadata and single-cell profiles from three key datasets:
#
# 1. **CPJUMP1 Pilot Dataset** ([link](https://github.com/jump-cellpainting/2024_Chandrasekaran_NatureMethods_CPJUMP1)): Metadata is downloaded and processed to identify and organize plates containing wells treated with compound perturbations for downstream analysis.
# 2. **MitoCheck Dataset**: Normalized and feature-selected single-cell profiles are downloaded for further analysis.
# 3. **CFReT Dataset**: Normalized and feature-selected single-cell profiles from the CFReT plate are downloaded for downstream analysis.

# In[1]:


import gzip
import pathlib
import sys
import tarfile
import zipfile

import polars as pl
import requests
from tqdm import tqdm

sys.path.append("../../")
from utils import io_utils

# ## Helpler functions

# In[2]:


def download_file(
    source_url: str,
    output_path: pathlib.Path | str,
    chunk_size: int = 8192,
) -> pathlib.Path:
    """Downloads a file from a URL with progress tracking.

    Downloads a file from the specified URL and saves it to the given output path.
    The download is performed in chunks to handle large files efficiently, and the progress is displayed using
    the `tqdm` library.

    Parameters
    ----------
    source_url : str
        URL to download the file from.
    output_path : pathlib.Path | str
        Full path where the file should be saved.
    chunk_size : int, optional
        Size of chunks to download in bytes. Defaults to 8192.

    Returns
    -------
    pathlib.Path
        The path where the file was downloaded.

    Raises
    ------
    requests.exceptions.RequestException
        If there is an error during the download request.
    TypeError
        If input types are invalid.
    FileNotFoundError
        If the output directory does not exist.
    """
    # type checking
    if not isinstance(source_url, str):
        raise TypeError(f"source_url must be a string, got {type(source_url)}")
    if not isinstance(output_path, (pathlib.Path, str)):
        raise TypeError(
            f"output_path must be a pathlib.Path or str, got {type(output_path)}"
        )
    if isinstance(output_path, str):
        output_path = pathlib.Path(output_path)
    if not output_path.parent.exists():
        raise FileNotFoundError(
            f"Output directory {output_path.parent} does not exist."
        )
    if output_path.exists() and not output_path.is_file():
        raise FileExistsError(f"Output path {output_path} exists and is not a file.")

    # starting downloading process
    try:
        # sending GET request to the source URL
        with requests.get(source_url, stream=True) as response:
            # raise an error if the request was unsuccessful
            response.raise_for_status()

            # get the total size of the file from the response headers
            total_size = int(response.headers.get("content-length", 0))

            # using tqdm to track the download progress
            with (
                open(output_path, "wb") as file,
                tqdm(
                    desc="Downloading",
                    total=total_size,
                    unit="B",
                    unit_scale=True,
                    unit_divisor=1024,
                ) as pbar,
            ):
                # iterating over the response content in chunks
                for chunk in response.iter_content(chunk_size=chunk_size):
                    if chunk:
                        file.write(chunk)

                        # this updates the progress bar
                        pbar.update(len(chunk))
        return output_path

    except requests.exceptions.RequestException as e:
        raise requests.exceptions.RequestException(f"Error downloading file: {e}")
    except Exception as e:
        raise Exception(f"Unexpected error during download: {e}")


def extract_file(
    file_path: pathlib.Path | str,
    extract_dir: pathlib.Path | str | None = None,
) -> None:
    """Extracts a compressed file (zip, tar, tar.gz, tgz, gz).

    Parameters
    ----------
    file_path : pathlib.Path | str
        Path to the compressed file.
    extract_dir : pathlib.Path | str, optional
        Directory where the file should be extracted. If None, extracts to the same directory as the file.

    Returns:
    --------
    None
        Extracted files are saved in the specified extract_dir or in the same
        directory if the extract_dir option is None

    """
    # type checking
    if isinstance(file_path, str):
        file_path = pathlib.Path(file_path)

    if not file_path.exists():
        raise FileNotFoundError(f"File {file_path} does not exist.")

    if extract_dir is None:
        extract_dir = file_path.parent
    elif isinstance(extract_dir, str):
        extract_dir = pathlib.Path(extract_dir)

    extract_dir.mkdir(parents=True, exist_ok=True)

    try:
        if file_path.suffix == ".gz" and not file_path.name.endswith(".tar.gz"):
            # handle single gzip files
            extracted_path = extract_dir / file_path.with_suffix("").name
            with gzip.open(file_path, "rb") as f_in:
                with open(extracted_path, "wb") as f_out:
                    f_out.write(f_in.read())
            print(f"Extracted to: {extracted_path}")

        elif file_path.suffix == ".zip":
            # handle zip files
            with zipfile.ZipFile(file_path, "r") as zip_ref:
                zip_ref.extractall(extract_dir)
            print(f"Extracted to: {extract_dir}")

        elif (
            file_path.suffix in [".tar", ".tgz"]
            or ".tar." in file_path.name
            or file_path.name.endswith(".tar.gz")
        ):
            # handle tar files
            with tarfile.open(file_path, "r:*") as tar_ref:
                tar_ref.extractall(extract_dir)
            print(f"Extracted to: {extract_dir}")
        else:
            print(f"Unsupported file format for extraction: {file_path.suffix}")

    except Exception as e:
        raise Exception(f"Unexpected error during extraction: {e}")


def download_compressed_file(
    source_url: str,
    output_path: pathlib.Path | str,
    chunk_size: int = 8192,
    extract: bool = True,
) -> None:
    """Downloads and optionally extracts a compressed file."""
    downloaded_path = download_file(source_url, output_path, chunk_size)
    if extract:
        extract_file(downloaded_path)


# ## Downloading data

# Parameters used in this notebook

# In[3]:


# setting perturbation type
# other options are "compound", "orf",
pert_type = "compound"


# setting input and output paths

# In[4]:


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

# create cpjump1 directory
cpjump1_dir = (profiles_dir / "cpjump1").resolve()
cpjump1_dir.mkdir(exist_ok=True)

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

# In[5]:


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
exp_metadata.write_csv(cpjump1_dir / f"cpjump1_{pert_type}_experimental-metadata.csv")

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

# In[6]:


# url source for the MitoCheck data
mitocheck_url = nb_configs["links"]["MitoCheck-profiles-source"]
save_path = (mitocheck_dir / "normalized_data").resolve()
if save_path.exists():
    print(f"File {save_path} already exists. Skipping download.")
else:
    download_compressed_file(mitocheck_url, save_path)


# ## Downloading CFReT Data
#
# In this section, we download feature-selected single-cell profiles from the CFReT plate `localhost230405150001`. This plate contains three treatments: DMSO (control), drug_x, and TGFRi. The dataset consists of high-content imaging data that has already undergone feature selection, making it suitable for downstream analysis.
#
# **Key Points:**
# - Only the processed single-cell profiles are downloaded [here](https://github.com/WayScience/cellpainting_predicts_cardiac_fibrosis/tree/main/3.process_cfret_features/data/single_cell_profiles)
# - The CFReT dataset was used and published in [this study](https://doi.org/10.1161/CIRCULATIONAHA.124.071956).

# In[7]:


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
