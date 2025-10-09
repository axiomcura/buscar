import pathlib
import sys

sys.path.append("../../")
from utils.heterogeneity import optimized_clustering
from utils.io_utils import load_profiles

cluster_search_param_grid = {
    # Clustering resolution: how granular the clusters should be
    "cluster_resolution": {"type": "float", "low": 0.1, "high": 3.0},
    # Number of neighbors for graph construction
    "n_neighbors": {"type": "int", "low": 5, "high": 50},
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

# set module and data directory paths
download_module_path = pathlib.Path("../0.download-data/").resolve(strict=True)
sc_profiles_path = (download_module_path / "data" / "sc-profiles").resolve(strict=True)


# setting profiles paths
cfret_profiles_path = (
    sc_profiles_path / "cfret" / "localhost230405150001_sc_feature_selected.parquet"
).resolve(strict=True)
cpjump1_trt_crispr_profiles_path = (
    sc_profiles_path
    / "cpjump1"
    / "trt-profiles"
    / "cpjump1_crispr_trt_profiles.parquet"
).resolve(strict=True)
mitocheck_trt_profiles_path = (
    sc_profiles_path / "mitocheck" / "mitocheck_concat_profiles.parquet"
).resolve(strict=True)

# create signature output paths
results_dir = pathlib.Path("./results/cluster-labels").resolve()
results_dir.mkdir(exist_ok=True, parents=True)

cpjump1_crispr_profiles_df = load_profiles(cpjump1_trt_crispr_profiles_path)


# split metadata and features for cpjump1
cpjump1_meta = [
    "index",
    "Metadata_broad_sample",
    "Metadata_ImageNumber",
    "Metadata_Plate",
    "Metadata_Site",
    "Metadata_Well",
    "Metadata_TableNumber",
    "Metadata_ObjectNumber_cytoplasm",
    "Metadata_Cytoplasm_Parent_Cells",
    "Metadata_Cytoplasm_Parent_Nuclei",
    "Metadata_ObjectNumber_cells",
    "Metadata_ObjectNumber",
    "Metadata_gene",
    "Metadata_pert_type",
    "Metadata_control_type",
    "Metadata_target_sequence",
    "Metadata_negcon_control_type",
    "__index_level_0__",
]

# split metadata and features for cpjump1
cpjump1_feats = cpjump1_crispr_profiles_df.drop(cpjump1_meta).columns

cpjump1_cluster_results = optimized_clustering(
    profiles=cpjump1_crispr_profiles_df,
    meta_features=cpjump1_meta,
    morph_features=cpjump1_feats,
    treatment_col="Metadata_gene",
    param_grid=cluster_search_param_grid,
    n_trials=30,
)
