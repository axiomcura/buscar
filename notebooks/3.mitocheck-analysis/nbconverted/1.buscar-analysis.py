#!/usr/bin/env python

# In[ ]:


import pathlib
import sys

import polars as pl
from tqdm import tqdm

sys.path.append("../../")
from buscar.metrics import measure_phenotypic_activity
from buscar.signatures import get_signatures
from utils.io_utils import load_configs, load_profiles

# In[ ]:


# set data path
data_dir = pathlib.Path("../0.download-data/data/sc-profiles/").resolve(strict=True)
mitocheck_data = (data_dir / "mitocheck").resolve(strict=True)

# sertting mitocheck paths
mitocheck_profile_path = (mitocheck_data / "mitocheck_concat_profiles.parquet").resolve(
    strict=True
)

# setting config paths
ensg_genes_config_path = (
    mitocheck_data / "mitocheck_ensg_to_gene_symbol_mapping.json"
).resolve(strict=True)
mitocheck_feature_space_config = (
    mitocheck_data / "mitocheck_feature_space_configs.json"
).resolve(strict=True)

# set results output path
results_dir = pathlib.Path("./results/").resolve()
results_dir.mkdir(exist_ok=True)

moa_analysis_output = (results_dir / "moa_analysis").resolve()
moa_analysis_output.mkdir(exist_ok=True)


# In[ ]:


# load in configs
ensg_genes_decoder = load_configs(ensg_genes_config_path)
feature_space_configs = load_configs(mitocheck_feature_space_config)
meta_feats = feature_space_configs["metadata-features"]
morph_feats = feature_space_configs["morphology-features"]


# In[ ]:


# load in mitocheck profiles
mitocheck_df = load_profiles(mitocheck_profile_path)
mitocheck_df = mitocheck_df.select(pl.col(meta_feats + morph_feats))

# removing failed qc
mitocheck_df = mitocheck_df.filter(pl.col("Metadata_Gene") != "failed QC")

# replace "negative_control" and "positive_control" values in Metadata_Gene with
# "negcon" and "poscon" respectively
mitocheck_df = mitocheck_df.with_columns(
    pl.col("Metadata_Gene").map_elements(
        lambda x: (
            "negcon"
            if x == "negative control"
            else ("poscon" if x == "positive control" else x)
        ),
        return_dtype=pl.String,
    )
)


# In[ ]:


labeled_mitocheck_df = mitocheck_df.filter(
    (pl.col("Mitocheck_Phenotypic_Class") != "negcon")
    & (pl.col("Mitocheck_Phenotypic_Class") != "poscon")
)

print("Shape of the labeled mitocheck profiles:", labeled_mitocheck_df.shape)
labeled_mitocheck_df.head()


# Generate proportion of cells states per treatment

# In[ ]:


# Creating a proportion dataframe for all genes and phenotypic classes
cell_proportion_df = (
    mitocheck_df.filter(
        (pl.col("Mitocheck_Phenotypic_Class") != "negcon")
        & (pl.col("Mitocheck_Phenotypic_Class") != "poscon")
    )
    .group_by(["Metadata_Gene", "Mitocheck_Phenotypic_Class"])
    .agg(pl.len().alias("count"))
    .with_columns(pl.col("count").sum().over("Metadata_Gene").alias("total_count"))
    .with_columns((pl.col("count") / pl.col("total_count")).alias("proportion"))
)


# ## Analysis 1: Positive Control Ranking
#
# We evaluate whether our on/off morphological signatures can correctly rank genes based on their association with the **Prometaphase** phenotype.
#
# Two reference states are used to define the signatures:
# - positive control: Prometaphase
# - negative control: Interphase
#
# We expect the ranking to reflect three tiers of phenotypic activity:
# 1. **High activity** — genes with a dominant Prometaphase phenotype
# 2. **Intermediate activity** — genes with a mixture of Prometaphase and other phenotypes
# 3. **Low activity** — genes with no Prometaphase phenotype, but other dominant phenotypes

# In[ ]:


negcon_state = "Interphase"
poscon_state = "Prometaphase"

# generating negative control profiles (paper states they are interphase)
# paper states negative controls are interphase profiles
negcon_profiles = mitocheck_df.filter(
    pl.col("Mitocheck_Phenotypic_Class") == "negcon"
).sample(fraction=0.1, seed=0)

# poscon  phenotype of intrest in this case Prometaphase
poscon_profiles = labeled_mitocheck_df.filter(
    pl.col("Mitocheck_Phenotypic_Class") == poscon_state
)


# In[ ]:


# generate on and off signatures with pooled negcon and poscon profiles
on_sigs, off_sigs, _ = get_signatures(
    ref_profiles=poscon_profiles,
    exp_profiles=negcon_profiles,
    morph_feats=morph_feats,
    p_threshold=0.05,
    test_method="ks_test",
)

# measure phenotypic activity of the signatures
prometa_phase_ranks = measure_phenotypic_activity(
    profiles=pl.concat([negcon_profiles, poscon_profiles]),
    meta_cols=meta_feats,
    on_signature=on_sigs,
    off_signature=off_sigs,
    ref_state=poscon_state,
    target_state=negcon_state,
    treatment_col="Metadata_Gene",
    state_col="Mitocheck_Phenotypic_Class",
    emd_n_threads=-1,
)

# remove negcon and poscon from the ranks dataframe
prometa_phase_ranks = prometa_phase_ranks.filter(
    (pl.col("treatment") != "negcon") & (pl.col("treatment") != "poscon")
)


# In[ ]:


# add cell proportion information to the prometa_phase_ranks dataframe
prometa_phase_ranks = prometa_phase_ranks.join(
    cell_proportion_df.select(
        ["Metadata_Gene", "Mitocheck_Phenotypic_Class", "proportion"]
    ),
    left_on=["treatment", "ref_profile"],
    right_on=["Metadata_Gene", "Mitocheck_Phenotypic_Class"],
    how="left",
).with_columns(pl.col("proportion").fill_null(0.0))

prometa_phase_ranks


# ## Analysis 2: Leave-One-Gene-Out Analysis
#
# In this analysis, we perform a leave-one-gene-out (LOGO) evaluation to assess whether data leakage from pooling single-cell profiles inflates phenotypic activity scores.
#
# For each gene known to be associated with the **Prometaphase** phenotype:
# 1. Its Prometaphase cells are **excluded** from building the on/off signatures.
# 2. The on/off signatures are computed from the remaining Prometaphase population.
# 3. The **excluded gene's cells** are then scored against those signatures using EMD.
#
# Here, **Prometaphase is used as the reference baseline**, so scores reflect how close the held-out gene's cells are to the Prometaphase phenotype. This means:
# - **Lower scores = good** — the held-out gene's cells are morphologically similar to Prometaphase, indicating genuine phenotypic signal.
# - If data leakage were present (i.e., the gene's own cells contributed to the signature), scores would be artificially low. Under the LOGO design, **scores that remain low confirm the signal is real** — those cells genuinely resemble Prometaphase even when they played no role in building the signature.
#

# Get cell state information

# In[ ]:


cell_states = (
    # remove negcon and poscon since they do not have cell state information
    mitocheck_df.filter(
        (pl.col("Mitocheck_Phenotypic_Class") != "negcon")
        & (pl.col("Mitocheck_Phenotypic_Class") != "poscon")
    )
    .select("Mitocheck_Phenotypic_Class")
    .unique()
    .to_series()
    .to_list()
)


# Caclulate the proportion of cell states that makes up a specific gene

# In[ ]:


on_off_sigs = []
min_cells = 2  # EMD requires at least 2 samples

results_df = []
for cell_state in tqdm(cell_states, desc="Processing cell states"):
    # poscon  phenotype of intrest in this case Prometaphase
    poscon_profiles = labeled_mitocheck_df.filter(
        pl.col("Mitocheck_Phenotypic_Class") == cell_state
    )

    # generate that are associated with the prometaphase
    genes_associated_with_state = (
        poscon_profiles.select("Metadata_Gene").unique().to_series().to_list()
    )

    # genes that are not associated with prometaphase
    genes_not_associated_with_state = (
        labeled_mitocheck_df.filter(
            ~pl.col("Metadata_Gene").is_in(genes_associated_with_state)
        )
        .select("Metadata_Gene")
        .unique()
        .to_series()
        .to_list()
    )

    associated_gene_scores = []
    for gene in tqdm(
        genes_associated_with_state,
        desc=f"  Processing genes for {cell_state}",
        leave=False,
    ):
        # filter the target profiles to only include cells treated with the current
        # gene of interest
        target_profiles = poscon_profiles.filter(pl.col("Metadata_Gene") == gene)

        # skip genes with too few cells (EMD requires >= 2 samples)
        if target_profiles.height < min_cells:
            print(
                f"Skipping gene '{gene}': only {target_profiles.height} cell(s), need >= "
                f"{min_cells}"
            )
            # create an empty dataframe with the same structure as the
            # associated_gene_score to maintain consistency
            associated_gene_score = pl.DataFrame(
                {
                    "rank": pl.Series([None], dtype=pl.UInt32),
                    "ref_profile": pl.Series([cell_state], dtype=pl.String),
                    "treatment": pl.Series([gene], dtype=pl.String),
                    "on_score": pl.Series([None], dtype=pl.Float64),
                    "off_score": pl.Series([None], dtype=pl.Float64),
                    "proportion": pl.Series([None], dtype=pl.Float64),
                }
            )
            associated_gene_scores.append(associated_gene_score)
            continue

        # remove the current gene's Prometaphase cells from the positive control profiles
        # to prevent data leakage: the gene being ranked must not influence its own signature
        state_pool = poscon_profiles.filter(pl.col("Metadata_Gene") != gene)

        # generate on and off signatures (leave-one-out: current gene's cells excluded)
        on_sig, off_sig, _ = get_signatures(
            state_pool,
            negcon_profiles,
            morph_feats=feature_space_configs["morphology-features"],
            test_method="ks_test",
            p_threshold=0.05,
        )

        # if not signature was found, skip the gene
        if len(on_sig) == 0 or len(off_sig) == 0:
            print(f"skipping {gene}")
            continue

        # rank the gene using the generated signatures
        associated_gene_score = measure_phenotypic_activity(
            profiles=pl.concat([negcon_profiles, target_profiles]),
            meta_cols=feature_space_configs["metadata-features"],
            on_signature=on_sig,
            off_signature=off_sig,
            target_state="negcon",
            ref_state=cell_state,
            treatment_col="Metadata_Gene",
            state_col="Mitocheck_Phenotypic_Class",
            emd_n_threads=-1,
        )

        # caclulate the proportion of cells that make up this phenotype with the
        # current gene perturbation
        try:
            cell_state_proportion = cell_proportion_df.filter(
                (pl.col("Metadata_Gene") == gene)
                & (pl.col("Mitocheck_Phenotypic_Class") == cell_state)
            )["proportion"][0]
        except IndexError as e:
            print(
                f"Index error caught for gene '{gene}' and cell state '{cell_state}': {e}"
            )
            cell_state_proportion = 0.0

        # remove negcon scores we are only intrested of the scores of the gene
        associated_gene_score = associated_gene_score.filter(
            pl.col("treatment") != "negcon"
        )

        # add cell state proportion to the associated gene scores df
        associated_gene_score = associated_gene_score.with_columns(
            pl.lit(cell_state_proportion).alias("proportion"),
        )

        # store on and off signatures and
        on_off_sigs.append((cell_state, on_sig, off_sig))
        associated_gene_scores.append(associated_gene_score)

    associated_gene_scores = pl.concat(associated_gene_scores)

    # Step 2: ranking genes that are not associated with the prometaphase state

    # creating on and off sigs with pooled poscon cell state
    on_sig, off_sig, _ = get_signatures(
        ref_profiles=poscon_profiles,
        exp_profiles=negcon_profiles,
        morph_feats=morph_feats,
        test_method="ks_test",
        p_threshold=0.05,
    )

    # rank all treatments that are not associated with the prometaphase state using the pooled poscon signatures
    not_associated_gene_scores = measure_phenotypic_activity(
        profiles=pl.concat(
            [
                poscon_profiles,
                mitocheck_df.filter(
                    pl.col("Metadata_Gene").is_in(genes_not_associated_with_state)
                ),
            ]
        ),
        meta_cols=meta_feats,
        on_signature=on_sig,
        off_signature=off_sig,
        target_state="negcon",
        ref_state=cell_state,
        treatment_col="Metadata_Gene",
        state_col="Mitocheck_Phenotypic_Class",
        emd_n_threads=-1,
    )

    # remove scores of genes that are associated with the cell state
    not_associated_gene_scores = not_associated_gene_scores.filter(
        pl.col("treatment").is_in(genes_not_associated_with_state)
    )

    # add proportion of cells, if a gene does not have any cells in the cell state,
    # assign a proportion of 0
    not_associated_gene_scores = not_associated_gene_scores.join(
        cell_proportion_df.select(
            ["Metadata_Gene", "Mitocheck_Phenotypic_Class", "proportion"]
        ),
        left_on=["treatment", "ref_profile"],
        right_on=["Metadata_Gene", "Mitocheck_Phenotypic_Class"],
        how="left",
    ).with_columns(pl.col("proportion").fill_null(0.0))

    # final result
    results_df.append(
        pl.concat([associated_gene_scores, not_associated_gene_scores], how="vertical")
    )

# step 3: store results
results_df = pl.concat(results_df)
results_df.write_parquet(moa_analysis_output / "mitocheck_moa_analysis_results.parquet")
