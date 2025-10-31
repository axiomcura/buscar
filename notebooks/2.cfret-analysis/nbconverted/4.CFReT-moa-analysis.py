#!/usr/bin/env python

# In[8]:


import json
import pathlib
import sys

import numpy as np
import polars as pl

sys.path.append("../../")
from utils.data_utils import split_meta_and_features
from utils.identify_hits import identify_compound_hit
from utils.metrics import measure_phenotypic_activity
from utils.signatures import get_signatures

# In[9]:


def average_precision(ranked_labels, expected_label):
    """
    Calculate Average Precision (AP).

    For each position where expected_label appears, calculate:
    - precision at that position = (# of matches so far) / (current position)

    Then average all these precision values.

    Example: ["path1", "path1", "path4", "path1", "path2"] with expected="path1"
    - Position 1: path1 → 1/1 = 1.0
    - Position 2: path1 → 2/2 = 1.0
    - Position 3: path4 → skip
    - Position 4: path1 → 3/4 = 0.75
    - Position 5: path2 → skip
    AP = (1.0 + 1.0 + 0.75) / 3 = 0.917
    """
    precisions = []
    num_matches = 0

    for position, label in enumerate(ranked_labels, start=1):
        if label == expected_label:
            num_matches += 1
            precision_at_position = num_matches / position
            precisions.append(precision_at_position)

    if len(precisions) == 0:
        return 0.0

    ap = sum(precisions) / len(precisions)
    return ap


# In[10]:


cfret_screen_path = pathlib.Path(
    "results/cfret-screen/cfret_screen_treatment_clustered.parquet"
).resolve(strict=True)

# results out dir
result_dir = pathlib.Path("results/cfret-screen").resolve(strict=True)
result_dir.mkdir(parents=True, exist_ok=True)


# In[11]:


# load profiles
cfret_df = pl.read_parquet(cfret_screen_path)
cfret_meta, cfret_feats = split_meta_and_features(cfret_df)


# In[12]:


# create a dictioanry where the Pathway is the key and the treatments are in a list value
pathway_treatments = (
    cfret_df.select(["Metadata_Pathway", "Metadata_treatment"])
    .filter(pl.col("Metadata_treatment").is_not_null())  # Remove None treatments
    .unique()
    .group_by("Metadata_Pathway")
    .agg(pl.col("Metadata_treatment").alias("treatments"))
    .to_dict(as_series=False)
)

# Convert to a more usable dict format and remove None pathways
pathway_dict = {
    pathway: treatments
    for pathway, treatments in zip(
        pathway_treatments["Metadata_Pathway"], pathway_treatments["treatments"]
    )
    if pathway is not None  # Also remove None pathways
}


# In[ ]:


# Create pathway metadata df
cfret_pathway_df = (
    cfret_df.select(["Metadata_Pathway", "Metadata_treatment"])
    .filter(pl.col("Metadata_treatment").is_not_null())
    .unique()
)

# Create log directory
log_dir = pathlib.Path("./logs")
log_dir.mkdir(parents=True, exist_ok=True)
log_path = log_dir / "cfret_moa_ap_scores.log"

# Iterate through each pathway and calculate AP
moa_scores = {}
for pathway, list_of_treatments in pathway_dict.items():
    print(f"Pathway: {pathway} Number of treatments: {len(list_of_treatments)}")
    treatment_ap_scores = []

    for i, treatment in enumerate(list_of_treatments, 1):
        # loggin which treatment is being processed
        print(f"\nProcessing treatment {i}/{len(list_of_treatments)}: {treatment}")

        # Creating signatures selecting DMSO_heart_11 as reference
        print("  Creating signatures...")
        ref_df = cfret_df.filter(pl.col("Metadata_treatment") == "DMSO_heart_11")
        target_df = cfret_df.filter(pl.col("Metadata_treatment") == treatment)
        on_sigs, off_sigs, _ = get_signatures(
            ref_profiles=ref_df,
            exp_profiles=target_df,
            morph_feats=cfret_feats,
            test_method="mann_whitney_u",
        )

        # Measure phenotypic activity using the selelected treatment as the reference
        print("  Measuring phenotypic activity...")
        treatment_phenotypic_dist_scores = measure_phenotypic_activity(
            profiles=cfret_df,
            on_signature=on_sigs,
            off_signature=off_sigs,
            ref_treatment=treatment,
            cluster_col="Metadata_cluster_id",
        )

        # Identify compound hits
        treatment_rankings = identify_compound_hit(
            distance_df=treatment_phenotypic_dist_scores, method="weighted_sum"
        )

        # Merge pathway information with treatment rankings
        print("  Merging pathway information...")
        treatment_rankings = treatment_rankings.join(
            cfret_pathway_df,
            left_on="treatment",
            right_on="Metadata_treatment",
            how="left",
        )

        # Calculate average precision for the treatment
        print("  Calculating average precision...")
        treatment_ap_score = average_precision(
            treatment_rankings["Metadata_Pathway"].to_list(),
            expected_label=pathway,
        )

        print(f"  AP Score: {treatment_ap_score:.3f}")
        treatment_ap_scores.append(treatment_ap_score)

        # making a log file
        with open(log_path, "a") as log_file:
            log_file.write(f"{pathway}\t{treatment}\t{treatment_ap_score:.6f}\n")

    # Take mean and keep as float
    mean_ap = np.mean(treatment_ap_scores)
    moa_scores[pathway] = mean_ap
    print(f"\n{'=' * 70}")
    print(f"Pathway '{pathway}' Mean AP: {mean_ap:.3f}")
    print(f"{'=' * 70}\n")


# In[ ]:


# write dictionary into a json file
moa_results_path = (result_dir / "cfret_moa_pathway_ap_scores.json").resolve(
    strict=True
)
with open(moa_results_path, "w") as f:
    json.dump(moa_scores, f, indent=4)

# convert moa_scores to a dataframe
moa_scores_df = pl.DataFrame(
    {"pathway": list(moa_scores.keys()), "ap_score": list(moa_scores.values())}
)

# sort scores
moa_scores_df = moa_scores_df.sort("ap_score", reverse=True)

# save scores to a csv file
moa_scores_path = (result_dir / "cfret_moa_pathway_ap_scores.csv").resolve(strict=True)
moa_scores_df.write_csv(moa_scores_path)
