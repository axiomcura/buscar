"""
This module contains utility functions for data preprocessing
"""

import polars as pl
from beartype import beartype
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


@beartype
def apply_pca(
    profiles: pl.DataFrame,
    meta_features: list[str],
    morph_features: list[str],
    var_explained=0.95,
    svd_solver="randomized",
    standardize=False,
    random_state=0,
    **kwargs,
) -> pl.DataFrame:
    """Apply PCA to the morphological features of the profiles DataFrame.

    Parameters
    ----------
    profiles : pl.DataFrame
        Input DataFrame containing morphological and metadata features.
    meta_features : list[str]
        List of column names corresponding to metadata features.
    morph_features : list[str]
        List of column names corresponding to morphological features.
    var_explained : float, optional
        The amount of variance to be explained by the selected components.
        Default is 0.95.
    svd_solver : str, optional
        The SVD solver to use. Default is "randomized".
    standardize : bool, optional
        Whether to standardize the morphological features before applying PCA.
        Default is False.
    random_state : int, optional
        Random state for reproducibility. Default is 0.
    **kwargs
        Additional keyword arguments for PCA that can be found here:
        https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.PCA.html

    Returns
    -------
    pl.DataFrame
        DataFrame containing the metadata features and the principal components.

    """

    # check if there are nans in the feature space
    if profiles.select(morph_features).null_count().sum_horizontal().sum() > 0:
        raise ValueError(
            "Input data contains NaNs. Please handle them before applying PCA."
        )

    # standardize data if specified
    if standardize:
        scaler = StandardScaler()
        data_scaled = scaler.fit_transform(profiles.select(morph_features).to_numpy())
    else:
        data_scaled = profiles.select(morph_features).to_numpy()

    # apply PCA
    pca = PCA(
        n_components=var_explained,
        svd_solver=svd_solver,
        random_state=random_state,
        **kwargs,
    )
    principal_components = pca.fit_transform(data_scaled)

    # concat metadata infromation with principal components
    pca_colnames = [f"PC{i + 1}" for i in range(principal_components.shape[1])]
    return pl.concat(
        [
            profiles.select(meta_features),  # metadata df
            pl.DataFrame(
                principal_components, schema=pca_colnames
            ),  # PCA components df
        ],
        how="horizontal",
    )
