from enum import Enum
from datetime import timedelta

import pandas as pd
import numpy as np

# If your environment has sklearn available:
from sklearn.linear_model import LinearRegression

from src.lib.interface import (
    TrainerInterface,
    Dataset,
    RegressorWithPredict,
    FeaturesPreprocessor,
)

from sklearn.impute import KNNImputer

# temporary: imputer
def knn_impute_targets(df_feat, df_targ, n_neighbors=5):
    """
    Impute missing values in df_targ using K-Nearest Neighbors (KNN) based on df_feat.

    Parameters:
    - df_feat: DataFrame containing feature data with the same MultiIndex as df_targ.
    - df_targ: DataFrame containing target data with missing values (NaNs) to be imputed.
    - n_neighbors: Number of neighbors to use for KNN imputation.

    Returns:
    - df_targ_filled: DataFrame with the same structure as df_targ, with missing values imputed.
    """
    # Ensure the indexes of df_feat and df_targ align
    if not df_feat.index.equals(df_targ.index):
        raise ValueError("The indexes of df_feat and df_targ must match.")

    # Combine feature and target data
    combined = df_feat.join(df_targ, how='inner')

    # Apply KNN imputation
    imputer = KNNImputer(n_neighbors=n_neighbors)
    imputed_data = imputer.fit_transform(combined)

    # Reconstruct the DataFrame with the original index and columns
    combined_imputed = pd.DataFrame(imputed_data, index=combined.index, columns=combined.columns)

    # Extract the imputed target data
    df_targ_filled = combined_imputed[df_targ.columns]

    return df_targ_filled

class SimplePreprocessor(FeaturesPreprocessor):
    """
    A trivial preprocessor that fills NaN values with 0
    and keeps track of feature names.
    """
    EXTERNAL_NAME = "SimplePreprocessor"
    valid_index: pd.DataFrame
    feature_names: list[str]

    def __init__(self):
        self.valid_index = pd.DataFrame()
        self.feature_names = []

    def transform(
            self,
            X: pd.DataFrame,
            y: pd.DataFrame | None = None
    ) -> tuple[pd.DataFrame, pd.DataFrame | None]:
        self.feature_names = X.columns.tolist()

        # fix nans across all folds
        X = X.interpolate(method="linear")
        y = knn_impute_targets(X, y, n_neighbors=5)

        return X, y

    def update_valid_index(self, X: pd.DataFrame, only_on: list[str] | None = None):
        # Not used here
        pass


class SimpleRegressor(RegressorWithPredict):
    """
    A scikit-learn wrapper that satisfies the RegressorWithPredict interface.
    """
    EXTERNAL_NAME = "SimpleRegressor"

    def __init__(self):
        self.model = LinearRegression()

    def fit(self, X: pd.DataFrame, y: pd.DataFrame):
        self.model.fit(X, y)

    def predict(self, X: pd.DataFrame | np.ndarray) -> np.ndarray:
        return self.model.predict(X)


class SimpleTrainer(TrainerInterface):
    """
    A minimal Trainer implementing TrainerInterface.
    """
    MODEL_CLASS = "LinearRegression"  # used for reference
    TRAINING_DATASET_LENGTH = timedelta(hours=27*24 + 28) # This is bad... But i run out of time
    EVALUATION_DATASET_LENGTH = timedelta(days=1)

    @property
    def EXTERNAL_NAME(self) -> str:
        return "SimpleTrainer"

    def train(
            self,
            dataset: Dataset,
    ) -> tuple[RegressorWithPredict, FeaturesPreprocessor, pd.DataFrame]:
        # Unpack train split
        X_train, _, y_train, _ = dataset.split()

        # Instantiate the preprocessor
        preprocessor = SimplePreprocessor()

        # Transform X_train, y_train
        X_train_transformed, y_train_transformed = preprocessor.transform(X_train, y_train)

        # Create and fit the regressor
        model = SimpleRegressor()
        model.fit(X_train_transformed, y_train_transformed)

        # Return (model, preprocessor, X_train)
        return model, preprocessor, X_train



class Trainers(Enum):
    SimpleTrainer = SimpleTrainer

    def __str__(self) -> str:
        return self.value.__name__
