"""
Contains the utility script for model fitting
"""

# External modules
import numpy as np
import pandas as pd
from copy import deepcopy
from typing import Optional
from glmnet import ElasticNet
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
# Internal modules
from utils import vprint


di_scoring = {'r2': +1,
              'mean_squared_error': -1,
              'mean_absolute_error': -1, 
              'median_absolute_error': -1}

class elnet_wrapper:
    def __init__(self,
                 alphas: float | np.ndarray = 1,
                 random_state: Optional[int] = None,
                 scoring: str = 'r2',
                 num_imputer_strategy: str = 'median',
                 cat_imputer_strategy: str = 'most_frequent',
                 **kwargs
                ) -> None:
        """
        Initialize the ModelPipeline class with specified configurations.

        Args:
            alphas: float | np.ndarray
                An array of alphas to fit the model over (defaults to 1)
            random_state (Optional[int]): 
                Seed for the random number generator.
            scoring: str
                One of the valid scoring metrics for glmnet.ElasticNet
            num_imputer_strategy (str): 
                Imputation strategy for numerical columns.
            cat_imputer_strategy (str): 
                Imputation strategy for categorical columns.
            **kwargs
                Any arguments to pass to the Elasticnet model

        Returns:
            None
        """
        # Input checks
        if not isinstance(alphas, np.ndarray):
            alphas = np.array(alphas)
        assert np.all( (alphas >= 0) & (alphas <= 1) ), 'alpha(s) need to be between 0 and 1'
        assert scoring in di_scoring, f'Could not find {scoring} in {list(di_scoring)}'
        # Store the values
        self.random_state = random_state
        self.num_imputer_strategy = num_imputer_strategy
        self.cat_imputer_strategy = cat_imputer_strategy
        self.pipeline = None
        self.model = None
        self.preprocessor = None
        self._is_fitted = False
        self.target_scaler = None
        self.alphas = np.atleast_1d(alphas)
        self.scoring = scoring
        self.best_alpha = None
        # Set a default scoring if it doesn't exist in kwargs
        self.kwargs = {**{'scoring': scoring}, **kwargs}


    def _create_preprocessor(self):
        """Creates the preprocessing pipeline."""
        numeric_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy=self.num_imputer_strategy)),
            ('scaler', StandardScaler())
        ])

        categorical_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy=self.cat_imputer_strategy)),
            ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False)),
            ('scaler', StandardScaler())
        ])

        self.preprocessor = ColumnTransformer(transformers=[
            ('numeric', numeric_transformer, self.cols_num),
            ('categorical', categorical_transformer, self.cols_cat)
        ])


    def fit_preprocessor(self, 
                         X: pd.DataFrame | np.ndarray,
                         y: pd.Series | np.ndarray | None = None):
        """Fit the preprocessor only once using the entire dataset."""
        # (i) Learn the X-preprocessor
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)
        # Determine categorical vs numeric columns
        self.cols_num = X.dtypes[X.dtypes != object].index.to_list()
        self.cols_cat = X.dtypes[X.dtypes == object].index.to_list()
        if self.preprocessor is None:
            self._create_preprocessor()
        self.preprocessor.fit(X)

        # (ii) Learn the y-normalized if it's supplied
        if y is not None:
            self.target_scaler = StandardScaler()
            y_vec = np.array(y).reshape(-1, 1)
            self.target_scaler.fit(y_vec)


    def fit_model(self, X:pd.DataFrame, y:np.ndarray | pd.Series, **kwargs):
        """Fit an ElasticNet model using the preprocessed data."""
        self.model = ElasticNet(**kwargs)
        if self.preprocessor is None:
            self.model.fit(X, y)
        else:
            # raise Exception("Preprocessor not fitted. Call 'fit_preprocessor' first.")
            X_transformed = self.preprocessor.transform(X)
            self.model.fit(X_transformed, y)


    def fit(self, 
            X: pd.DataFrame | np.ndarray, 
            y: pd.Series | np.ndarray, 
            verbose: bool = False
        ) -> None:
        """Fits the model to the data, normalizing the targets during training."""
        # (i) Initialize the expected attributes
        self.model = None
        self.best_alpha = None
        
        # (ii) Transform y if relevant
        if self.target_scaler is None:
            y_normalized = np.array(y)
        else:
            y_vec = np.array(y).reshape(-1, 1)
            y_normalized = self.target_scaler.transform(y_vec)
            y_normalized = y_normalized.ravel()

        # (iii) Make sure X is a dataframe
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)

        # (iv) Loop over the different alphas
        score_baseline = -np.infty
        direction = di_scoring[self.scoring]
        for alpha in self.alphas:
            vprint(f'best alpha before fit = {str(self.best_alpha)}', verbose)
            vprint(f'model before fit is None = {self.model is None}', verbose)
            # Make a copy of the model
            old_model = deepcopy(self.model)
            di_alpha = {**self.kwargs, **{'alpha':alpha}}
            self.fit_model(X, y_normalized, **di_alpha)
            score_alpha = np.max(direction * self.model.cv_mean_score_)
            
            # If performance is worse than previous model, keep the previous 
            is_better = score_alpha > direction * score_baseline
            vprint(f'alpha = {alpha}, score = {score_alpha:.3f} (metric = {self.model.scoring})', verbose)
            if is_better:
                vprint(f'{score_alpha} > {direction * score_baseline}, updating baseline', verbose)
                score_baseline = score_alpha
                self.best_alpha = alpha
            else:
                vprint('perforning is worse, rolling back', verbose)
                self.model = old_model
            vprint(f'best alpha after fit = {str(self.best_alpha)}', verbose)
            vprint(f'model after fit is None = {self.model is None}\n', verbose)

        # If model is fit, set to true
        self._is_fitted = True


    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Predicts using the fitted model, returning predictions to the original scale.

        Args:
            X (pd.DataFrame): The input features.

        Returns:
            np.ndarray: The predicted values, transformed back to the original scale.
        """
        if not self._is_fitted:
            raise ValueError("ModelPipeline instance is not fitted yet.")
        # Make sure X is a dataframe
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)
        # Call the prediction
        if self.preprocessor is None:
            y_pred_normalized = self.model.predict(X).reshape(-1, 1)
        else:
            X_transformed = self.preprocessor.transform(X)
            y_pred_normalized = self.model.predict(X_transformed).reshape(-1, 1)
        if self.target_scaler is None:
            y_pred = y_pred_normalized.copy()
        else:
            y_pred = self.target_scaler.inverse_transform(y_pred_normalized).ravel()
        y_pred = y_pred.flatten()
        return y_pred
