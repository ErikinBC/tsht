"""
TEST TSHT ON RWD

python3 4_rwd.py
"""

# External modules
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
# Internal modules
from parameters import path_clean_age, num_cat_min
from utils import FrameCountVectorizer, find_string_columns, find_numeric_columns


#################################
# --- (0) RE-USABLE ENCODER --- #

# Set up encoder for "token1; token2; ..."
semicolon_string_encoder = FrameCountVectorizer(input='content', 
                decode_error='ignore',
                token_pattern = None,
                lowercase=True,
                strip_accents='ascii',
                tokenizer=lambda x: x.split('; '), 
                min_df=num_cat_min,
                binary=True)

# String encoder with normalization
categorical_string_pipeline = Pipeline([
    ('encoder', semicolon_string_encoder), 
    ('scaler', StandardScaler())
])

# Standard normalizations
numeric_pipeline = Pipeline([
    ('scaler', StandardScaler())
])



###########################
# --- (1) AGE DATASET --- #

# (i) Load the dataset
df_age = pd.read_csv(path_clean_age)
y_age = df_age['y'].copy()
df_age.drop(columns='y', inplace=True)
cols_str_age = find_string_columns(df_age)
cols_num_age = find_numeric_columns(df_age) 
print(f'Found {len(cols_str_age)} string columns for age = {cols_str_age}')
print(f'Found {len(cols_num_age)} string columns for age = {cols_num_age}')


# Set up the preprocessor the age dataset
preprocessor_age = ColumnTransformer([
    ('num', numeric_pipeline, cols_num_age),
    ('cat', categorical_string_pipeline, cols_str_age)
])





