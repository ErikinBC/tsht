"""
https://github.com/Moradnejad/AgeDataset

Let's embed these with Ads basically (make sure to take out "Death Year and "Age of Death")

python3 0_age_dataset.py
"""

# External module
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.feature_extraction.text import CountVectorizer

# Example data
data = pd.DataFrame({
    'x1': ['Canada; USA', 'Canada; Mexico', 'Canada; Bhutan', 'Mexico', 'USA'],
    'x2': ['Red; Blue', 'Green; Blue', 'Yellow; Red', 'Red', 'Green'],
    'x3': [1, 2, 3, 4, 5],
    'x4': [6, 7, 8, 9, 10]
})


enc = CountVectorizer(input='content', 
                decode_error='ignore',
                lowercase=True,
                strip_accents='ascii',
                tokenizer=lambda x: x.split('; '), 
                min_df=2,
                binary=True)

qq=enc.fit_transform(data['x1'])
data.x1
pd.DataFrame(qq.toarray(), columns=enc.get_feature_names_out())


# Custom pipeline for categorical columns
categorical_pipeline = Pipeline([
    ('encoder', CountVectorizer(tokenizer=lambda x: x.split(';'), binary=True)), 
])

# Pipeline for numeric columns
numeric_pipeline = Pipeline([
    ('scaler', StandardScaler())
])

# Combine both pipelines into a ColumnTransformer
preprocessor = ColumnTransformer([
    ('num', numeric_pipeline, ['x3', 'x4']),
    ('cat', categorical_pipeline, ['x1', 'x2'])
])


# Apply the preprocessor
preprocessed_data = preprocessor.fit_transform(data)

print(preprocessed_data)

assert False




################################
# --- (1) PREPARE THE DATA --- #

# Tidy up column names
di_rename = {'Gender':'gender', 
             'Country': 'country', 
             'Birth year': 'birth_year', 
             'Occupation': 'occupation', 
             'Age of death': 'death_age',
             'Death year': 'death_year',
             'Manner of death': 'death_manner'}

# Define the label column
colname_y = 'death_age'
colname_x = ['gender', 'country', 'birth_year', 'occupation', 'death_manner']

# Load the dataset
df_age = pd.read_csv('data/AgeDataset.csv', nrows=10000)
df_age.rename(columns=di_rename, inplace=True)

# Check that missing age of death is always for missing death year
assert df_age.loc[df_age['death_age'].isnull(), 'death_year'].isnull().all()
assert df_age.loc[df_age['death_year'].isnull(), 'death_age'].isnull().all()

# Keep only non-missing rows
df_age = df_age[df_age['death_age'].notnull()].reset_index(drop=True)
print(f'A total of {len(df_age):,} remain')

# Extract the label
yy = df_age[colname_y].copy()
xx = df_age[colname_x].copy()
del df_age

################################
# --- (1) PREPARE THE DATA --- #

# (i) Set up encoder
xx['country'].str.count(';').value_counts(dropna=False)
xx['occupation'].str.count(';').value_counts(dropna=False)
xx['death_manner'].str.count(';').value_counts(dropna=False)
xx.loc[0]

# gender: replace with "other" if not Male/Female
xx['gender'].value_counts(dropna=False)
xx['death_manner'].value_counts(dropna=False)

