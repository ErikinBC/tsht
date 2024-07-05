"""
Script to create clean up versions of datasets that can be read as a CSV and fed into an sklearn encoder

python3 0_datasets.py
"""

# External module
import os
import requests
import numpy as np
import pandas as pd
# Internal
from utils import try_command_line
from parameters import dir_data, \
                        path_url_age, path_raw_age, path_clean_age, \
                        di_rename_age, colname_y_age, colname_x_age


###########################
# --- (1) AGE DATASET --- #

# (i) Download the dataset if it doesn't exist (https://github.com/Moradnejad/AgeDataset)
# Label will be "Age of Death", and will allow the cause of death to be known
if not os.path.exists(path_raw_age):
    print(f'Cannot find {path_raw_age}, downloading')
    # Download the data and save as the rar
    response = requests.get(url=path_url_age)
    assert response.status_code == 200, 'download didnt work!'
    path_rar = os.path.join(dir_data, path_url_age.split('/')[-1])
    with open(path_rar, 'wb') as f:
        f.write(response.content)
    del response
    # Unzip it was unrar
    try_command_line('unrar')
    os.system(f'unrar x -r {path_rar} {dir_data}')
    # Delete the old file
    os.remove(path_rar)
    assert os.path.exists(path_raw_age), 'expected the AgeDataset file to exist!'        
else:
    print(f'File {path_raw_age} already exists')


# (ii) Load and rename
df_age = pd.read_csv(path_raw_age)
df_age.rename(columns=di_rename_age, inplace=True)

# (iii) Check that missing age of death is always for missing death year
assert df_age.loc[df_age['death_age'].isnull(), 'death_year'].isnull().all()
assert df_age.loc[df_age['death_year'].isnull(), 'death_age'].isnull().all()

# (iv) Keep only non-missing rows
n_age_before = df_age.shape[0]
df_age = df_age[df_age['death_age'].notnull()].reset_index(drop=True)
print(f'After removing null values there are {len(df_age):,} rows for the Age dataset (from {n_age_before:,})')

# (v) Drop non-relevant columns
df_age = df_age[[colname_y_age] + colname_x_age]
df_age.rename(columns = {colname_y_age:'y'}, inplace=True)

# (vi) Log-scale the response (more likely to be normal)
df_age['y'] = np.log(df_age['y'] + 1)

# (vii) Save for later
df_age.to_csv(path_clean_age, index=False)

