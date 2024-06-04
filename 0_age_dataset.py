"""
https://github.com/Moradnejad/AgeDataset

Let's embed these with Ads basically (make sure to take out "Death Year and "Age of Death")
"""

import pandas as pd
# Load the dataset
df_age = pd.read_csv('data/AgeDataset.csv', nrows=100)
# Extract the features
colname_death = df_age.columns[df_age.columns.str.contains('death', case=False)].to_list()
colname_death