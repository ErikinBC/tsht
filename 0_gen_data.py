"""
Create the datasets for regression testing

python3 0_gen_data.py --remove_data
"""

import argparse
from parameters import dir_data

def main():
    """Generates the two different housing datasets"""
    # External
    import os
    import pandas as pd
    from sklearn.datasets import fetch_california_housing
    from kaggle.api.kaggle_api_extended import KaggleApi
    # Internal
    from parameters import path_sberbank, path_california
    from utils import makeifnot, unzip_folder, remove_path
    makeifnot(dir_data)

    ################################
    # --- (1) SBERBANK HOUSING --- #

    # Initialize API
    api = KaggleApi()
    api.authenticate()

    # Download the dataset
    competition = 'sberbank-russian-housing-market'
    api.competition_download_files(competition, path=dir_data, quiet=False)

    # Unzip as long as there are zip files
    still_zip = True
    while still_zip:
        unzip_folder(dir_data)
        still_zip = 'zip' in [z.split('.')[-1] for z in os.listdir(dir_data)]

    # Rename the train and delete the others
    remove_files = [z for z in os.listdir(dir_data) if z != 'train.csv']
    for file in remove_files:
        remove_path(os.path.join(dir_data, file))
    path_train_old = os.path.join(dir_data, 'train.csv')
    assert os.path.exists(path_train_old), 'expected train.csv to be left over from sberbank extract'
    os.rename(path_train_old, path_sberbank)


    ##################################
    # --- (2) CALIFORNIA HOUSING --- #

    # Load the data
    data = fetch_california_housing()
    housing = pd.DataFrame(data.data, columns=data.feature_names)
    housing.insert(0, 'price', data.target)

    # Save for later
    housing.to_csv(path_california, index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Generate the housing price datasets')
    parser.add_argument('--remove_data', action='store_true', help='Whether the data folder should be deleted beforehand')
    args = parser.parse_args()
    if args.remove_data:
        print('Removing data')
        import shutil
        shutil.rmtree(dir_data)

    # Run the scripts
    main()

    print('~~~ End of 0_gen_data.py ~~~')

