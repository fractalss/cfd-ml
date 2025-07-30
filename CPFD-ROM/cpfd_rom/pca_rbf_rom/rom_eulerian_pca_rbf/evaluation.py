import numpy as np
import pandas as pd
from sklearn.metrics import root_mean_squared_error

def compute_rmse(rom_df, cfd_df, field_variable):
    merged_df = pd.merge(rom_df, cfd_df, on=['x', 'y', 'z'], suffixes=('_ROM', '_CFD'))
    rmse = root_mean_squared_error(merged_df[f'{field_variable}_CFD'], merged_df[f'{field_variable}_ROM'])
    return rmse, merged_df