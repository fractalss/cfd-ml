import numpy as np
import pandas as pd
from sklearn.metrics import root_mean_squared_error


def compute_rmse(merged_df, field_variable):
    y_true = merged_df[f'{field_variable}_CFD']
    y_pred = merged_df[f'{field_variable}_ROM']

    rmse = root_mean_squared_error(y_true, y_pred)
    return rmse, merged_df