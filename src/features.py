import pandas as pd
import numpy as np

def crear_features(mensual):
    df = mensual.copy()

    df["time_index"] = range(1, len(df) + 1)
    df["lag_1"] = df["consumo_minutos"].shift(1)
    df["lag_2"] = df["consumo_minutos"].shift(2)
    df["consumo_log"] = np.log1p(df["consumo_minutos"])

    return df