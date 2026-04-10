import pandas as pd

def cargar_datos(ruta):
    df = pd.read_csv(ruta)
    return df

def limpiar_datos(df):
    df["calldate"] = pd.to_datetime(df["calldate"])
    df = df[df["billsec"] > 0]
    return dfgit 