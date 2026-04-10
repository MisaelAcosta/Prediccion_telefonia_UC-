import pandas as pd

def cargar_datos(ruta):
    df = pd.read_csv(ruta)
    return df

def limpiar_datos(df):
    df["calldate"] = pd.to_datetime(df["calldate"], errors="coerce")
    df["billsec"] = pd.to_numeric(df["billsec"], errors="coerce")

    df = df.dropna(subset=["calldate", "billsec"])
    df = df[df["billsec"] > 0]

    return df

def crear_consumo_mensual(df):
    df["year"] = df["calldate"].dt.year
    df["month"] = df["calldate"].dt.month

    # Convertir segundos a minutos
    df["billmin"] = df["billsec"] / 60

    mensual = (
        df.groupby(["year", "month"])
        .agg(
            consumo_minutos=("billmin", "sum"),
            num_llamadas=("uniqueid", "count")
        )
        .reset_index()
    )

    mensual = mensual.sort_values(["year", "month"]).reset_index(drop=True)
    return mensual



