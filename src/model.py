from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np
import pandas as pd

def separar_train_test(df):
    train = df[(df["year"] < 2026)]
    test = df[
        (df["year"] == 2026) &
        (df["month"].isin([1, 2, 3]))
    ]

    features = ["time_index", "lag_1", "lag_2", "num_llamadas"]

    train = train.dropna(subset=features + ["consumo_log", "consumo_minutos"])
    test = test.dropna(subset=features + ["consumo_log", "consumo_minutos"])

    X_train = train[features]
    y_train = train["consumo_log"]

    X_test = test[features]
    y_test = test["consumo_log"]

    return X_train, X_test, y_train, y_test, train, test


def entrenar_modelo(X_train, y_train):
    modelo = LinearRegression()
    modelo.fit(X_train, y_train)
    return modelo


def evaluar_modelo(modelo, X_test, y_test):
    # Predicción en escala log
    y_pred_log = modelo.predict(X_test)

    # Volver a escala real (minutos)
    y_pred = np.expm1(y_pred_log)
    y_test_real = np.expm1(y_test)

    mae = mean_absolute_error(y_test_real, y_pred)
    mse = mean_squared_error(y_test_real, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test_real, y_pred) if len(y_test_real) > 1 else None

    return y_pred, y_test_real, {
        "MAE": mae,
        "MSE": mse,
        "RMSE": rmse,
        "R2": r2
    }


# Predicción futura: abril 2026
def predecir_futuros(modelo, df, meses_objetivo):
    df_base = df.copy()
    resultados = []

    for year_obj, month_obj in meses_objetivo:
        ultimo = df_base.iloc[-1]
        penultimo = df_base.iloc[-2]

        # Si no existe num_llamadas futuro, usamos el último valor conocido
        num_llamadas_estimado = ultimo["num_llamadas"]

        nueva_fila = {
            "year": year_obj,
            "month": month_obj,
            "time_index": ultimo["time_index"] + 1,
            "lag_1": ultimo["consumo_minutos"],
            "lag_2": penultimo["consumo_minutos"],
            "num_llamadas": num_llamadas_estimado
        }

        fila_df = pd.DataFrame([nueva_fila])

        X_nuevo = fila_df[["time_index", "lag_1", "lag_2", "num_llamadas"]]

        # Predicción en log
        prediccion_log = modelo.predict(X_nuevo)[0]

        # Volver a minutos reales
        prediccion_real = np.expm1(prediccion_log)

        fila_df["consumo_minutos"] = prediccion_real
        fila_df["consumo_log"] = prediccion_log

        df_base = pd.concat([df_base, fila_df], ignore_index=True)
        resultados.append(fila_df)

    return pd.concat(resultados, ignore_index=True)