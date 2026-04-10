from preprocessing import cargar_datos, limpiar_datos, crear_consumo_mensual
from features import crear_features
from model import separar_train_test, entrenar_modelo, evaluar_modelo, predecir_futuros
from visualization import graficar_consumo

def main():
    ruta = "data/alt_simple.csv"

    # 1. Cargar y limpiar datos
    df = cargar_datos(ruta)
    df = limpiar_datos(df)

    print("Datos cargados correctamente")
    print(df.head())
    print("\nColumnas del archivo:")
    print(df.columns)
    print("\nCantidad de registros:")
    print(len(df))

    # 2. Crear tabla mensual
    mensual = crear_consumo_mensual(df)
    print("\nConsumo mensual:")
    print(mensual.head(15))

    # 3. Crear features
    mensual_features = crear_features(mensual)
    print("\nTabla con features:")
    print(mensual_features.head(15))

    # 4. Separar train y test
    X_train, X_test, y_train, y_test, train, test = separar_train_test(mensual_features)

    print("\nTamaño train:", len(X_train))
    print("Tamaño test:", len(X_test))

    if len(X_test) == 0:
        print("\nNo hay datos suficientes para evaluar.")
        return

    # 5. Entrenar
    modelo = entrenar_modelo(X_train, y_train)

    # 6. Evaluar enero, febrero y marzo 2026
    y_pred, y_test_real, metricas = evaluar_modelo(modelo, X_test, y_test)

    print("\nComparación real vs predicho (enero-marzo 2026)")
    for i in range(len(test)):
        print(
            f"{int(test.iloc[i]['year'])}-{int(test.iloc[i]['month']):02d} | "
            f"Real: {y_test_real.iloc[i]} | "
            f"Predicho: {y_pred[i]}"
        )

    print("\nMétricas:")
    for k, v in metricas.items():
        print(f"{k}: {v}")

    # 7. Coeficientes del modelo
    print("\nCoeficientes del modelo:")
    for nombre, coef in zip(X_train.columns, modelo.coef_):
        print(f"{nombre}: {coef}")
    print("Intercepto:", modelo.intercept_)

    # 8. Solo abril 2026 como predicción futura
    meses_objetivo = [(2026, 4)]
    futuros = predecir_futuros(modelo, mensual_features, meses_objetivo)

    print("\nPredicción futura:")
    print(futuros[["year", "month", "consumo_minutos"]])

    # 9. Gráfico
    graficar_consumo(
        mensual_features.dropna(subset=["lag_1", "lag_2"]),
        test,
        y_pred,
        futuros
    )

if __name__ == "__main__":
    main()