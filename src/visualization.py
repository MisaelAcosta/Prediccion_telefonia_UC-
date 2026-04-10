import matplotlib.pyplot as plt

def graficar_consumo(df_historico, test, y_pred, futuros=None):
    plt.figure(figsize=(12, 6))

    # Histórico real
    plt.plot(
        df_historico["time_index"],
        df_historico["consumo_minutos"],
        marker="o",
        label="Consumo real histórico"
    )

    # Reales de enero, febrero y marzo 2026
    if len(test) > 0:
        plt.scatter(
            test["time_index"],
            test["consumo_minutos"],
            s=120,
            label="Real 2026 (ene-mar)"
        )

        plt.scatter(
            test["time_index"],
            y_pred,
            s=120,
            label="Predicho 2026 (ene-mar)"
        )

    # Abril futuro
    if futuros is not None and len(futuros) > 0:
        plt.scatter(
            futuros["time_index"],
            futuros["consumo_minutos"],
            s=120,
            label="Abril 2026 predicho"
        )

    plt.title("Consumo mensual de telefonía")
    plt.xlabel("Índice temporal")
    plt.ylabel("Consumo total minutos")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()