import joblib
import time
import pandas as pd
import numpy as np

# Caminho do modelo treinado (ajuste conforme seu arquivo)
MODEL_PATH = 'flood_risk_model.joblib'

# Mapeamento dos níveis de risco
RISK_LEVELS = {0: "Normal", 1: "Alerta", 2: "Inundação"}

def load_model(path):
    """Carrega o modelo de ML do arquivo joblib."""
    try:
        model = joblib.load(path)
        print(f"Modelo carregado com sucesso de {path}")
        return model
    except Exception as e:
        print(f"Erro ao carregar o modelo: {e}")
        return None

def predict_risk(model, distance):
    """Usa o modelo para prever o risco com base na distância."""
    try:
        input_data = pd.DataFrame([[distance]], columns=["distancia_cm"])
        prediction = model.predict(input_data)
        return prediction[0]
    except Exception as e:
        print(f"Erro durante a predição: {e}")
        return None

def main():
    model = load_model(MODEL_PATH)
    if model is None:
        print("Encerrando devido ao erro no carregamento do modelo.")
        return

    print("\n--- Simulação do Monitor de Risco de Enchente ---")
    print("Gerando leituras simuladas do sensor...")

    while True:
        # Simular uma leitura do sensor entre 5 e 100 cm
        simulated_distance = np.random.uniform(5, 100)

        print(f"Distância simulada: {simulated_distance:.2f} cm")
        risk_code = predict_risk(model, simulated_distance)

        if risk_code is not None:
            risk_label = RISK_LEVELS.get(risk_code, "Desconhecido")
            print(f">>> Nível de Risco Previsto: {risk_label} (Código: {risk_code})")
        else:
            print("Não foi possível fazer a predição.")

        time.sleep(2)  # Espera 2 segundos entre leituras

if __name__ == "__main__":
    main()
