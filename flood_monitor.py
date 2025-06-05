import joblib
import pandas as pd
import paho.mqtt.client as mqtt

MODEL_PATH = 'flood_risk_model.joblib'
RISK_LEVELS = {0: "Normal", 1: "Alerta", 2: "Inundação"}

def load_model(path):
    try:
        model = joblib.load(path)
        print(f"Modelo carregado de {path}")
        return model
    except Exception as e:
        print(f"Erro ao carregar modelo: {e}")
        return None

def predict_risk(model, distance):
    try:
        input_data = pd.DataFrame([[distance]], columns=["distancia_cm"])
        prediction = model.predict(input_data)
        return prediction[0]
    except Exception as e:
        print(f"Erro na predição: {e}")
        return None

def on_message(client, userdata, msg):
    model = userdata['model']
    try:
        distance = float(msg.payload.decode())
        print(f"Distância recebida: {distance:.2f} cm")
        risk_code = predict_risk(model, distance)
        risk_label = RISK_LEVELS.get(risk_code, "Desconhecido")
        print(f">>> Nível de Risco Previsto: {risk_label} (Código: {risk_code})\n")
    except ValueError:
        print("Mensagem inválida recebida.")

def main():
    model = load_model(MODEL_PATH)
    if not model:
        return

    client = mqtt.Client(userdata={'model': model})
    client.on_message = on_message
    client.connect("broker.hivemq.com", 1883, 60)
    client.subscribe("esp32/flood/distance")

    print("Conectado ao broker MQTT. Aguardando dados do ESP32 (Wokwi)...")
    client.loop_forever()

if __name__ == "__main__":
    main()
