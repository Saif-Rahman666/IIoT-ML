import json
import paho.mqtt.client as mqtt
from inference_pipeline import predict_step

BROKER = "localhost"      # or IP of broker / Raspberry Pi
PORT = 1883
TOPIC = "engine/+/sensors"

AE_SENSORS = ["s2", "s3", "s4", "s7", "s11", "s12", "s15"]





def on_connect(client, userdata, flags, rc):
    print("✅ MQTT connected:", rc)
    client.subscribe(TOPIC)

def on_message(client, userdata, msg):
    try:
        data = json.loads(msg.payload.decode())
        unit = int(data["unit"])

        sensors = {}
        for s in AE_SENSORS:
            sensors[s] = float(data[s])

        result = predict_step(unit, sensors)
        print("📡 Prediction:", result)
        
        PRED_TOPIC = f"engine/{unit}/prediction"
        client.publish(PRED_TOPIC, json.dumps(result))

    except Exception as e:
        print("❌ Error:", e)
        

        


client = mqtt.Client()
client.on_connect = on_connect
client.on_message = on_message

client.connect(BROKER, PORT, 60)
client.loop_forever()
