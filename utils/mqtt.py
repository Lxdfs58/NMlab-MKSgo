import json
import argparse
import paho.mqtt.client as mqtt

def on_message(client, obj, msg):
    print((msg.payload).decode('utf-8'))
    # print(f"TOPIC:{msg.topic}, VALUE:{msg.payload}")

def run_mqtt():
    # Establish connection to mqtt broker
    client = mqtt.Client()
    client.on_message = on_message
    client.connect(host="localhost", port=1883)
    client.subscribe('ID', 0)

    try:
        client.loop_forever()
    except KeyboardInterrupt as e:
        pass

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ip",
                        default="localhost",
                        help="service ip of MQTT broker")
    parser.add_argument("--port",
                        default=1883,
                        type=int,
                        help="service port of MQTT broker")
    args = vars(parser.parse_args())
    run_mqtt()

