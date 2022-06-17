---

# NMlab Final - MKSgo

## Application

- See [PPT](https://docs.google.com/presentation/d/1FloBqipfjMyI04nXGqOx4vx5GWF9FisV/edit?usp=sharing&ouid=101804700068760254808&rtpof=true&sd=true)

## Requirements

- mediapipe 0.8.1
- OpenCV 3.4.2 or Later
- Tensorflow 2.3.0 or Later
- MQTT 1.5.1 or Later
- Python 3.6.9 or Later

## How to run

### On Jetson Nano

- Install MQTT Broker
  ```bash
  # Run the following command to upgrade and update your system
  $ sudo apt update && sudo apt upgrade
  # Install the Mosquitto Broker
  $ sudo apt install -y mosquitto mosquitto-clients
  # Make Mosquitto auto start when the Ubuntu boots
  $ sudo systemctl enable mosquitto.service
  # Test the installation, which will return the Mosquitto version that is currently running
  $ mosquitto -v
  # Mosquitto Broker Enable Remote Access (No Authentication)
  $ sudo vim /etc/mosquitto/mosquitto.conf
  # Append following two lines
  listener 1883
  allow_anonymous true
  # Restart Mosquitto for the changes to take effect
  $ sudo systemctl restart mosquitto
  ```
- Install mediapipe & tensorflow packages

  ```bash
  $ python3 -m pip install -r requirements.txt
  ```

- Main program

  ```bash
  $ python3 app.py
  ```

### On NodeMCU ESP8266

- In Arduino IDE
  - Goto `Sketch > Include Library > Add . ZIP library`
  - Select the library to be downloaded.
- Install the Async MQTT Client Library
  - [Link](https://github.com/marvinroger/async-mqtt-client)
- Install the ESPAsync TCP Library
  - [Link](https://github.com/me-no-dev/ESPAsyncTCP)
- Compile and upload `nm_final/nm_final.ino`

### On your device

- run ffplay and mointor MKS status through RTMP protocol

  ```bash
  $ ffplay -fflags nobuffer rtmp://{Jetson IP}/rtmp/live
  ```

### On Grafana dashboard

- open https://grafana.ntuee.org
- Log in with given username and password

## Edit files

`app.py`

- Top control code, wait for ESP8266 MQTT incoming studentID
- Estimate borrow and return hand pose using MediaPipe

`utils/uploadFolder.py`

- Upload pictures to AWS S3 cloud service and run Rekognition custom label model
- Update Google Sheet borrow & reocrd data

`utils/mqtt.py`

- MQTT subscriber

`utils/client_secret.json`

- GCP API key

`utils/cvfpscalc.py`

- This is a module for FPS measurement.

## Reference

- [Kazuhito00/hand-gesture-recognition](https://github.com/Kazuhito00/hand-gesture-recognition-using-mediapipe)
- [MediaPipe](https://mediapipe.dev/)
- [ESP8266 NodeMCU MQTT](https://randomnerdtutorials.com/esp8266-nodemcu-mqtt-publish-bme280-arduino/)
- [Install Mosquitto MQTT Broker on Raspberry Pi](https://randomnerdtutorials.com/how-to-install-mosquitto-broker-on-raspberry-pi/)

## Author

- 王維芯 B08901073
- 莊詠翔 B08901093
- 周柏融 B08901159
- 勞志毅 B08901199
