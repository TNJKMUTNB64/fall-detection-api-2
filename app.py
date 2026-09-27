from flask import Flask, request, jsonify
import numpy as np
import tensorflow as tf
from threading import Lock
from datetime import datetime, timezone

app = Flask(__name__)

# =========================
# LOAD TFLITE MODEL
# =========================

MODEL_PATH = "fall_detection_model.tflite"

interpreter = tf.lite.Interpreter(
    model_path=MODEL_PATH
)

interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

latest_result = "No Fall"
TIME_STEPS = 50
NUM_FEATURES = 3
THRESHOLD = 0.5

# เก็บผลลัพธ์แยกตามอุปกรณ์
device_results = {}

# ป้องกันการเข้าถึง Model พร้อมกัน
model_lock = Lock()

# ป้องกันการอ่าน/เขียน Dictionary พร้อมกัน
results_lock = Lock()

@app.route('/predict', methods=['POST'])
def predict():
    try:
        req = request.get_json(silent=True)

        if not isinstance(req, dict):
            return jsonify({
                "error": "Invalid JSON"
            }), 400

        device_id = req.get("device_id")
        data = req.get("inputs")

        if not isinstance(device_id, str) or not device_id.strip():
            return jsonify({
                "error": "Missing device_id"
            }), 400

        if data is None:
            return jsonify({
                "error": "Missing inputs"
            }), 400

        input_array = np.array(
            data,
            dtype=np.float32
        )

        if input_array.shape != (TIME_STEPS, NUM_FEATURES):
            return jsonify({
                "error": "Invalid input shape",
                "expected": [50, 3],
                "received": list(input_array.shape)
            }), 400

        if not np.isfinite(input_array).all():
            return jsonify({
                "error": "Inputs contain NaN or Infinity"
            }), 400

        # เพิ่ม Batch Dimension
        input_array = np.expand_dims(
            input_array,
            axis=0
        )

        # Predict โดยใช้ Model เดียวกัน
        
        with model_lock:

            interpreter.set_tensor(
            input_details[0]['index'],
            input_array.astype(np.float32)
        )

        interpreter.invoke()

        prediction = interpreter.get_tensor(
            output_details[0]['index']
        )


        fall_probability = float(prediction[0][0])

        result = (
            "Fall"
            if fall_probability > THRESHOLD
            else "No Fall"
        )

        # บันทึกผลแยกตาม Device
        with results_lock:
            device_results[device_id] = {
                "result": result,
                "probability": fall_probability,
                "updated_at": datetime.now(
                    timezone.utc
                ).isoformat()
            }

        print(
            f"Device: {device_id} | "
            f"Probability: {fall_probability:.6f} | "
            f"Result: {result}"
        )

        return jsonify({
            "device_id": device_id,
            "result": result,
            "probability": fall_probability
        })

    except Exception as e:
        print("ERROR:", str(e))

        return jsonify({
            "error": str(e)
        }), 500


# ดูผลของ ESP32 ทุกเครื่อง
@app.route('/status', methods=['GET'])
def status():
    with results_lock:
        return jsonify(dict(device_results))


# ดูผลเฉพาะ ESP32 เครื่องที่ต้องการ
@app.route('/status/<device_id>', methods=['GET'])
def device_status(device_id):

    with results_lock:
        result = device_results.get(device_id)

        if result is None:
            return jsonify({
                "error": "Device not found"
            }), 404

        return jsonify({
            "device_id": device_id,
            **result
        })


if __name__ == '__main__':
    app.run(
        host='0.0.0.0',
        port=5000,
        threaded=True
    )