from flask import Flask, request, jsonify
from engine.gpu_manager import GpuManager
from core.logger import create_logger
from core.config_loader import ConfigLoader

app = Flask(__name__)
logger = create_logger("webserver")

# Inizializza GpuManager sulla GPU 0
gpu_manager = GpuManager(gpu_id=0)
cfg = ConfigLoader()

@app.route("/train", methods=["POST"])
def train():
    data = request.get_json()

    # Verifica presenza campi obbligatori
    required_fields = ["training_id", "worker_type", "dataset"]
    missing = [f for f in required_fields if f not in data]
    if missing:
        return jsonify({
            "status": "error",
            "message": f"Missing fields: {', '.join(missing)}"
        }), 400

    logger.info(f"Received training request: {data}")

    try:
        # Lancia job
        gpu_manager.run_job(data)

        return jsonify({
            "status": "queued",
            "message": "Training job queued",
            "training_id": data["training_id"]
        }), 200

    except Exception as e:
        logger.error(f"Errore durante l'accodamento del job: {e}")
        return jsonify({
            "status": "error",
            "message": str(e)
        }), 500


@app.route("/status", methods=["GET"])
def status():
    """
    Ritorna lo stato della coda per la GPU.
    """
    return jsonify(gpu_manager.get_status())

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
