import threading
from queue import Queue
from engine.trainer import Trainer
from core.logger import create_logger

class GpuManager:
    def __init__(self, gpu_id: int):
        self.gpu_id = gpu_id
        self.logger = create_logger(f"gpu_{gpu_id}")
        self.queue = Queue()
        self._lock = threading.Lock()
        self._active_jobs = set()

        # Avvio thread che processa i job
        self.worker_thread = threading.Thread(target=self._process_jobs, daemon=True)
        self.worker_thread.start()

    def run_job(self, job_data: dict):
        """
        Aggiunge un job alla coda associata a questa GPU.
        """
        training_id = job_data.get("training_id", "unknown")
        self.logger.info(f"Received job data: {job_data}")
        with self._lock:
            self._active_jobs.add(training_id)
        self.queue.put(job_data)

    def _process_jobs(self):
        """
        Thread worker che processa continuamente i job nella coda.
        """
        while True:
            job = self.queue.get()
            try:
                self._handle_job(job)
            except Exception as e:
                self.logger.error(f"[ERROR] Job {job.get('training_id')}: {e}")
            finally:
                self.queue.task_done()

    def _handle_job(self, job_data: dict):
        """
        Esegue un singolo job, invocando il Trainer.
        """
        training_id = job_data.get("training_id", "unknown")
        worker_type = job_data.get("worker_type", "").lower()
        self.logger.info(f"[START] Job {training_id} ({worker_type})")

        try:
            trainer = Trainer()

            # Estrai config e metadata opzionali
            config_yaml = job_data.get("config_yaml")
            metadata = job_data.get("dataset", {}).get("metadata", None)
            dataset_data = job_data.get("dataset", {})

            # Esegui in base al tipo di job
            if worker_type == "anomalibdetection":
                trainer.run_anomalib_job(dataset_data, config_yaml or [])
            elif worker_type == "yolodetection":
                trainer.run_yolo_job(dataset_data, config_yaml or [], metadata)
            else:
                self.logger.warning(f"Tipo worker sconosciuto: {worker_type}")
                return

            self.logger.info(f"[DONE] Job {training_id}")

        except Exception as e:
            self.logger.error(f"[ERROR] Job {training_id}: {e}")

        finally:
            with self._lock:
                self._active_jobs.discard(training_id)

    def get_status(self) -> dict:
        """
        Ritorna lo stato attuale: numero job in coda e in esecuzione.
        """
        with self._lock:
            return {
                "queued": self.queue.qsize(),
                "running": len(self._active_jobs)
            }

# TODO: in futuro, permettere più GPU o più worker thread (1 per GPU o per job)
# TODO: tracciare lo stato e progresso del job (es: % completamento, errore, done)
