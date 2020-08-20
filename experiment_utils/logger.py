import mlflow
import mlflow.pytorch
import mlflow.exceptions
from .server import RemoteTracking
import os


class MlFlowLogger:
    def __init__(self,
                experiment_name = 'Default',
                tracking_uri = 'http://127.0.0.1:5000'):

        self.tracking_uri = tracking_uri
        self.experiment_name = experiment_name
        self.logger = RemoteTracking(tracking_uri)

        self.local_experiment_dir = './mlruns'
        self.local_experiment_id = '0'

    def get_local_run_id(self):
        files = os.listdir((os.path.join(self.local_experiment_dir, self.local_experiment_id)))
        print(files)
        for file in files:
            if not file.endswith('.yaml'):
                print(file)
                return file

    def log_tags(self):
        run_id = self.get_local_run_id()
        mlflow.set_tracking_uri(self.local_experiment_dir)
        run = mlflow.get_run(run_id = run_id)
        tags = run.data.tags
        self.logger.set_tags(self.run_id, tags)

    def start_session(self):
        experiment_id = self.logger.get_experiment_id(self.experiment_name)
        self.run_id = self.logger.get_run_id(experiment_id)

        mlflow.set_tracking_uri(self.tracking_uri)
        mlflow.set_experiment(self.experiment_name)

        experiment = mlflow.start_run(run_id=self.run_id, nested=False)

    def log_params(self, params):
        self.logger.log_params(self.run_id, params)

    def log_metrics(self, metrics, step=None):
        self.logger.log_metrics(self.run_id, metrics, step)

    def log_model(self, model, path):
        mlflow.pytorch.log_model(model, path)

    def log_artifacts(self, dir, artifact_path=None):
        self.logger.log_artifacts(self.run_id, dir, artifact_path)

    def end_session(self):
        mlflow.end_run()
        self.log_tags()
