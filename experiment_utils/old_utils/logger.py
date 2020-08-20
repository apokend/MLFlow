import os
import mlflow
from .server import RemoteTracking
import mlflow.pytorch
from mlflow.utils import mlflow_tags

class MlFlowLogger:
    def __init__(self,
                 experiment_name = 'Default',
                 tracking_uri = 'http://127.0.0.1:5000'):

        self.tracking_uri = tracking_uri
        self.experiment_name = experiment_name
        self.remote_server =  RemoteTracking(tracking_uri=tracking_uri)
        self.local_experiment_dir = '/home/alexander/airflow/mlruns'
        self.local_experiment_id = '0'

        remote_experiment_id = self.remote_server.get_experiment_id(name = experiment_name)
        self.remote_run_id = self.remote_server.get_run_id(remote_experiment_id)
        print(self.remote_run_id)

        mlflow.set_tracking_uri(self.tracking_uri)
        mlflow.set_experiment(self.experiment_name)

    def get_local_run_id(self):
        files = os.listdir((os.path.join(self.local_experiment_dir, self.local_experiment_id)))
        for file in files:
            if not file.endswith('.yaml'):
                return file

    def log_tags(self, remote_run_id):
        run_id = self.get_local_run_id()
        mlflow.set_tracking_uri(self.local_experiment_dir)
        run = mlflow.get_run(run_id = run_id)
        tags = run.data.tags
        self.remote_server.set_tags(remote_run_id, tags)

    def log_params(self, params):
        self.remote_server.log_hyperparams(self.remote_run_id, params)

    def log_metrics(self, metrics, step = None):
        self.remote_server.log_metrics(self.remote_run_id, metrics, step)

    def finish(self, model, path = 'model'):
        with mlflow.start_run(run_id=self.remote_run_id, nested=False) as active_run:
            mlflow.pytorch.log_model(model, "models")

        self.log_tags(self.remote_run_id)

        #     print('WOW')
        #     if not os.path.exists('outputs'):
        #         os.makedirs('outputs')
        #     with open('outputs/test.txt', 'w') as f:
        #         f.write('hello world!')
        #     mlflow.log_artifact('outputs')
        # print('Success')
        # self.log_tags(self.remote_run_id)


        # if not os.path.exists('outputs'):
        #     os.makedirs('outputs')
        # with open('outputs/test.txt', 'w') as f:
        #     f.write('hello world!')
        # mlflow.log_artifact('outputs')
        #
        # self.log_targs(self.remote_run_id)
        # mlflow.pytorch.log_model(model, artifact_path=path)
        # mlflow.end_run()
        #
        # with mlflow.start_run(run_id=self.remote_run_id, nested=False) as active_run:
        #     print('WOW')
        #     if not os.path.exists('outputs'):
        #         os.makedirs('outputs')
        #     with open('outputs/test.txt', 'w') as f:
        #         f.write('hello world!')
        #     mlflow.log_artifact('outputs')
        # self.log_tags(self.remote_run_id)
        # mlflow.pytorch.log_model(model, artifact_path=path)
