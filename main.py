from argparse import ArgumentParser
import mlflow
from mlflow.utils import mlflow_tags
from server.RemoteServer import RemoteTracking
import os

class FlowTraining:
    def __init__(self, tracking_uri, experiment_name):
        self.tracking_uri = tracking_uri
        self.experiment_name = experiment_name
        self.remote_server = RemoteTracking(tracking_uri=tracking_uri)
        self.local_experiment_dir = './mlruns'
        self.local_experiment_id = '0'

    def log_tags_and_params(self, remote_run_id):
        run_id = self.get_local_run_id()
        mlflow.set_tracking_uri(self.local_experiment_dir)
        run = mlflow.get_run(run_id = run_id)
        params = run.data.params
        tags = run.data.tags
        print(tags)
        print(params)
        self.remote_server.set_tags(remote_run_id, tags)
        self.remote_server.log_params(remote_run_id, params)

    def get_local_run_id(self):
        files = os.listdir(os.path.join(self.local_experiment_dir, self.local_experiment_id))
        for file in files:
            if not file.endswith('.yaml'):
                return file

    def run(self):
        print('Entry point: Workflow')
        remote_experiment_id = self.remote_server.get_experiment_id(name = self.experiment_name)
        remote_run_id = self.remote_server.get_run_id(remote_experiment_id)

        mlflow.set_tracking_uri(self.tracking_uri)
        mlflow.set_experiment(self.experiment_name)

        with mlflow.start_run(run_id=remote_run_id, nested=False) as active_run:
            print('WOW')
            if not os.path.exists('outputs'):
                os.makedirs('outputs')
            with open('outputs/test.txt', 'w') as f:
                f.write('hello world!')
            mlflow.log_artifact('outputs')

        self.log_tags_and_params(remote_run_id)

if __name__ == '__main__':
    print('Entry_point: main.py')
    parser = ArgumentParser()
    parser.add_argument('--experiment_name', default ='Default', type = str)
    parser.add_argument('--tracking_uri', default = 'http://127.0.0.1:5000', type = str)
    args = parser.parse_args()
    print(args)
    tracking = FlowTraining(experiment_name=args.experiment_name,
                            tracking_uri=args.tracking_uri)
    tracking.run()
