from argparse import ArgumentParser
import mlflow
from mlflow.utils import mlflow_tags
from server.RemoteServer import RemoteTracking


class FlowTraining:
    def __init__(self, tracking_uri, experiment_name):
        self.tracking_uri = tracking_uri
        self.experiment_name = experiment_name
        self.remote_server = RemoteTracking(tracking_uri=tracking_uri)
        #self.local_experiment_dir = './mlruns'
        #self.local_experiment_id = '0'

    def run(self):
        print('Entry point: Workflow')
        remote_experiment_id = self.remote_server.get_experiment_id(name = self.experiment_name)
        remote_run_id = self.remote_server.get_run_id(remote_experiment_id)

        mlflow.set_tracking_uri(self.tracking_uri)
        mlflow.set_experiment(self.experiment_name)

        print(remote_run_id)

        with mlflow.start_run(run_id = remote_run_id) as active_run:
            git_commit = active_run.data.tags.get(mlflow_tags.MLFLOW_GIT_COMMIT)
            print(git_commit)
            mlflow.log_param("param2", 'Hello World!')

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
