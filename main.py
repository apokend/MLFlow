from argparse import ArgumentParser
import mlflow

def workflow(args):
    with mlflow.start_run() as active_run:
        git_commit = active_run.data.tags.get(mlflow_tags.MLFLOW_GIT_COMMIT)
        print(git_commit)


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('-name', '--experiment_name', default ='Default', type = str)
    args = parser.parse_args()
    workflow(args)

    #print(args.experiment_name)
