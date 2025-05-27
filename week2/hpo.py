import os
import pickle
import click
import mlflow
import numpy as np
from hyperopt import STATUS_OK, Trials, fmin, hp, tpe
from hyperopt.pyll import scope
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import root_mean_squared_error

mlflow.set_tracking_uri("http://127.0.0.1:5000")
mlflow.set_experiment("random-forest-hyperopt")


def load_pickle(filename: str):
    with open(filename, "rb") as f_in:
        return pickle.load(f_in)


@click.command()
@click.option(
    "--data_path",
    default="./output",
    help="Location where the processed NYC taxi trip data was saved"
)
@click.option(
    "--num_trials",
    default=15,
    help="The number of parameter evaluations for the optimizer to explore"
)
def run_optimization(data_path: str, num_trials: int):

    X_train, y_train = load_pickle(os.path.join(data_path, "train.pkl"))
    X_val, y_val = load_pickle(os.path.join(data_path, "val.pkl"))

    def objective(params):
        with mlflow.start_run(nested=True):
            try:
                mlflow.set_tag("model", "RandomForestRegressor")
                mlflow.log_params(params)
                
                rf = RandomForestRegressor(**params)
                rf.fit(X_train, y_train)
                y_pred = rf.predict(X_val)
                rmse = root_mean_squared_error(y_val, y_pred)
                mlflow.log_metric("rmse", rmse)

                return {'loss': rmse, 'status': STATUS_OK}
            except Exception as e:
                mlflow.log_param("error", str(e))
                return {'loss': 999, 'status': STATUS_OK}


    search_space = {
        'max_depth': scope.int(hp.quniform('max_depth', 1, 20, 1)),
        'n_estimators': scope.int(hp.quniform('n_estimators', 10, 50, 1)),
        'min_samples_split': scope.int(hp.quniform('min_samples_split', 2, 10, 1)),
        'min_samples_leaf': scope.int(hp.quniform('min_samples_leaf', 1, 4, 1)),
        'random_state': 42
    }

    rstate = np.random.default_rng(42)  # for reproducible results
    with mlflow.start_run(run_name="hyperopt_tuning") as parent_run:
        trials = Trials()
        best_params = fmin(
            fn=objective,
            space=search_space,
            algo=tpe.suggest,
            max_evals=num_trials,
            trials=trials,
            rstate=rstate
        )
        mlflow.log_param("best_params", best_params)
        print("Best parameters found:", best_params)

        # Log the best parameters and the best RMSE to the parent run
        best_params = trials.best_trial['misc']['vals']
        best_rmse = trials.best_trial['result']['loss']
        mlflow.log_params({f"best_{k}": v[0] for k, v in best_params.items()}) # best_params are lists
        mlflow.log_metric("best_rmse", best_rmse)
        print(f"Best parameters found: {best_params}")
        print(f"Best RMSE: {best_rmse}")


if __name__ == '__main__':
    run_optimization()
