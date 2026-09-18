"""MLflow tracks metadata and metrics, never copies model weights."""
import json
import os
from contextlib import contextmanager
from dataclasses import asdict
from pathlib import Path
import mlflow

PROJECT = Path(__file__).resolve().parent
DEFAULT_TRACKING_URI = f'sqlite:///{PROJECT / "mlflow.db"}'


@contextmanager
def tracking_run(config, signature, output, tracking_uri=None,
                 experiment='person-query-parser', run_name=None):
    uri = tracking_uri or os.getenv('MLFLOW_TRACKING_URI') or DEFAULT_TRACKING_URI
    mlflow.set_tracking_uri(uri)
    existing = mlflow.get_experiment_by_name(experiment)
    if existing is None:
        # Local artifacts are small JSON documents. Remote servers own their store.
        artifact = (PROJECT / 'mlartifacts').as_uri() if uri == DEFAULT_TRACKING_URI else None
        experiment_id = mlflow.create_experiment(experiment, artifact_location=artifact)
    else:
        experiment_id = existing.experiment_id
    with mlflow.start_run(experiment_id=experiment_id, run_name=run_name,
                          log_system_metrics=False) as run:
        parameters = asdict(config)
        weights = parameters.pop('loss_weights')
        parameters.update({f'loss_weight.{key}': value for key, value in weights.items()})
        mlflow.log_params(parameters)
        mlflow.log_params({key: value for key, value in signature.items() if key != 'labels'})
        mlflow.log_dict(asdict(config), 'config.json')
        mlflow.log_dict(signature, 'dataset_signature.json')
        mlflow.set_tags({'checkpoint_policy': 'champion_only',
                         'champion_selection': 'exact_match_accuracy desc, validation_loss asc',
                         'checkpoint_path': str(Path(output).resolve() / 'champion')})
        yield run.info.run_id


def log_epoch(metrics, epoch, step, learning_rate, promoted):
    mlflow.log_metrics({**metrics, 'global_step': step, 'learning_rate': learning_rate,
                        'champion_promoted': int(promoted)}, step=epoch, synchronous=True)
    if promoted:
        mlflow.set_tags({'champion_epoch': str(epoch)})
        mlflow.log_metrics({'champion_exact_match_accuracy': metrics['exact_match_accuracy'],
                            'champion_validation_loss': metrics['validation_loss']},
                           step=epoch, synchronous=True)


def log_champion(path):
    # No log_model/log_artifacts: weights remain exclusively in checkpoints.
    mlflow.log_dict(json.loads((Path(path) / 'manifest.json').read_text()), 'champion.json')
