import json
import os
from pathlib import Path
import subprocess
import sys
from mlflow import MlflowClient
from project.inference import QueryParser


def test_training_keeps_only_champion_and_tracks_all_epochs(tiny, tmp_path):
    _, _, config, _ = tiny
    config_path = tmp_path / 'config.json'
    config.save(config_path)
    output = tmp_path / 'run'
    root = Path(__file__).resolve().parents[2]
    uri = f'sqlite:///{tmp_path / "tracking.db"}'
    client = MlflowClient(tracking_uri=uri)
    experiment_id = client.create_experiment('test-parser', artifact_location=(tmp_path / 'artifacts').as_uri())
    command = [sys.executable, '-m', 'project.train', '--train', 'project/train.json',
        '--validation', 'project/validation.json', '--output', str(output), '--config', str(config_path),
        '--tracking-uri', uri, '--experiment', 'test-parser']
    environment = {**os.environ, 'ACCELERATE_USE_CPU': 'true', 'OMP_NUM_THREADS': '1'}
    run = subprocess.run(command, cwd=root, env=environment, capture_output=True, text=True, timeout=120)
    assert run.returncode == 0, run.stdout + run.stderr
    assert {p.name for p in output.iterdir()} == {'champion'}
    assert len(list(output.rglob('*.safetensors'))) == 1
    assert 'gender' in QueryParser.from_checkpoint(output).parse('người áo trắng')
    manifest = json.loads((output / 'champion/manifest.json').read_text())
    run_id = manifest['training']['mlflow_run_id']
    recorded = client.get_run(run_id)
    assert recorded.info.status == 'FINISHED'
    assert recorded.info.experiment_id == experiment_id
    assert recorded.data.params['epochs'] == '2'
    assert recorded.data.tags['checkpoint_policy'] == 'champion_only'
    for key in ['validation_loss', 'exact_match_accuracy', 'gender_accuracy',
                'attribute_micro_f1', 'train_total_loss', 'train_hair_loss', 'validation_accessory_loss']:
        history = client.get_metric_history(run_id, key)
        assert [point.step for point in sorted(history, key=lambda m: m.step)] == [1, 2]
    rows = [json.loads(line) for line in run.stdout.splitlines() if line.startswith('{')]
    expected = max(rows, key=lambda row: (row['exact_match_accuracy'], -row['validation_loss']))
    assert manifest['training']['epoch'] == expected['epoch']
    assert manifest['metrics']['validation_loss'] == expected['validation_loss']
    assert client.get_metric_history(run_id, 'global_step')[-1].value == 6
    artifacts = client.list_artifacts(run_id)
    assert {a.path for a in artifacts} == {'config.json', 'dataset_signature.json', 'champion.json'}
    assert all(not a.is_dir for a in artifacts)  # No weight archives in MLflow.
    rejected = subprocess.run(command, cwd=root, env=environment, capture_output=True, text=True, timeout=120)
    assert rejected.returncode != 0
    assert 'Output is not empty' in rejected.stderr
