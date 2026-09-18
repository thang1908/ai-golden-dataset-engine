import json
from unittest.mock import patch
import pytest
from project.checkpoint import promote_champion
from project.inference import QueryParser


def test_champion_selection_and_write_failure_preserves_old(tiny, tmp_path):
    model, tokenizer, config, labels = tiny
    output = tmp_path / 'output'
    def promote(accuracy, loss, epoch):
        return promote_champion(output, model, tokenizer, config, labels,
            {'exact_match_accuracy': accuracy, 'validation_loss': loss}, {'epoch': epoch})
    assert promote(.5, 2, 1)
    old_bytes = (output / 'champion/manifest.json').read_bytes()
    with patch('project.checkpoint.save_bundle') as save:
        assert not promote(.4, 1, 2)
        assert not promote(.5, 2, 3)
        save.assert_not_called()
    with patch('project.checkpoint.save_bundle', side_effect=OSError('disk full')):
        with pytest.raises(OSError, match='disk full'):
            promote(.6, 1, 4)
    assert (output / 'champion/manifest.json').read_bytes() == old_bytes
    import os
    replace = os.replace
    def fail_swap(source, target):
        if str(source).endswith('.candidate'):
            raise OSError('swap failed')
        return replace(source, target)
    with patch('project.checkpoint.os.replace', side_effect=fail_swap):
        with pytest.raises(OSError, match='swap failed'):
            promote(.6, 1, 4)
    assert (output / 'champion/manifest.json').read_bytes() == old_bytes
    assert {p.name for p in output.iterdir()} == {'champion'}
    assert promote(.5, 1, 5)  # Loss breaks equal-accuracy ties.
    manifest = json.loads((output / 'champion/manifest.json').read_text())
    assert manifest['training']['epoch'] == 5
    assert {p.name for p in output.iterdir()} == {'champion'}
    assert 'gender' in QueryParser.from_checkpoint(output).parse('người')
    with pytest.raises(ValueError, match='finite'):
        promote(float('nan'), 1, 6)


def test_legacy_migration_tracks_before_pruning(tiny, tmp_path):
    from mlflow import MlflowClient
    from project.checkpoint import save_bundle
    from project.migrate_checkpoint import migrate
    model, tokenizer, config, labels = tiny
    root = tmp_path / 'legacy'
    for epoch, accuracy in [(1, .8), (2, .4)]:
        save_bundle(root / f'epoch-{epoch}', model, tokenizer, config, labels,
                    {'exact_match_accuracy': accuracy, 'validation_loss': 1.0})
        state = root / f'state-{epoch}'
        state.mkdir()
        (state / 'progress.json').write_text(json.dumps({'config': config.__dict__,
                                                        'next_epoch': epoch, 'global_step': epoch}))
    (root / 'best.txt').write_text('epoch-1')
    (root / 'last.txt').write_text('epoch-2')
    uri = f'sqlite:///{tmp_path / "migration.db"}'
    client = MlflowClient(tracking_uri=uri)
    client.create_experiment('migration', artifact_location=(tmp_path / 'artifacts').as_uri())
    run_id = migrate(root, uri, 'migration')
    assert {p.name for p in root.iterdir()} == {'champion'}
    assert len(client.get_metric_history(run_id, 'exact_match_accuracy')) == 2
    assert client.get_run(run_id).info.status == 'FINISHED'
    assert client.get_run(run_id).data.tags['legacy_pruned'] == 'true'
    assert json.loads((root / 'champion/manifest.json').read_text())['training']['epoch'] == 1
    assert 'gender' in QueryParser.from_checkpoint(root).parse('người')
