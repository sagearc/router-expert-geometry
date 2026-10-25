import hashlib
import json

import numpy as np
import pytest

from src.analysis.specialization import counts_routing, load_damage_and_share


def test_normalization_requires_the_routing_file_used_for_ablation(tmp_path):
    routing = tmp_path / 'routing.npz'
    losses = tmp_path / 'losses.npz'
    np.savez(routing, counts=np.array([[[2], [1]]]),
             counts_w=np.array([[[.8], [.2]]]), domains=['dolma_stack'])
    metadata = {'domain': 'dolma_stack',
                'counts_sha256': hashlib.sha256(routing.read_bytes()).hexdigest()}
    np.savez(losses, clean=[1., 2.], r1=[2., 3.], metadata_json=json.dumps(metadata))
    damage, share = load_damage_and_share(losses, routing, 'dolma_stack', 1, 'aux_loss')
    np.testing.assert_array_equal(damage, [1., 1.])
    assert share == pytest.approx(.8)
    with pytest.raises(ValueError, match='domain differs'):
        load_damage_and_share(losses, routing, 'dolma_wiki', 1, 'aux_loss')

    # Same dimensions and domain, but a different run's routing mass.
    np.savez(routing, counts=np.array([[[1], [2]]]),
             counts_w=np.array([[[.3], [.7]]]), domains=['dolma_stack'])
    with pytest.raises(ValueError, match='used to select'):
        load_damage_and_share(losses, routing, 'dolma_stack', 1, 'aux_loss')


def test_retained_unhashed_artifact_still_reduces(tmp_path):
    routing, losses = tmp_path / 'routing.npz', tmp_path / 'losses.npz'
    np.savez(routing, counts=np.array([[[2], [1]]]),
             counts_w=np.array([[[.8], [.2]]]), domains=['dolma_stack'])
    np.savez(losses, clean=[1., 2.], r1=[2., 3.])
    _, share = load_damage_and_share(losses, routing, 'dolma_stack', 1, 'aux_loss')
    assert share == pytest.approx(.8)


def test_specialist_cli_uses_the_captured_code_domain(tmp_path, monkeypatch, capsys):
    import sys

    from src.cli.specialist_ablation import main

    routing = tmp_path / 'routing.npz'
    np.savez(routing, counts=np.array([[[2], [1]]]),
             counts_w=np.array([[[.8], [.2]]]), domains=['dolma_stack'])
    metadata = {'domain': 'dolma_stack', 'sequences_sha256': 'same-sequences',
                'counts_sha256': hashlib.sha256(routing.read_bytes()).hexdigest()}
    aux, loss_free = tmp_path / 'aux.npz', tmp_path / 'loss-free.npz'
    for path, ablated in ((aux, [2., 3.]), (loss_free, [3., 4.])):
        np.savez(path, clean=[1., 2.], r1=ablated, metadata_json=json.dumps(metadata))
    monkeypatch.setattr(sys, 'argv', ['specialist_ablation', '--aux-losses', str(aux),
                                    '--loss-free-losses', str(loss_free), '--aux-routing', str(routing),
                                    '--loss-free-routing', str(routing)])
    main()
    result = json.loads(capsys.readouterr().out)
    assert result['domain'] == 'dolma_stack'
    assert result['raw_ratio'] == 2.
    assert result['sequence_pairing_hash_verified'] is True


def test_ablation_capture_must_match_the_expected_model(tmp_path):
    routing, losses = tmp_path / 'routing.npz', tmp_path / 'losses.npz'
    np.savez(routing, counts=np.array([[[2], [1]]]), counts_w=np.array([[[.8], [.2]]]), domains=['dolma_stack'])
    np.savez(losses, clean=[1., 2.], r1=[2., 3.], metadata_json=json.dumps({'routing': 'loss_free'}))
    assert load_damage_and_share(losses, routing, 'dolma_stack', 1, 'loss_free')[1] == pytest.approx(.8)
    with pytest.raises(ValueError, match='expected a aux_loss ablation capture'):
        load_damage_and_share(losses, routing, 'dolma_stack', 1, 'aux_loss')


def test_counts_routing_reads_the_model_and_checks_the_sequences(tmp_path):
    sequences, other, counts = tmp_path / 'sequences.npz', tmp_path / 'other.npz', tmp_path / 'counts.npz'
    np.savez(sequences, sequences=np.zeros((1, 2)))
    np.savez(other, sequences=np.ones((1, 2)))
    np.savez(counts, counts=np.zeros(1))
    with pytest.raises(ValueError, match='with metadata'):
        counts_routing(counts, sequences)
    digest = hashlib.sha256(sequences.read_bytes()).hexdigest()
    np.savez(counts, metadata_json=json.dumps({'routing': 'loss_free', 'sequences_sha256': digest}))
    assert counts_routing(counts, sequences) == 'loss_free'
    with pytest.raises(ValueError, match='different sequences'):
        counts_routing(counts, other)
    np.savez(counts, metadata_json=json.dumps({'routing': 'kmeans', 'sequences_sha256': digest}))
    with pytest.raises(ValueError, match='unsupported routing'):
        counts_routing(counts, sequences)
