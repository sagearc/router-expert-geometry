import json
import sys

import numpy as np
import pytest

from src.cli.kmeans_errors import main


def test_combined_kmeans_capture_checks_reference_identity_and_token_order(tmp_path, monkeypatch, capsys):
    km, ref = tmp_path / 'km.npz', tmp_path / 'ref.npz'
    common = {'targets': np.array([[1, 2, 3], [4, 5, 6]]),
              'domain_idx': np.array([0, 1]), 'domain_names': np.array(['a', 'b']),
              'nll': np.ones((2, 3)), 'margin6': np.arange(54).reshape(9, 2, 3)}
    metadata = {'sequences_sha256': 'same-input', 'routing': 'kmeans', 'checkpoint': 'km-21k'}
    reference = {**metadata, 'routing': 'loss_free', 'checkpoint': 'lf-21k'}
    np.savez(km, **common, metadata_json=json.dumps(metadata))
    monkeypatch.setattr(sys, 'argv', ['kmeans_errors', '--kmeans', str(km),
                                    '--reference', str(ref)])
    for override, error in [({'routing': 'kmeans'}, 'expected loss_free'),
                            ({'sequences_sha256': 'different'}, 'hashes must agree')]:
        np.savez(ref, **common, metadata_json=json.dumps({**reference, **override}))
        with pytest.raises(SystemExit):
            main()
        assert error in capsys.readouterr().err
    np.savez(ref, **{**common, 'targets': common['targets'][::-1]},
             metadata_json=json.dumps(reference))
    with pytest.raises(SystemExit):
        main()
    assert 'disagree on targets' in capsys.readouterr().err
    np.savez(ref, **common, metadata_json=json.dumps(reference))
    main()
    result = json.loads(capsys.readouterr().out)
    assert result['sequence_hash_verified'] is True
    assert result['mean_nll_gap'] == 0


def test_unhashed_combined_capture_reports_reference_pairing(tmp_path, monkeypatch, capsys):
    km, ref = tmp_path / 'km.npz', tmp_path / 'ref.npz'
    common = {'targets': np.arange(6).reshape(2, 3), 'nll': np.ones((2, 3)),
              'domain_idx': [0, 0], 'domain_names': ['a']}
    np.savez(km, **common, margin6=np.arange(54).reshape(9, 2, 3))
    np.savez(ref, **common)
    monkeypatch.setattr(sys, 'argv', ['kmeans_errors', '--kmeans', str(km),
                                    '--reference', str(ref)])
    main()
    result = json.loads(capsys.readouterr().out)
    assert result['sequence_hash_verified'] is False
    assert result['margin_alignment'].startswith('shared K-means capture')
    np.savez(km, **common)
    with pytest.raises(SystemExit):
        main()
    assert 'must contain both nll and margin6' in capsys.readouterr().err
