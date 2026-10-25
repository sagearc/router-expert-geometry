import numpy as np

from src.sequences import pack_shard


def test_raw_uint16_and_numpy_shards_keep_order_and_drop_only_tail(tmp_path):
    pack = pack_shard
    tokens = np.arange(4099, dtype=np.uint16)
    raw, npy = tmp_path / 'raw.npy', tmp_path / 'array.npy'
    tokens.tofile(raw)
    np.save(npy, tokens)
    for path in (raw, npy):
        result, dropped = pack(path)
        assert result.shape == (2, 2048) and dropped == 3
        np.testing.assert_array_equal(result.ravel(), tokens[:4096])
