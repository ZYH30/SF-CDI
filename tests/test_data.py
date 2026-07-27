import numpy as np

from sfcdi.data import (
    SeriesWindowDataset,
    valid_window_anchors,
)


def test_windows_never_cross_split_or_segment_boundaries():
    split_ids = np.array([0] * 8 + [1] * 8 + [2] * 8, dtype=np.int8)
    segments = np.array([0] * 5 + [1] * 11 + [2] * 8, dtype=np.int64)
    anchors = valid_window_anchors(
        split_ids, segments, split=1, sequence_length=3, prediction_length=2
    )
    for anchor in anchors:
        start = anchor - 2
        end = anchor + 2
        assert split_ids[start] == split_ids[end] == 1
        assert segments[start] == segments[end]


def test_window_dataset_uses_future_after_history_only():
    features = np.arange(20, dtype=np.float32)[:, None]
    target = np.arange(20, dtype=np.float32)
    dataset = SeriesWindowDataset(
        features,
        target,
        np.zeros(20, dtype=np.int64),
        anchors=[7],
        sequence_length=4,
        prediction_length=3,
    )
    x, history, future, regime, anchor = dataset[0]
    assert x[:, 0].tolist() == [4.0, 5.0, 6.0, 7.0]
    assert history[:, 0].tolist() == [4.0, 5.0, 6.0, 7.0]
    assert future.tolist() == [8.0, 9.0, 10.0]
    assert int(regime) == 0
    assert int(anchor) == 7
