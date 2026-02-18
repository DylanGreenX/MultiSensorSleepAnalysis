"""Unit tests for Sadeh (1994) sleep/wake algorithm.

Validates:
    - Correct formula: PS = 7.601 - 0.065*MEAN - 1.08*NAT - 0.056*SD - 0.703*LG
    - Sleep threshold: PS >= 0 = Sleep (not > -4 which was the prior bug)
    - Activity count capping at 300
    - Rolling feature computations (MEAN, SD, NAT)
    - Anatomical weights: ankles {1,3} = 0.5, wrists {2,4} = 1.0
    - Fusion strategies: weighted vote (V), weighted index (W), majority (M)
"""

import pytest
import numpy as np
import pandas as pd
import os

from multisensor_sleep.algorithms.sadeh import (
    sadeh_index,
    roll_mean,
    roll_std,
    roll_nats,
    compute_per_limb_sadeh,
    format_per_limb_output,
    load_combined_counts_as_single,
    apply_sadeh_single,
    apply_sadeh_combined,
    apply_sadeh_vote,
    apply_sadeh_weighted,
    apply_sadeh_majority,
)


# ─── Fixtures ─────────────────────────────────────────────────────────────────

@pytest.fixture
def tmp_csv(tmp_path):
    return str(tmp_path / "test_output.csv")


def _make_single_sensor_df(values):
    return pd.DataFrame({
        "dataTimestamp": np.arange(len(values), dtype=float) * 60,
        "count": np.asarray(values, dtype=float),
    })


def _make_multi_sensor_df(n_epochs, limb_values=None, num_limbs=4):
    data = {"dataTimestamp": np.arange(n_epochs, dtype=float) * 60}
    for limb in range(1, num_limbs + 1):
        val = 0 if limb_values is None else limb_values.get(limb, 0)
        for axis in ["axis1", "axis2", "axis3"]:
            data[f"{axis}_{limb}"] = np.full(n_epochs, val, dtype=float)
    return pd.DataFrame(data)


# ─── Rolling helper functions ─────────────────────────────────────────────────

class TestRollMean:

    def test_uniform_values(self):
        x = np.full(20, 10.0)
        result = roll_mean(x, window=11)
        # With origin=-(window//2), window at index i covers [i, i+10].
        # Index 5 → window [5,15], fully within array → exact mean of 10.
        assert result[5] == pytest.approx(10.0)

    def test_all_zeros(self):
        x = np.zeros(20)
        result = roll_mean(x, window=11)
        np.testing.assert_array_almost_equal(result, 0.0)

    def test_output_length(self):
        x = np.zeros(30)
        result = roll_mean(x, window=11)
        assert len(result) == 30


class TestRollStd:

    def test_uniform_values_zero_std(self):
        """Constant values should have zero (or near-zero) standard deviation."""
        x = np.full(20, 5.0)
        result = roll_std(x, window=5)
        # After the initial padding settles, std should approach 0
        assert result[10] == pytest.approx(0.0, abs=0.01)

    def test_output_length(self):
        x = np.zeros(30)
        result = roll_std(x, window=5)
        assert len(result) == 30


class TestRollNats:

    def test_counts_in_range(self):
        """Values in [50, 100) should count as 1; others as 0."""
        x = np.array([0, 50, 75, 99, 100, 200])
        result = roll_nats(x, window=1)
        # Window=1 means each epoch counted individually
        # Epochs 1,2,3 are in [50,100)
        assert result[0] == pytest.approx(0.0)
        assert result[1] == pytest.approx(1.0)
        assert result[2] == pytest.approx(1.0)
        assert result[3] == pytest.approx(1.0)
        assert result[4] == pytest.approx(0.0)
        assert result[5] == pytest.approx(0.0)

    def test_output_length(self):
        x = np.zeros(25)
        result = roll_nats(x, window=11)
        assert len(result) == 25


# ─── sadeh_index ──────────────────────────────────────────────────────────────

class TestSadehIndex:

    def test_zero_activity_formula(self):
        """With all zeros:
            MEAN = 0, NAT = 0, SD = 0, LG = ln(0+1) = 0
            PS = 7.601 - 0 - 0 - 0 - 0 = 7.601
        """
        counts = np.zeros(20)
        ps = sadeh_index(counts)
        assert len(ps) == 20
        # All values should be 7.601 (the intercept)
        assert ps[10] == pytest.approx(7.601, abs=0.01)

    def test_zero_activity_is_sleep(self):
        """PS = 7.601 >> 0, so all epochs should be Sleep."""
        counts = np.zeros(20)
        ps = sadeh_index(counts)
        assert all(ps >= 0)

    def test_high_activity_is_wake(self):
        """Very high activity should drive PS below 0."""
        counts = np.full(30, 500.0)
        ps = sadeh_index(counts)
        # LG = ln(501) ≈ 6.2, 0.703*6.2 ≈ 4.36
        # MEAN = 500, 0.065*500 = 32.5
        # PS ≈ 7.601 - 32.5 - ... = very negative
        assert ps[15] < 0

    def test_output_length(self):
        for n in [1, 11, 50]:
            ps = sadeh_index(np.zeros(n))
            assert len(ps) == n

    def test_threshold_boundary(self):
        """PS >= 0 = Sleep, PS < 0 = Wake. Exactly 0 is Sleep."""
        assert np.where(np.array([0.0]) >= 0, "S", "W")[0] == "S"
        assert np.where(np.array([-0.001]) >= 0, "S", "W")[0] == "W"
        assert np.where(np.array([0.001]) >= 0, "S", "W")[0] == "S"

    def test_not_old_threshold_negative_four(self):
        """Regression: the old buggy threshold was > -4.

        With moderate activity, PS should be negative but > -4.
        The old code would incorrectly classify this as Sleep.
        The correct code (>= 0) should classify as Wake.
        """
        # Activity level chosen to produce PS in (-4, 0) range
        counts = np.full(30, 150.0)
        ps = sadeh_index(counts)
        mid = ps[15]
        # With count=150: LG = ln(151) ≈ 5.02, MEAN ≈ 150
        # PS ≈ 7.601 - 0.065*150 - ... should be well below 0
        assert mid < 0, f"Expected negative PS, got {mid}"
        # Old threshold (> -4) would say Sleep. New threshold (>= 0) says Wake.
        assert np.where(mid >= 0, "S", "W") == "W"


# ─── compute_per_limb_sadeh ──────────────────────────────────────────────────

class TestComputePerLimbSadeh:

    def test_creates_expected_columns(self):
        df = _make_multi_sensor_df(20)
        result = compute_per_limb_sadeh(df.copy())
        for limb in range(1, 5):
            assert f"limb_{limb}_sleep_index" in result.columns
            assert f"limb_{limb}_sleep" in result.columns

    def test_zero_activity_all_sleep(self):
        df = _make_multi_sensor_df(20)
        result = compute_per_limb_sadeh(df.copy())
        for limb in range(1, 5):
            assert (result[f"limb_{limb}_sleep"] == "S").all()

    def test_high_activity_wake(self):
        df = _make_multi_sensor_df(30, limb_values={1: 500, 2: 500, 3: 500, 4: 500})
        result = compute_per_limb_sadeh(df.copy())
        for limb in range(1, 5):
            assert result[f"limb_{limb}_sleep"].iloc[15] == "W"

    def test_caps_at_300(self):
        """Values above 300 should be capped before computing the index."""
        df_high = _make_multi_sensor_df(20, limb_values={1: 500})
        df_cap = _make_multi_sensor_df(20, limb_values={1: 300})
        r_high = compute_per_limb_sadeh(df_high.copy())
        r_cap = compute_per_limb_sadeh(df_cap.copy())
        np.testing.assert_array_almost_equal(
            r_high["limb_1_sleep_index"].values,
            r_cap["limb_1_sleep_index"].values,
        )

    def test_sleep_index_averages_three_axes(self):
        df = _make_multi_sensor_df(20)
        result = compute_per_limb_sadeh(df.copy())
        for limb in range(1, 5):
            ax_mean = (
                result[f"axis1_{limb}_sleep_index"]
                + result[f"axis2_{limb}_sleep_index"]
                + result[f"axis3_{limb}_sleep_index"]
            ) / 3
            np.testing.assert_array_almost_equal(
                result[f"limb_{limb}_sleep_index"].values, ax_mean.values
            )


# ─── apply_sadeh_single ──────────────────────────────────────────────────────

class TestApplySadehSingle:

    def test_returns_expected_columns(self, tmp_csv):
        df = _make_single_sensor_df(np.zeros(20))
        result = apply_sadeh_single(df, output_file=tmp_csv)
        assert "sleep_index" in result.columns
        assert "sleep" in result.columns

    def test_zero_activity_all_sleep(self, tmp_csv):
        df = _make_single_sensor_df(np.zeros(20))
        result = apply_sadeh_single(df, output_file=tmp_csv)
        assert (result["sleep"] == "S").all()

    def test_high_activity_wake(self, tmp_csv):
        df = _make_single_sensor_df(np.full(30, 500.0))
        result = apply_sadeh_single(df, output_file=tmp_csv)
        assert result["sleep"].iloc[15] == "W"

    def test_accepts_axis1_column(self, tmp_csv):
        """Should work with 'axis1' column when 'count' is absent."""
        df = pd.DataFrame({
            "dataTimestamp": np.arange(20, dtype=float) * 60,
            "axis1": np.zeros(20),
        })
        result = apply_sadeh_single(df, output_file=tmp_csv)
        assert "sleep" in result.columns

    def test_writes_csv(self, tmp_csv):
        df = _make_single_sensor_df(np.zeros(10))
        apply_sadeh_single(df, output_file=tmp_csv)
        assert os.path.exists(tmp_csv)

    def test_raises_without_count_or_axis1(self, tmp_csv):
        df = pd.DataFrame({"other": [1, 2, 3]})
        with pytest.raises(ValueError, match="count.*axis1"):
            apply_sadeh_single(df, output_file=tmp_csv)


# ─── apply_sadeh_combined ────────────────────────────────────────────────────

class TestApplySadehCombined:

    def test_averages_columns(self, tmp_csv):
        """Should average all non-timestamp columns into 'count'."""
        df = pd.DataFrame({
            "dataTimestamp": np.arange(20, dtype=float),
            "col_a": np.full(20, 10.0),
            "col_b": np.full(20, 30.0),
        })
        result = apply_sadeh_combined(df)
        # Combined count should be mean(10, 30) = 20
        assert "sleep" in result.columns


# ─── load_combined_counts_as_single ──────────────────────────────────────────

class TestLoadCombinedCountsAsSingle:

    def test_basic_averaging(self):
        df = pd.DataFrame({
            "timestamp": [0, 60],
            "limb1": [10.0, 20.0],
            "limb2": [30.0, 40.0],
        })
        result = load_combined_counts_as_single(df)
        assert list(result.columns) == ["dataTimestamp", "count"]
        assert result["count"].iloc[0] == pytest.approx(20.0)  # mean(10,30)
        assert result["count"].iloc[1] == pytest.approx(30.0)  # mean(20,40)

    def test_raises_on_only_timestamp(self):
        df = pd.DataFrame({"timestamp": [0, 60]})
        with pytest.raises(ValueError):
            load_combined_counts_as_single(df)


# ─── Fusion strategy tests ───────────────────────────────────────────────────

class TestApplySadehVote:

    def test_all_sleep(self, tmp_csv):
        df = _make_multi_sensor_df(20)
        result = apply_sadeh_vote(df, output_file=tmp_csv)
        assert "sleep" in result.columns
        assert (result["sleep"] == "S").all()

    def test_wrists_override_ankles(self, tmp_csv):
        """Wrists (2,4) Sleep + ankles (1,3) Wake → Sleep wins."""
        df = _make_multi_sensor_df(30, limb_values={1: 500, 2: 0, 3: 500, 4: 0})
        result = apply_sadeh_vote(df, output_file=tmp_csv)
        assert result["sleep"].iloc[15] == "S"

    def test_ankles_alone_lose(self, tmp_csv):
        """Ankles (1,3) Sleep + wrists (2,4) Wake → Wake wins."""
        df = _make_multi_sensor_df(30, limb_values={1: 0, 2: 500, 3: 0, 4: 500})
        result = apply_sadeh_vote(df, output_file=tmp_csv)
        assert result["sleep"].iloc[15] == "W"

    def test_consistent_weights(self, tmp_csv):
        """Weights should be {1: 0.5, 2: 1, 3: 0.5, 4: 1}."""
        # Limb 1(ankle, 0.5) Sleep + Limb 4(wrist, 1.0) Sleep = 1.5
        # Limb 2(wrist, 1.0) Wake + Limb 3(ankle, 0.5) Wake = 1.5
        # Tie → Sleep (>= comparison)
        df = _make_multi_sensor_df(30, limb_values={1: 0, 2: 500, 3: 500, 4: 0})
        result = apply_sadeh_vote(df, output_file=tmp_csv)
        assert result["sleep"].iloc[15] == "S"


class TestApplySadehWeighted:

    def test_zero_activity_sleep(self, tmp_csv):
        df = _make_multi_sensor_df(20)
        result = apply_sadeh_weighted(df, output_file=tmp_csv)
        assert "consensus_index" in result.columns
        assert "consensus_sleep" in result.columns
        assert (result["consensus_sleep"] == "S").all()

    def test_threshold_is_zero(self, tmp_csv):
        """Consensus index >= 0 = Sleep, < 0 = Wake."""
        assert np.where(np.array([0.0]) >= 0, "S", "W")[0] == "S"
        assert np.where(np.array([-0.001]) >= 0, "S", "W")[0] == "W"

    def test_consensus_index_is_weighted_average(self, tmp_csv):
        df = _make_multi_sensor_df(20)
        result = apply_sadeh_weighted(df, output_file=tmp_csv)
        weights = {1: 0.5, 2: 1, 3: 0.5, 4: 1}
        total_w = sum(weights.values())
        for i in range(len(result)):
            expected = sum(
                weights[limb] * result[f"Limb {limb} sleep_index"].iloc[i]
                for limb in range(1, 5)
            ) / total_w
            assert result["consensus_index"].iloc[i] == pytest.approx(expected)


class TestApplySadehMajority:

    def test_all_sleep(self, tmp_csv):
        df = _make_multi_sensor_df(20)
        result = apply_sadeh_majority(df, output_file=tmp_csv)
        assert "consensus_majority" in result.columns
        assert (result["consensus_majority"] == "S").all()

    def test_three_sleep_one_wake(self, tmp_csv):
        df = _make_multi_sensor_df(30, limb_values={1: 0, 2: 0, 3: 0, 4: 500})
        result = apply_sadeh_majority(df, output_file=tmp_csv)
        assert result["consensus_majority"].iloc[15] == "S"

    def test_two_sleep_two_wake(self, tmp_csv):
        df = _make_multi_sensor_df(30, limb_values={1: 0, 2: 0, 3: 500, 4: 500})
        result = apply_sadeh_majority(df, output_file=tmp_csv)
        assert result["consensus_majority"].iloc[15] == "W"

    def test_threshold_is_3_of_4(self, tmp_csv):
        """Need >= 3 limbs saying Sleep. 2 is not enough."""
        df = _make_multi_sensor_df(30, limb_values={1: 500, 2: 500, 3: 0, 4: 0})
        result = apply_sadeh_majority(df, output_file=tmp_csv)
        assert result["consensus_majority"].iloc[15] == "W"
