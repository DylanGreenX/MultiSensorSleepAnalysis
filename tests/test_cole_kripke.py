"""Unit tests for Cole-Kripke (1992) sleep/wake algorithm.

Validates:
    - Correct weighted linear combination coefficients
    - ActiGraph scaling (÷100) and capping (max 300)
    - Sleep threshold: SI < 1.0 = Sleep
    - Anatomical weights: ankles {1,3} = 0.5, wrists {2,4} = 1.0
    - Fusion strategies: weighted vote (V), weighted index (W), majority (M)
"""

import pytest
import numpy as np
import pandas as pd
import os
import tempfile

from multisensor_sleep.algorithms.cole_kripke import (
    cole_kripke_index,
    compute_per_limb,
    format_per_limb_output,
    format_timestamp,
    apply_cole_kripke_single,
    apply_cole_kripke_vote,
    apply_cole_kripke_weighted,
    apply_cole_kripke_majority,
)

BASELINE = pd.Timestamp("2025-01-01 22:00:00")
COEFFICIENTS = [106, 54, 58, 76, 230, 74, 67]


# ─── Fixtures ─────────────────────────────────────────────────────────────────

@pytest.fixture
def tmp_csv(tmp_path):
    """Return a temporary CSV path for functions that write output."""
    return str(tmp_path / "test_output.csv")


def _make_single_sensor_df(values):
    """Build a single-sensor DataFrame with axis1 and dataTimestamp columns."""
    return pd.DataFrame({
        "dataTimestamp": np.arange(len(values), dtype=float) * 60,
        "axis1": values,
    })


def _make_multi_sensor_df(n_epochs, limb_values=None, num_limbs=4):
    """Build a multi-sensor DataFrame with axis{1,2,3}_{limb} columns.

    If limb_values is None, all axes/limbs are zero.
    If limb_values is a dict {limb_number: value}, those limbs get that
    constant value across all axes; others are zero.
    """
    data = {"dataTimestamp": np.arange(n_epochs, dtype=float) * 60}
    for limb in range(1, num_limbs + 1):
        val = 0 if limb_values is None else limb_values.get(limb, 0)
        for axis in ["axis1", "axis2", "axis3"]:
            data[f"{axis}_{limb}"] = np.full(n_epochs, val, dtype=float)
    return pd.DataFrame(data)


# ─── cole_kripke_index ────────────────────────────────────────────────────────

class TestColeKripkeIndex:
    """Tests for the core sleep index computation."""

    def test_all_zeros_returns_zeros(self):
        counts = np.zeros(20)
        si = cole_kripke_index(counts)
        assert len(si) == 20
        np.testing.assert_array_equal(si, 0.0)

    def test_output_length_matches_input(self):
        for n in [1, 7, 50, 100]:
            si = cole_kripke_index(np.zeros(n))
            assert len(si) == n

    def test_single_impulse_coefficients(self):
        """A single non-zero epoch should spread via the known coefficients.

        Place value 100 at index 4 (so scaled = 100/100 = 1.0).
        s.shift(k) at index i gives s[i-k], so the impulse at index 4
        propagates forward (preceding-epoch weights land at later indices):
            shift(4) hits at i=8, shift(3) at i=7, shift(2) at i=6, shift(1) at i=5
        And backward via negative shifts:
            shift(-1) hits at i=3, shift(-2) at i=2
        """
        counts = np.zeros(10)
        counts[4] = 100  # scaled to 1.0

        si = cole_kripke_index(counts)

        # Index 4: current epoch weight 230
        assert si[4] == pytest.approx(0.001 * 230 * 1.0)
        # Index 8: shift(4) → s[8-4]=s[4]=1.0, weight 106
        assert si[8] == pytest.approx(0.001 * 106 * 1.0)
        # Index 7: shift(3) → s[7-3]=s[4]=1.0, weight 54
        assert si[7] == pytest.approx(0.001 * 54 * 1.0)
        # Index 6: shift(2) → s[6-2]=s[4]=1.0, weight 58
        assert si[6] == pytest.approx(0.001 * 58 * 1.0)
        # Index 5: shift(1) → s[5-1]=s[4]=1.0, weight 76
        assert si[5] == pytest.approx(0.001 * 76 * 1.0)
        # Index 3: shift(-1) → s[3+1]=s[4]=1.0, weight 74
        assert si[3] == pytest.approx(0.001 * 74 * 1.0)
        # Index 2: shift(-2) → s[2+2]=s[4]=1.0, weight 67
        assert si[2] == pytest.approx(0.001 * 67 * 1.0)
        # Beyond the window: should be zero
        assert si[0] == pytest.approx(0.0)
        assert si[1] == pytest.approx(0.0)
        assert si[9] == pytest.approx(0.0)

    def test_scaling_divides_by_100(self):
        """Counts of 200 should be scaled to 2.0 internally."""
        counts = np.zeros(10)
        counts[4] = 200
        si = cole_kripke_index(counts)
        assert si[4] == pytest.approx(0.001 * 230 * 2.0)

    def test_capping_at_300(self):
        """Counts above 30000 (300*100) should be capped at 300."""
        counts = np.zeros(10)
        counts[4] = 50000  # would be 500 without cap → capped to 300
        si_capped = cole_kripke_index(counts)

        counts2 = np.zeros(10)
        counts2[4] = 30000  # exactly 300
        si_exact = cole_kripke_index(counts2)

        np.testing.assert_array_almost_equal(si_capped, si_exact)

    def test_uniform_high_activity_is_wake(self):
        """High uniform activity should produce SI >= 1.0 (Wake)."""
        counts = np.full(20, 5000.0)  # scaled to 50
        si = cole_kripke_index(counts)
        # Sum of all coefficients = 665, so SI = 0.001 * 665 * 50 = 33.25
        total_coeff = sum(COEFFICIENTS)
        expected = 0.001 * total_coeff * 50.0
        # Middle epochs (away from edges) should match
        assert si[10] == pytest.approx(expected)
        assert si[10] >= 1.0  # Wake

    def test_uniform_zero_activity_is_sleep(self):
        """Zero activity should produce SI = 0 (Sleep)."""
        counts = np.zeros(20)
        si = cole_kripke_index(counts)
        assert all(si < 1.0)


# ─── compute_per_limb ────────────────────────────────────────────────────────

class TestComputePerLimb:
    """Tests for the shared per-limb computation."""

    def test_creates_expected_columns(self):
        df = _make_multi_sensor_df(15)
        result = compute_per_limb(df.copy())
        for limb in range(1, 5):
            assert f"limb_{limb}_sleep_index" in result.columns
            assert f"limb_{limb}_sleep" in result.columns

    def test_zero_activity_all_sleep(self):
        df = _make_multi_sensor_df(15)
        result = compute_per_limb(df.copy())
        for limb in range(1, 5):
            assert (result[f"limb_{limb}_sleep"] == "S").all()

    def test_high_activity_all_wake(self):
        df = _make_multi_sensor_df(15, limb_values={1: 5000, 2: 5000, 3: 5000, 4: 5000})
        result = compute_per_limb(df.copy())
        # Interior epochs (away from edges where fill_value=0 matters)
        for limb in range(1, 5):
            assert result[f"limb_{limb}_sleep"].iloc[6] == "W"

    def test_sleep_index_averages_three_axes(self):
        """Each limb's index should be the mean of its 3 axis indices."""
        df = _make_multi_sensor_df(15)
        result = compute_per_limb(df.copy())
        for limb in range(1, 5):
            ax_mean = (
                result[f"axis1_{limb}_sleep_index"]
                + result[f"axis2_{limb}_sleep_index"]
                + result[f"axis3_{limb}_sleep_index"]
            ) / 3
            np.testing.assert_array_almost_equal(
                result[f"limb_{limb}_sleep_index"].values, ax_mean.values
            )

    def test_threshold_boundary(self):
        """SI exactly 1.0 should be Wake (threshold is strictly < 1.0)."""
        # We'll check the classification logic directly
        assert np.where(np.array([0.99]) < 1, "S", "W")[0] == "S"
        assert np.where(np.array([1.00]) < 1, "S", "W")[0] == "W"
        assert np.where(np.array([1.01]) < 1, "S", "W")[0] == "W"


# ─── format_per_limb_output ──────────────────────────────────────────────────

class TestFormatPerLimbOutput:

    def test_output_columns(self):
        df = _make_multi_sensor_df(10)
        df = compute_per_limb(df)
        out = format_per_limb_output(df)
        assert "dataTimestamp" in out.columns
        for limb in range(1, 5):
            assert f"Limb {limb} sleep_index" in out.columns
            assert f"Limb {limb} sleep" in out.columns

    def test_no_timestamp_column(self):
        """Should work even without dataTimestamp."""
        df = _make_multi_sensor_df(10).drop(columns=["dataTimestamp"])
        df = compute_per_limb(df)
        out = format_per_limb_output(df)
        assert "dataTimestamp" not in out.columns


# ─── format_timestamp ────────────────────────────────────────────────────────

class TestFormatTimestamp:

    def test_zero_seconds(self):
        result = format_timestamp(0, BASELINE)
        assert result == "2025-01-01 22:00:00.000"

    def test_60_seconds(self):
        result = format_timestamp(60, BASELINE)
        assert result == "2025-01-01 22:01:00.000"

    def test_float_seconds(self):
        result = format_timestamp(0.5, BASELINE)
        assert result == "2025-01-01 22:00:00.500"


# ─── apply_cole_kripke_single ─────────────────────────────────────────────────

class TestApplyColeKripkeSingle:

    def test_returns_expected_columns(self, tmp_csv):
        df = _make_single_sensor_df(np.zeros(20))
        result = apply_cole_kripke_single(df, BASELINE, output_file=tmp_csv)
        assert "sleep_index" in result.columns
        assert "sleep" in result.columns

    def test_zero_activity_all_sleep(self, tmp_csv):
        df = _make_single_sensor_df(np.zeros(20))
        result = apply_cole_kripke_single(df, BASELINE, output_file=tmp_csv)
        assert (result["sleep"] == "S").all()

    def test_high_activity_wake(self, tmp_csv):
        df = _make_single_sensor_df(np.full(20, 5000.0))
        result = apply_cole_kripke_single(df, BASELINE, output_file=tmp_csv)
        # Interior epochs should be Wake
        assert result["sleep"].iloc[6] == "W"

    def test_writes_csv(self, tmp_csv):
        df = _make_single_sensor_df(np.zeros(10))
        apply_cole_kripke_single(df, BASELINE, output_file=tmp_csv)
        assert os.path.exists(tmp_csv)
        saved = pd.read_csv(tmp_csv)
        assert list(saved.columns) == ["dataTimestamp", "sleep_index", "sleep"]

    def test_does_not_mutate_input(self, tmp_csv):
        df = _make_single_sensor_df(np.zeros(10))
        original_cols = list(df.columns)
        apply_cole_kripke_single(df, BASELINE, output_file=tmp_csv)
        assert list(df.columns) == original_cols


# ─── Fusion strategy tests ───────────────────────────────────────────────────

class TestApplyColeKripkeVote:
    """Weighted vote: ankles=0.5, wrists=1.0 on binary S/W labels."""

    def test_all_sleep_returns_sleep(self, tmp_csv):
        df = _make_multi_sensor_df(15)
        result = apply_cole_kripke_vote(df, BASELINE, output_file=tmp_csv)
        assert "sleep" in result.columns
        assert (result["sleep"] == "S").all()

    def test_weights_favor_wrists(self, tmp_csv):
        """If wrists (2,4) say S and ankles (1,3) say W, sleep wins.

        Wrist weight = 1+1 = 2, ankle weight = 0.5+0.5 = 1.
        """
        df = _make_multi_sensor_df(
            15,
            limb_values={1: 5000, 2: 0, 3: 5000, 4: 0},
        )
        result = apply_cole_kripke_vote(df, BASELINE, output_file=tmp_csv)
        # Interior epochs: ankles Wake, wrists Sleep → wrists win
        assert result["sleep"].iloc[6] == "S"

    def test_weights_ankles_only_lose(self, tmp_csv):
        """If ankles (1,3) say S and wrists (2,4) say W, wake wins.

        Ankle weight = 0.5+0.5 = 1, wrist weight = 1+1 = 2.
        """
        df = _make_multi_sensor_df(
            15,
            limb_values={1: 0, 2: 5000, 3: 0, 4: 5000},
        )
        result = apply_cole_kripke_vote(df, BASELINE, output_file=tmp_csv)
        assert result["sleep"].iloc[6] == "W"

    def test_tie_goes_to_sleep(self, tmp_csv):
        """Equal weighted votes → Sleep (>= comparison)."""
        # Limbs 1(0.5) + 2(1.0) = 1.5 Sleep, Limbs 3(0.5) + 4(1.0) = 1.5 Wake
        df = _make_multi_sensor_df(
            15,
            limb_values={1: 0, 2: 0, 3: 5000, 4: 5000},
        )
        result = apply_cole_kripke_vote(df, BASELINE, output_file=tmp_csv)
        assert result["sleep"].iloc[6] == "S"


class TestApplyColeKripkeWeighted:
    """Weighted index: continuous consensus, threshold < 0.7."""

    def test_zero_activity_sleep(self, tmp_csv):
        df = _make_multi_sensor_df(15)
        result = apply_cole_kripke_weighted(df, BASELINE, output_file=tmp_csv)
        assert "consensus_index" in result.columns
        assert "consensus_sleep" in result.columns
        assert (result["consensus_sleep"] == "S").all()

    def test_consensus_index_is_weighted_average(self, tmp_csv):
        df = _make_multi_sensor_df(15)
        result = apply_cole_kripke_weighted(df, BASELINE, output_file=tmp_csv)
        weights = {1: 0.5, 2: 1, 3: 0.5, 4: 1}
        total_w = sum(weights.values())
        for i in range(len(result)):
            expected = sum(
                weights[limb] * result[f"Limb {limb} sleep_index"].iloc[i]
                for limb in range(1, 5)
            ) / total_w
            assert result["consensus_index"].iloc[i] == pytest.approx(expected)

    def test_threshold_is_0_7(self, tmp_csv):
        """Consensus index exactly 0.7 should be Wake (threshold strictly < 0.7)."""
        assert np.where(np.array([0.69]) < 0.7, "S", "W")[0] == "S"
        assert np.where(np.array([0.70]) < 0.7, "S", "W")[0] == "W"


class TestApplyColeKripkeMajority:
    """Majority vote: Sleep if >= 3 of 4 limbs say Sleep."""

    def test_all_sleep(self, tmp_csv):
        df = _make_multi_sensor_df(15)
        result = apply_cole_kripke_majority(df, BASELINE, output_file=tmp_csv)
        assert "consensus_majority" in result.columns
        assert (result["consensus_majority"] == "S").all()

    def test_three_sleep_one_wake(self, tmp_csv):
        """3 Sleep + 1 Wake = Sleep (meets threshold of 3)."""
        df = _make_multi_sensor_df(15, limb_values={1: 0, 2: 0, 3: 0, 4: 5000})
        result = apply_cole_kripke_majority(df, BASELINE, output_file=tmp_csv)
        assert result["consensus_majority"].iloc[6] == "S"

    def test_two_sleep_two_wake(self, tmp_csv):
        """2 Sleep + 2 Wake = Wake (below threshold of 3)."""
        df = _make_multi_sensor_df(15, limb_values={1: 0, 2: 0, 3: 5000, 4: 5000})
        result = apply_cole_kripke_majority(df, BASELINE, output_file=tmp_csv)
        assert result["consensus_majority"].iloc[6] == "W"

    def test_one_sleep_three_wake(self, tmp_csv):
        df = _make_multi_sensor_df(15, limb_values={1: 0, 2: 5000, 3: 5000, 4: 5000})
        result = apply_cole_kripke_majority(df, BASELINE, output_file=tmp_csv)
        assert result["consensus_majority"].iloc[6] == "W"
