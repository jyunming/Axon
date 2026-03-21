"""Tests for _copilot_device_flow in axon.llm (lines 38-82)."""
from unittest.mock import MagicMock, patch

import pytest


def _make_mock_post(device_resp_data, token_resp_data_list):
    """Build a mock for httpx.post with configurable responses."""
    device_mock = MagicMock()
    device_mock.raise_for_status.return_value = None
    device_mock.json.return_value = device_resp_data

    token_mocks = []
    for data in token_resp_data_list:
        m = MagicMock()
        m.json.return_value = data
        token_mocks.append(m)

    side_effects = [device_mock] + token_mocks
    return MagicMock(side_effect=side_effects)


class TestCopilotDeviceFlow:
    def test_success_on_first_poll(self, capsys):
        """Device flow returns access token on first poll (lines 38-74)."""
        from axon.llm import _copilot_device_flow

        device_data = {
            "verification_uri": "https://github.com/login/device",
            "user_code": "ABCD-1234",
            "device_code": "device_abc",
            "interval": 1,
            "expires_in": 300,
        }
        token_data = {"access_token": "ghu_test_oauth_token"}

        mock_post = _make_mock_post(device_data, [token_data])

        with (
            patch("httpx.post", mock_post),
            patch("time.sleep"),
            patch("time.time", side_effect=[0, 0, 1]),  # within deadline
        ):
            result = _copilot_device_flow()

        assert result == "ghu_test_oauth_token"
        captured = capsys.readouterr()
        assert "github.com" in captured.out or "browser" in captured.out

    def test_success_after_pending_then_token(self, capsys):
        """Polls through authorization_pending then gets token (lines 75-80)."""
        from axon.llm import _copilot_device_flow

        device_data = {
            "verification_uri": "https://github.com/login/device",
            "user_code": "EFGH-5678",
            "device_code": "device_def",
            "interval": 1,
            "expires_in": 300,
        }
        # First poll: pending, second poll: token
        pending_resp = {"error": "authorization_pending"}
        success_resp = {"access_token": "ghu_second_attempt_token"}

        mock_post = _make_mock_post(device_data, [pending_resp, success_resp])

        call_count = [0]

        def fake_time():
            call_count[0] += 1
            return 0 if call_count[0] <= 4 else 0

        with (
            patch("httpx.post", mock_post),
            patch("time.sleep"),
            patch("time.time", side_effect=[0, 0, 0, 0, 0]),
        ):
            result = _copilot_device_flow()

        assert result == "ghu_second_attempt_token"

    def test_slow_down_increases_interval(self, capsys):
        """slow_down error increases polling interval (lines 76-77)."""
        from axon.llm import _copilot_device_flow

        device_data = {
            "verification_uri": "https://github.com/login/device",
            "user_code": "IJKL-9012",
            "device_code": "device_ghi",
            "interval": 2,
            "expires_in": 300,
        }
        slow_down_resp = {"error": "slow_down"}
        success_resp = {"access_token": "ghu_slow_token"}

        mock_post = _make_mock_post(device_data, [slow_down_resp, success_resp])
        sleep_calls = []

        with (
            patch("httpx.post", mock_post),
            patch("time.sleep", side_effect=lambda s: sleep_calls.append(s)),
            patch("time.time", side_effect=[0, 0, 0, 0]),
        ):
            result = _copilot_device_flow()

        assert result == "ghu_slow_token"
        # After slow_down, interval should be interval + 5 = 7
        assert any(s == 7 for s in sleep_calls)

    def test_unknown_error_raises_runtime_error(self, capsys):
        """Unknown OAuth error raises RuntimeError (lines 78-80)."""
        from axon.llm import _copilot_device_flow

        device_data = {
            "verification_uri": "https://github.com/login/device",
            "user_code": "MNOP-3456",
            "device_code": "device_jkl",
            "interval": 1,
            "expires_in": 300,
        }
        error_resp = {"error": "access_denied"}

        mock_post = _make_mock_post(device_data, [error_resp])

        with (
            patch("httpx.post", mock_post),
            patch("time.sleep"),
            patch("time.time", side_effect=[0, 0, 0]),
        ):
            with pytest.raises(RuntimeError, match="GitHub OAuth error: access_denied"):
                _copilot_device_flow()

    def test_timeout_raises_runtime_error(self, capsys):
        """Deadline expiry raises RuntimeError (lines 81-82)."""
        from axon.llm import _copilot_device_flow

        device_data = {
            "verification_uri": "https://github.com/login/device",
            "user_code": "QRST-7890",
            "device_code": "device_mno",
            "interval": 1,
            "expires_in": 5,
        }

        mock_post = _make_mock_post(device_data, [])

        # Simulate time.time() starting at 0, first call > deadline to exit loop immediately
        with (
            patch("httpx.post", mock_post),
            patch("time.sleep"),
            patch(
                "time.time", side_effect=[0, 100]
            ),  # deadline = 0+5=5, next call returns 100 > deadline
        ):
            with pytest.raises(RuntimeError, match="timed out"):
                _copilot_device_flow()
