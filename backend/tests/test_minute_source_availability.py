import pytest

from app.services.minute_failover import MinuteSourcesUnavailableError


@pytest.mark.parametrize("backup", ["not_requested", "backup_unavailable", "source_unsupported"])
def test_unsupported_primary_and_missing_backup_is_source_outage(backup):
    error = MinuteSourcesUnavailableError([
        {"source": "promax", "symbol": "920701", "date": "2026-10-21", "reason": "source_unsupported"},
        {"source": "tdx", "symbol": "920701", "date": "2026-10-21", "reason": backup,
         "primary_reason": "source_unsupported", "retry_after_seconds": 180}])
    assert error.reason == "upstream_unavailable"
    assert error.retry_after_seconds == 180


def test_real_conflict_is_not_reclassified_as_an_outage():
    error = MinuteSourcesUnavailableError([
        {"source": "tdx", "symbol": "920701", "date": "2026-10-21", "reason": "existing_values_conflict",
         "primary_reason": "source_unsupported"}])
    assert error.reason == "minute_validation_failed"
    assert not error.retryable
