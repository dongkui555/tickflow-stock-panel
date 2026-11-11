"""Maintenance readers must not race the app's daily-partition startup repair."""
import importlib.util
from pathlib import Path

import pytest

spec = importlib.util.spec_from_file_location('restore_panel_services',
                                            Path(__file__).with_name('restore_panel_services.py'))
service = importlib.util.module_from_spec(spec)
spec.loader.exec_module(service)


def test_app_startup_precedes_all_shadow_reads():
    events = []
    checks = iter([False, False, True])

    def healthy():
        ready = next(checks)
        events.append(('health', ready))
        return ready

    service.restore(list(reversed(service.ROLES)), start=lambda r: events.append(('start', r)),
                    healthy=healthy, wait=lambda _: events.append(('wait', None)),
                    monotonic=lambda: 0)
    assert events == [('start', 'app'), ('health', False), ('wait', None),
                      ('health', False), ('wait', None), ('health', True),
                      ('start', 'frozen-pair-shadow'), ('start', 'decision-pair-shadow'),
                      ('start', 'repaired-pair-shadow')]


def test_failed_startup_does_not_resume_shadow_readers():
    started, clock = [], iter([0, 181])
    with pytest.raises(TimeoutError, match='remain paused'):
        service.restore(service.ROLES, start=started.append, healthy=lambda: False,
                        wait=lambda _: None, monotonic=lambda: next(clock))
    assert started == ['app']


def test_only_services_paused_by_this_operation_are_started():
    started = []
    service.restore(['repaired-pair-shadow'], start=started.append, healthy=lambda: True,
                    wait=lambda _: None, monotonic=lambda: 0)
    assert started == ['repaired-pair-shadow']


@pytest.mark.parametrize('roles', [[], ['app', 'app'], ['stockdb-api']])
def test_invalid_pause_record_cannot_start_services(roles):
    with pytest.raises(ValueError):
        service.restore(roles, start=lambda _: pytest.fail('Unexpected service start'),
                        healthy=lambda: True, wait=lambda _: None, monotonic=lambda: 0)


def test_docker_start_failure_cannot_start_dependent_readers():
    def broken(_):
        raise RuntimeError('Docker start failed')

    with pytest.raises(RuntimeError, match='Docker start failed'):
        service.restore(service.ROLES, start=broken,
                        healthy=lambda: pytest.fail('Startup failure reached health check'),
                        wait=lambda _: None, monotonic=lambda: 0)
