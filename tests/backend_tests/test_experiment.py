import os

import pytest

from vivid.backends.experiments import LocalExperimentBackend, ExperimentBackend


@pytest.mark.parametrize('backend', [
    ExperimentBackend(),
    LocalExperimentBackend()
])
def test_can_call_method(backend: ExperimentBackend):
    backend.save_object('foo', {})
    backend.save_as_python_object('foo', {})

    assert not backend.can_save, backend.to
    assert backend.get_marked() is None


@pytest.mark.parametrize('backend', [
    ExperimentBackend(),
    LocalExperimentBackend()
])
def test_set_silent(backend: ExperimentBackend):
    with backend.silent():
        assert backend.logger.disabled == True


def test_local_mark(tmpdir):
    experiment = LocalExperimentBackend(tmpdir)
    assert experiment.can_save

    obj = {
        'bar': [1, 2]
    }
    experiment.mark('foo', obj)

    marking = experiment.get_marked()

    assert marking.get('foo') == obj
    assert os.path.exists(tmpdir)
