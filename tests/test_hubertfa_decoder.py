from __future__ import annotations

import importlib
import sys

import pytest

import inference.HubertFA.tools.decoder as decoder


def test_hubertfa_decoder_reload_does_not_import_plot_module():
    sys.modules.pop("inference.HubertFA.tools.plot", None)

    reloaded = importlib.reload(decoder)

    assert "inference.HubertFA.tools.plot" not in sys.modules
    assert reloaded.AlignmentDecoder is not None


def test_hubertfa_plot_helpers_require_optional_matplotlib(monkeypatch):
    def _raise_missing_matplotlib(_name, package=None):
        del package
        exc = ModuleNotFoundError("No module named 'matplotlib'")
        exc.name = "matplotlib"
        raise exc

    monkeypatch.setattr(decoder.importlib, "import_module", _raise_missing_matplotlib)

    with pytest.raises(RuntimeError, match="optional 'matplotlib'"):
        decoder._load_plot_helpers()
