"""Compatibility entrypoint for the Fluent GUI.

The implementation is split into the `gui` package.
"""

import onnxruntime  # Preload ORT before Qt to avoid DLL init conflicts in portable Python.

from gui.fluent_main import run_app


if __name__ == '__main__':
    run_app()
