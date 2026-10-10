from pathlib import Path

import pytest

from inference.io import audio_io


@pytest.mark.skipif(
    not Path("/opt/homebrew/bin/ffmpeg").is_file(),
    reason="Apple Silicon Homebrew ffmpeg is not installed on this machine",
)
def test_find_ffmpeg_from_finder_style_minimal_path(monkeypatch):
    """Finder launches do not inherit Homebrew's PATH."""
    monkeypatch.setenv("PATH", "/usr/bin:/bin:/usr/sbin:/sbin")
    monkeypatch.setattr(audio_io, "_FFMPEG_PATH", Path("/missing/project/ffmpeg"))

    assert Path(audio_io._find_ffmpeg()).resolve() == Path("/opt/homebrew/bin/ffmpeg").resolve()
