import plistlib

from scripts.build_macos_app import build_bundle


def test_build_bundle_creates_relocatable_app_shell(tmp_path):
    project = tmp_path / "checkout"
    project.mkdir()
    (project / "app_fluent.py").write_text("# test\n", encoding="utf-8")

    bundle = build_bundle(project, tmp_path / "Vocal2Midi.app", sign=False)

    launcher = bundle / "Contents" / "MacOS" / "Vocal2Midi"
    assert launcher.is_file()
    assert launcher.stat().st_mode & 0o111
    assert str(project.resolve()) in (bundle / "Contents" / "Resources" / "project-path.txt").read_text()
    with (bundle / "Contents" / "Info.plist").open("rb") as handle:
        assert plistlib.load(handle)["CFBundleIdentifier"] == "com.xiantaidu.vocal2midi.mac"
