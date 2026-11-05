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


def test_build_bundle_refuses_to_delete_non_bundle(tmp_path):
    import pytest

    project = tmp_path / "checkout"
    project.mkdir()
    (project / "app_fluent.py").write_text("# test\n", encoding="utf-8")
    not_a_bundle = tmp_path / "Data.app"
    not_a_bundle.mkdir()
    (not_a_bundle / "keep.txt").write_text("x", encoding="utf-8")

    with pytest.raises(ValueError):
        build_bundle(project, tmp_path / "plain_dir", sign=False)
    with pytest.raises(ValueError):
        build_bundle(project, not_a_bundle, sign=False)
    assert (not_a_bundle / "keep.txt").is_file()


def test_build_bundle_refuses_symlinked_output_bundle(tmp_path):
    import pytest

    project = tmp_path / "checkout"
    project.mkdir()
    (project / "app_fluent.py").write_text("# test\n", encoding="utf-8")
    other_app = tmp_path / "Other.app"
    (other_app / "Contents").mkdir(parents=True)
    (other_app / "Contents" / "keep.txt").write_text("x", encoding="utf-8")
    link = tmp_path / "Link.app"
    link.symlink_to(other_app, target_is_directory=True)

    with pytest.raises(ValueError):
        build_bundle(project, link, sign=False)
    assert link.is_symlink()
    assert (other_app / "Contents" / "keep.txt").is_file()
