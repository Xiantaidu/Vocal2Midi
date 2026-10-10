#!/usr/bin/env python3
"""Build llama.cpp for Vocal2Midi's Qwen3-ASR runtime on macOS and Windows.

This script is deliberately **non-invasive**:

* it never patches llama.cpp -- upstream sources are only cloned and compiled;
* it never touches Vocal2Midi's Python sources -- the platform specific
  library names it produces are exactly the ones
  ``inference/qwen3asr_dml/llama.py`` already looks up in its ``bin`` folder;

      darwin  -> libggml.dylib / libggml-base.dylib / libllama.dylib
      win32   -> ggml.dll      / ggml-base.dll      / llama.dll
      linux   -> libggml.so    / libggml-base.so    / libllama.so

  so switching llama.cpp builds is a matter of dropping files into ``bin``.

Typical use::

    # macOS (Metal + CPU), the default for this project
    python scripts/build_llama_runtime.py

    # Windows (DirectML + CPU)
    python scripts/build_llama_runtime.py --backend dml

    # keep the checkout, just rebuild after changing options
    python scripts/build_llama_runtime.py --skip-fetch
"""

from __future__ import annotations

import argparse
import os
import platform
import re
import shutil
import subprocess
import sys
from pathlib import Path

REPO_URL = "https://github.com/ggml-org/llama.cpp.git"
DEFAULT_TAG = "b9453"

# Core libraries that llama.py loads through ctypes, keyed by platform.
# Values are the "logical" names without any version suffix.
CORE_LIBS = ("ggml", "ggml-base", "llama")

# Optional backend modules that ggml_backend_load_all() dlopen()s.
BACKEND_MODULES = {
    "metal": "ggml-metal",
    "cpu": "ggml-cpu",
    "dml": "ggml-dml",
    "cuda": "ggml-cuda",
    "vulkan": "ggml-vulkan",
    "blas": "ggml-blas",
    "rpc": "ggml-rpc",
    "cann": "ggml-cann",
    "sycl": "ggml-sycl",
    "musa": "ggml-musa",
    "opencl": "ggml-opencl",
    "amx": "ggml-amx",
    "cpu_arm": "ggml-cpu-aarch64",
    "cpu_x64": "ggml-cpu-x64",
}


def lib_name(logical: str, system: str) -> str:
    """Return the file name llama.py expects for ``logical`` on ``system``."""
    if system == "win32":
        return f"{logical}.dll"
    if system == "darwin":
        return f"lib{logical}.dylib"
    return f"lib{logical}.so"


def sover_name(logical: str, system: str) -> str:
    """The ``.0`` compatibility symlink name (POSIX only)."""
    if system == "win32":
        return f"{logical}.dll"
    if system == "darwin":
        return f"lib{logical}.0.dylib"
    return f"lib{logical}.so.0"


def run(cmd: list[str], cwd: Path | None = None, dry_run: bool = False) -> None:
    printable = " ".join(str(c) for c in cmd)
    print(f"  $ {printable}")
    if dry_run:
        return
    subprocess.run([str(c) for c in cmd], cwd=str(cwd) if cwd else None, check=True)


def find_cmake() -> str:
    """Locate cmake: PATH first, then a pip-installed one (``pip install cmake``)."""
    found = shutil.which("cmake")
    if found:
        return found

    import glob

    patterns = [
        os.path.join(sys.prefix, "lib", "python*", "site-packages", "cmake", "data", "bin", "cmake"),
        os.path.join(os.path.expanduser("~"), ".workbuddy", "binaries", "python", "envs",
                     "*", "lib", "python*", "site-packages", "cmake", "data", "bin", "cmake"),
    ]
    for pattern in patterns:
        for hit in sorted(glob.glob(pattern)):
            if os.access(hit, os.X_OK):
                return hit

    sys.exit("error: 'cmake' not found on PATH. Install it with `pip install cmake` "
             "or from https://cmake.org/download/.")


def ensure_tool(name: str, hint: str) -> str:
    found = shutil.which(name)
    if not found:
        sys.exit(f"error: '{name}' not found on PATH. {hint}")
    return found


def cmake_configure_args(system: str, backend: str, install_prefix: Path) -> list[str]:
    args = [
        "-DCMAKE_BUILD_TYPE=Release",
        "-DBUILD_SHARED_LIBS=ON",
        "-DLLAMA_BUILD_TESTS=OFF",
        "-DLLAMA_BUILD_EXAMPLES=OFF",
        "-DLLAMA_BUILD_TOOLS=OFF",
        "-DLLAMA_BUILD_SERVER=OFF",
        "-DLLAMA_CURL=OFF",
        "-DGGML_BLAS=OFF",
    ]

    if system == "darwin":
        args += ["-DGGML_METAL=ON" if backend in ("auto", "metal") else "-DGGML_METAL=OFF"]
        args += ["-DGGML_ACCELERATE=ON"]
    elif system == "win32":
        if backend == "dml":
            args += ["-DGGML_DML=ON"]
        elif backend == "cuda":
            args += ["-DGGML_CUDA=ON"]
        elif backend == "vulkan":
            args += ["-DGGML_VULKAN=ON"]
    elif system == "linux":
        if backend == "cuda":
            args += ["-DGGML_CUDA=ON"]
        elif backend == "vulkan":
            args += ["-DGGML_VULKAN=ON"]

    return args


def pick_generator(system: str, jobs: int) -> list[str]:
    """Prefer Ninja when available; fall back to the platform default."""
    if shutil.which("ninja"):
        return ["-G", "Ninja"]
    if system == "win32":
        return ["-G", "Visual Studio 17 2022", "-A", "x64"]
    return []


RUNTIME_LIB_RE = re.compile(
    r"^(lib)?(llama|ggml)(-[A-Za-z0-9_]+)*(\.[0-9]+)*\.(dylib|so(\.[0-9]+)*|dll)$"
)


def purge_stale_libraries(install_dir: Path, dry_run: bool) -> list[str]:
    """Remove llama/ggml libraries left over from a previous build.

    Needed because ``ggml_backend_load_all()`` dlopen()s every backend module
    sitting next to the core libraries -- a stale ``libggml-metal.dylib`` would
    resurrect the Metal backend even after switching to a CPU-only build.
    """
    removed = []
    for path in sorted(install_dir.iterdir()):
        if RUNTIME_LIB_RE.match(path.name):
            print(f"  x  清理旧库 {path.name}")
            if not dry_run:
                path.unlink()
            removed.append(path.name)
    return removed


def install_libraries(build_dir: Path, install_dir: Path, system: str,
                      backend: str, dry_run: bool) -> list[str]:
    """Copy the built libraries into ``install_dir`` using llama.py's names."""
    src_bin = build_dir / "bin"
    if not src_bin.is_dir():
        alt = build_dir / "src" / "bin"
        src_bin = alt if alt.is_dir() else build_dir
    if not src_bin.is_dir():
        if dry_run:
            print(f"  (dry-run) 期望产物目录: {src_bin}")
            return []
        sys.exit(f"error: 编译产物目录不存在: {src_bin}")

    wanted = list(CORE_LIBS)
    if system == "darwin":
        wanted.append(BACKEND_MODULES["metal"])
        wanted.append(BACKEND_MODULES["cpu"])
    elif system == "win32":
        if backend == "dml":
            wanted.append(BACKEND_MODULES["dml"])
        wanted.append(BACKEND_MODULES["cpu"])

    installed: list[str] = []
    for logical in wanted:
        pattern = re.compile(
            r"^(lib)?%s(\.[0-9]+)*\.(dylib|so(\.[0-9]+)*|dll)$" % re.escape(logical)
        )
        candidates = [p for p in src_bin.iterdir()
                      if pattern.match(p.name) and not p.is_symlink() and p.is_file()]
        if not candidates:
            print(f"  ! 未找到 {logical} 的产物，跳过")
            continue

        # Prefer the file carrying the highest version suffix (the real object).
        real = sorted(candidates, key=lambda p: len(p.name))[-1]
        target = install_dir / lib_name(logical, system)
        print(f"  -> {target.name}   (from {real.name})")
        if not dry_run:
            shutil.copy2(real, target)
        installed.append(target.name)

        # POSIX: recreate the ".0" compatibility symlink llama.cpp uses.
        if system != "win32":
            link = install_dir / sover_name(logical, system)
            if not dry_run:
                if link.is_symlink() or link.exists():
                    link.unlink()
                link.symlink_to(target.name)
            installed.append(link.name)

    return installed


def verify(install_dir: Path, system: str) -> bool:
    """Load the installed libraries the same way llama.py does."""
    import ctypes

    try:
        ggml = ctypes.CDLL(str(install_dir / lib_name("ggml", system)))
        ggml_base = ctypes.CDLL(str(install_dir / lib_name("ggml-base", system)))
        llama = ctypes.CDLL(str(install_dir / lib_name("llama", system)))
        ggml.ggml_backend_load_all.restype = None
        ggml.ggml_backend_load_all()
        for sym in ("llama_model_default_params", "llama_model_load_from_file",
                    "llama_init_from_model", "llama_decode"):
            getattr(llama, sym)
    except OSError as exc:
        print(f"  验证失败: {exc}")
        return False
    print("  运行时加载验证通过")
    return True


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tag", default=DEFAULT_TAG,
                    help=f"llama.cpp tag/release to build (default: {DEFAULT_TAG})")
    ap.add_argument("--source-dir", type=Path,
                    help="llama.cpp checkout to use (default: <build-dir>/../llama.cpp-<tag>)")
    ap.add_argument("--build-dir", type=Path,
                    help="cmake build directory (default: <source-dir>/build-<tag>)")
    ap.add_argument("--install-dir", type=Path,
                    help="where the runtime libraries go "
                         "(default: <repo>/inference/qwen3asr_dml/bin)")
    ap.add_argument("--backend", default="auto",
                    choices=["auto", "metal", "dml", "cuda", "vulkan", "cpu"],
                    help="accelerator backend (default: auto -> metal on macOS, dml on Windows)")
    ap.add_argument("--jobs", type=int, default=0, help="parallel build jobs (0 = auto)")
    ap.add_argument("--skip-fetch", action="store_true",
                    help="reuse an existing checkout instead of cloning")
    ap.add_argument("--no-backup", action="store_true",
                    help="do not back up the libraries already in --install-dir")
    ap.add_argument("--no-verify", action="store_true",
                    help="skip the post-install ctypes load check")
    ap.add_argument("--no-clean", action="store_true",
                    help="keep llama/ggml libraries from a previous build "
                         "(by default they are removed so backends stay consistent)")
    ap.add_argument("--dry-run", action="store_true",
                    help="print the commands without executing them")
    args = ap.parse_args()

    repo_root = Path(__file__).resolve().parent.parent
    system = sys.platform
    backend = args.backend
    if backend == "auto":
        backend = "metal" if system == "darwin" else ("dml" if system == "win32" else "cpu")

    install_dir = args.install_dir or repo_root / "inference" / "qwen3asr_dml" / "bin"
    source_dir = args.source_dir or Path.home() / "llama-cpp" / f"{args.tag}"
    build_dir = args.build_dir or source_dir / f"build-{args.tag}"
    jobs = args.jobs or (os.cpu_count() or 4)

    print(f"llama.cpp {args.tag} -> {system} / backend={backend}")
    print(f"  source : {source_dir}")
    print(f"  build  : {build_dir}")
    print(f"  install: {install_dir}")

    cmake = find_cmake()

    # 1. fetch -------------------------------------------------------------
    if args.skip_fetch or (source_dir / "CMakeLists.txt").is_file():
        print(f"[1/4] 复用已有源码: {source_dir}")
    else:
        print(f"[1/4] 克隆 {REPO_URL} @{args.tag}")
        source_dir.parent.mkdir(parents=True, exist_ok=True)
        run(["git", "clone", "--depth", "1", "--branch", args.tag,
             REPO_URL, str(source_dir)], dry_run=args.dry_run)

    # 2. configure ---------------------------------------------------------
    print("[2/4] cmake configure")
    cfg = [cmake, "-S", str(source_dir), "-B", str(build_dir)]
    cfg += pick_generator(system, jobs)
    cfg += cmake_configure_args(system, backend, install_dir)
    run(cfg, dry_run=args.dry_run)

    # 3. build -------------------------------------------------------------
    print("[3/4] 编译 llama 库")
    build = [cmake, "--build", str(build_dir), "--target", "llama",
             "--config", "Release"]
    if shutil.which("ninja"):
        build += ["--", "-j", str(jobs)]
    else:
        build += ["--", "-j", str(jobs)]
    run(build, dry_run=args.dry_run)

    # 4. install -----------------------------------------------------------
    print("[4/4] 安装到运行时目录")
    if not args.dry_run:
        install_dir.mkdir(parents=True, exist_ok=True)
        if not args.no_backup and any(install_dir.iterdir()):
            backup = install_dir.parent / f"bin.backup-{args.tag}"
            if backup.exists():
                shutil.rmtree(backup)
            shutil.copytree(install_dir, backup)
            print(f"  旧库已备份到 {backup}")
        if not args.no_clean:
            purge_stale_libraries(install_dir, args.dry_run)
    names = install_libraries(build_dir, install_dir, system, backend, args.dry_run)
    print(f"  已安装 {len(names)} 个文件")

    if not args.no_verify and not args.dry_run and names:
        verify(install_dir, system)

    print("\n完成。llama.py 无需修改，直接使用上述库。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
