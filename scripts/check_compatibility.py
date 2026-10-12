"""Run locked tests in isolated environments using explicitly installed Pythons."""

import argparse
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import subprocess
import tempfile


ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--uv", default="uv")
    parser.add_argument("--jobs", type=int, default=2)
    parser.add_argument("versions", nargs="+", help="Python versions to check")
    args = parser.parse_args()
    if args.jobs < 1:
        parser.error("--jobs must be positive")
    with tempfile.TemporaryDirectory(prefix="coding-theory-compatibility-") as directory:
        requirements = Path(directory) / "requirements.txt"
        subprocess.run([args.uv, "export", "--locked", "--all-groups", "--no-emit-project", "--output-file", str(requirements)], cwd=ROOT, check=True, capture_output=True, text=True)

        def check(version):
            print(f"Checking Python {version}", flush=True)
            statement = "import platform, numpy, pytest; print('Python', platform.python_version(), 'NumPy', numpy.__version__, 'pytest', pytest.__version__, flush=True); raise SystemExit(pytest.main(['-q']))"
            command = [args.uv, "run", "--isolated", "--no-project", "--no-python-downloads", "--python", version, "--with-requirements", str(requirements), "python", "-c", statement]
            result = subprocess.run(command, cwd=ROOT, check=False)
            print(f"Python {version}: {'PASS' if result.returncode == 0 else 'FAIL'}", flush=True)
            return result.returncode == 0

        with ThreadPoolExecutor(max_workers=args.jobs) as pool:
            outcomes = list(pool.map(check, args.versions))
    return 0 if all(outcomes) else 1


if __name__ == "__main__":
    raise SystemExit(main())
