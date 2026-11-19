"""Launch training with the configured Intel Arc environment."""
import subprocess
import sys
from pathlib import Path


def main():
    script = Path(__file__).resolve()
    project = script.parents[1]
    config = project / '.xpu-python'
    if config.is_file():
        preferred_python = Path(config.read_text(encoding='utf-8').strip()).resolve()
        if not preferred_python.is_file():
            raise SystemExit(f'Configured GPU interpreter is missing: {preferred_python}')
        if Path(sys.executable).resolve() != preferred_python:
            return subprocess.call([str(preferred_python), '-E', str(script), *sys.argv[1:]])
        # Keep Torch and its DLLs from the GPU environment. Fall back to the
        # project environment for application packages missing there.
        sys.path.append(str(project / '.venv' / 'Lib' / 'site-packages'))

    from pipeline_cli import train_main

    train_main()
    return 0

if __name__ == '__main__':
    raise SystemExit(main())
