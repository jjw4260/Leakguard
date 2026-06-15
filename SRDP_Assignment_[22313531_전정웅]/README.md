# SRDP Assignment Source Code

Student ID: 22313531
Name: Jeon Jeongwoong

## Files

- `src/pirate_full_experiment.py`: full PIRATE / Shadow-RAG experiment code reflected from the existing experiment.
- `src/srdp_full_safe_framework.py`: assignment-safe executable framework runner.
- `src/srdp_gui.py`: PySide6 desktop GUI runner for Windows and Linux.
- `src/safe_srdp_working_demo.py`: lightweight demo runner.
- `outputs/`: sample output CSV files.

## Install

```bash
pip install -r requirements.txt
```

## Run GUI

Windows:

```bat
run.bat
```

Linux:

```bash
bash run.sh
```

The GUI lets the user configure epochs, trials, seeds, memory policies, model parameters, and output paths. It writes summary, seed-level, and row-level CSV files to the selected output folder.

## Run CLI

```bash
python src/srdp_full_safe_framework.py --epochs 6 --trials 10 --out-prefix outputs/srdp_assignment
```

Or use `run_cli.bat` on Windows and `bash run_cli.sh` on Linux.

## Build GUI executable

Windows:

```bat
build_windows.bat
```

Linux:

```bash
bash build_linux.sh
```

Build artifacts are written under `dist/` by PyInstaller.
