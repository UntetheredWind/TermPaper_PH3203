# Noisy Superdense Coding (Qiskit)

This project simulates superdense coding with a **single-qubit depolarizing channel** acting only during transmission of Alice's qubit to Bob.

## What it computes

- Overall decoding success probability versus depolarizing parameter `p`
- Empirical mutual information `I(X;Y)` (bits/use) as a capacity proxy
- Publication-quality figure and CSV export under `results/`

## 1) Create and use virtual environment (Windows PowerShell)

```powershell
Set-Location d:\TermPaper_veblu
C:/Users/swast/AppData/Local/Microsoft/WindowsApps/python3.12.exe -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m ipykernel install --user --name termpaper-veblu --display-name "Python (.venv TermPaper_veblu)"
```

## 2) Run as notebook-style script

In VS Code, open `superdense_coding_noisy_notebook.py` and run cell-by-cell (`# %%` cells), selecting kernel:

`Python (.venv TermPaper_veblu)`

Or run end-to-end in terminal:

```powershell
Set-Location d:\TermPaper_veblu
.\.venv\Scripts\python.exe .\superdense_coding_noisy_notebook.py
```

## Outputs

- `results/superdense_noisy_metrics.png`
- `results/superdense_noisy_metrics.csv`

## Teleportation thermal-relaxation study

Run the new notebook-style teleportation script end-to-end:

```powershell
Set-Location d:\TermPaper_veblu
.\.venv\Scripts\python.exe .\teleportation_thermal_relaxation_notebook.py
```

Generated outputs:

- `results/teleportation_thermal_relaxation_fidelity.png`
- `results/teleportation_thermal_relaxation_metrics.csv`

The script teleports `|psi> = (|0> + |1>)/sqrt(2)`, applies thermal relaxation on
Bob's pre-correction delay marker with a swept timing scale, reconstructs Bob's
single-qubit density matrix from `X/Y/Z` expectation values, and reports
fidelity to the ideal target with a classical-limit guide line at `2/3`.

## Notes

- Shots are set to `8192` (as requested).
- Sweep uses `p in [0, 1]` with 101 evenly spaced values.
- Pauli encoding uses `Z^b1 X^b0` on Alice's qubit to generate the four Bell states.
