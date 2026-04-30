# %% [markdown]
# # Quantum Teleportation Under Thermal Relaxation
#
# This notebook-style Python script simulates teleportation of
# |psi> = (|0> + |1>)/sqrt(2) while applying thermal-relaxation decoherence to
# Bob's qubit before the correction stage. For each decoherence scale factor,
# it performs single-qubit tomography-style reconstruction and computes fidelity
# against the ideal target state.

# %%
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
from qiskit import ClassicalRegister, QuantumCircuit, QuantumRegister, transpile
from qiskit.quantum_info import DensityMatrix, Statevector, state_fidelity
from qiskit_aer import AerSimulator
from qiskit_aer.noise import NoiseModel, thermal_relaxation_error


# %%
# Simulation controls
SHOTS = 8192
SEED_SIMULATOR = 314159
SEED_TRANSPILE = 271828

# Physical constants (representative fixed backend-like values)
T1_US = 80.0
T2_US = 60.0
BASE_DELAY_NS = 500.0

# Decoherence knob: total delay = BASE_DELAY_NS * scale_factor
SCALE_FACTORS = np.linspace(0.0, 200.0, 81)

T1_S = T1_US * 1e-6
T2_S = T2_US * 1e-6
TARGET_STATE = Statevector([1 / np.sqrt(2), 1 / np.sqrt(2)])
TARGET_DM = DensityMatrix(TARGET_STATE)

BASIS_LABELS: List[str] = ["Z", "X", "Y"]


@dataclass(frozen=True)
class SweepResult:
    scale_factors: np.ndarray
    delay_ns: np.ndarray
    bloch_x: np.ndarray
    bloch_y: np.ndarray
    bloch_z: np.ndarray
    fidelity: np.ndarray


# %%
def validate_physical_parameters(t1_s: float, t2_s: float, delay_ns: float) -> None:
    if t1_s <= 0.0 or t2_s <= 0.0:
        raise ValueError("T1 and T2 must be strictly positive.")
    if t2_s > 2.0 * t1_s:
        raise ValueError("Physical constraint violated: T2 must satisfy T2 <= 2*T1.")
    if delay_ns < 0.0:
        raise ValueError("Delay duration must be nonnegative.")


# %%
def build_teleportation_tomography_circuit(basis: str, delay_ns: float) -> QuantumCircuit:
    """Build a coherent teleportation circuit with Bob delay before correction.

    Deferred measurement equivalence is used for corrections:
    Bell-measurement-conditioned Pauli corrections are implemented as coherent
    controlled gates CNOT(q1->q2) and CZ(q0->q2).
    """
    if basis not in BASIS_LABELS:
        raise ValueError(f"Unknown basis '{basis}'. Choose from {BASIS_LABELS}.")

    validate_physical_parameters(T1_S, T2_S, delay_ns)

    q = QuantumRegister(3, "q")
    c = ClassicalRegister(1, "c")
    qc = QuantumCircuit(q, c, name=f"teleport_{basis}_delay_{int(round(delay_ns))}ns")

    # Input state |psi> = |+> on Alice's source qubit q0.
    qc.h(q[0])

    # Shared Bell pair between q1 (Alice) and q2 (Bob).
    qc.h(q[1])
    qc.cx(q[1], q[2])

    # Bell-basis transform on Alice's two qubits.
    qc.cx(q[0], q[1])
    qc.h(q[0])

    # Dedicated delay marker where thermal relaxation acts on Bob before correction.
    qc.delay(int(round(delay_ns)), q[2], unit="ns")

    # Coherent correction stage equivalent to classically conditioned X/Z.
    qc.cx(q[1], q[2])
    qc.cz(q[0], q[2])

    # Tomography setting on Bob's qubit.
    if basis == "X":
        qc.h(q[2])
    elif basis == "Y":
        qc.sdg(q[2])
        qc.h(q[2])

    qc.measure(q[2], c[0])
    return qc


# %%
def build_noise_model_for_delay(delay_ns: float) -> NoiseModel:
    validate_physical_parameters(T1_S, T2_S, delay_ns)

    noise_model = NoiseModel()
    delay_s = delay_ns * 1e-9
    delay_error = thermal_relaxation_error(T1_S, T2_S, delay_s)

    # Apply thermal relaxation only to Bob's explicit delay marker.
    noise_model.add_quantum_error(delay_error, ["delay"], [2])
    return noise_model


# %%
def expectation_from_counts(counts: Dict[str, int]) -> float:
    n0 = counts.get("0", 0)
    n1 = counts.get("1", 0)
    total = n0 + n1
    if total == 0:
        raise RuntimeError("Counts are empty; cannot estimate expectation value.")
    return (n0 - n1) / total


# %%
def reconstruct_density_matrix(ex: float, ey: float, ez: float) -> DensityMatrix:
    i2 = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=complex)
    x = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=complex)
    y = np.array([[0.0, -1j], [1j, 0.0]], dtype=complex)
    z = np.array([[1.0, 0.0], [0.0, -1.0]], dtype=complex)

    rho = 0.5 * (i2 + ex * x + ey * y + ez * z)

    # Numerical hygiene for finite-shot inversion.
    rho = 0.5 * (rho + rho.conj().T)
    tr = np.trace(rho)
    if abs(tr) > 0.0:
        rho = rho / tr

    eigvals, eigvecs = np.linalg.eigh(rho)
    eigvals = np.clip(eigvals, 0.0, None)
    if eigvals.sum() <= 0.0:
        eigvals = np.array([1.0, 0.0])
    eigvals = eigvals / eigvals.sum()
    rho_psd = eigvecs @ np.diag(eigvals) @ eigvecs.conj().T

    return DensityMatrix(rho_psd)


# %%
def save_teleportation_circuit_diagram() -> None:
    """Save a clean Matplotlib teleportation circuit diagram for publication use."""
    circuit = build_teleportation_tomography_circuit("Z", delay_ns=0.0)
    fig = circuit.draw(
        "mpl",
        filename="teleportation_circuit.png",
        fold=-1,
        idle_wires=False,
        scale=0.95,
        style={
            "name": "bw",
            "fontsize": 12,
            "dpi": 300,
            "subfontsize": 10,
        },
    )
    plt.close(fig)


# %%
def run_sweep(scale_factors: np.ndarray, shots: int = SHOTS) -> SweepResult:
    fidelity_vals: List[float] = []
    x_vals: List[float] = []
    y_vals: List[float] = []
    z_vals: List[float] = []
    delay_vals: List[float] = []

    for idx, scale in enumerate(scale_factors):
        delay_ns = BASE_DELAY_NS * float(scale)
        noise_model = build_noise_model_for_delay(delay_ns)

        circuits = [build_teleportation_tomography_circuit(b, delay_ns) for b in BASIS_LABELS]
        sim = AerSimulator(
            noise_model=noise_model,
            seed_simulator=SEED_SIMULATOR,
        )
        transpiled = transpile(circuits, sim, seed_transpiler=SEED_TRANSPILE)
        result = sim.run(transpiled, shots=shots).result()

        ex = expectation_from_counts(result.get_counts(transpiled[1]))
        ey = expectation_from_counts(result.get_counts(transpiled[2]))
        ez = expectation_from_counts(result.get_counts(transpiled[0]))

        rho = reconstruct_density_matrix(ex, ey, ez)
        fidelity = float(state_fidelity(rho, TARGET_DM))

        x_vals.append(ex)
        y_vals.append(ey)
        z_vals.append(ez)
        delay_vals.append(delay_ns)
        fidelity_vals.append(fidelity)

        if (idx + 1) % 10 == 0 or (idx + 1) == len(scale_factors):
            print(f"Completed {idx + 1}/{len(scale_factors)} scale points")

    return SweepResult(
        scale_factors=np.array(scale_factors, dtype=float),
        delay_ns=np.array(delay_vals, dtype=float),
        bloch_x=np.array(x_vals, dtype=float),
        bloch_y=np.array(y_vals, dtype=float),
        bloch_z=np.array(z_vals, dtype=float),
        fidelity=np.array(fidelity_vals, dtype=float),
    )


# %%
def save_data_csv(result: SweepResult, out_csv: Path) -> None:
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    arr = np.column_stack(
        [
            result.scale_factors,
            result.delay_ns,
            result.bloch_x,
            result.bloch_y,
            result.bloch_z,
            result.fidelity,
        ]
    )
    header = "scale_factor,delay_ns,bloch_x,bloch_y,bloch_z,fidelity"
    np.savetxt(out_csv, arr, delimiter=",", header=header, comments="")


# %%
def make_report_plot(result: SweepResult, out_path: Path, show_plot: bool = True) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)

    plt.style.use("seaborn-v0_8-whitegrid")
    plt.rcParams.update(
        {
            "font.family": "STIXGeneral",
            "font.size": 12,
            "axes.titlesize": 15,
            "axes.labelsize": 13,
            "legend.fontsize": 11,
            "xtick.labelsize": 11,
            "ytick.labelsize": 11,
            "figure.dpi": 140,
            "savefig.dpi": 320,
            "axes.linewidth": 1.1,
        }
    )

    fig, ax = plt.subplots(figsize=(9.2, 5.6))

    ax.plot(
        result.scale_factors,
        result.fidelity,
        color="#0B4F6C",
        linewidth=2.4,
        label="Teleportation fidelity",
    )
    ax.axhline(
        2.0 / 3.0,
        color="#E36414",
        linestyle="--",
        linewidth=2.0,
        label="Classical limit (2/3)",
    )

    ax.set_xlabel("Decoherence timing scale factor")
    ax.set_ylabel("State fidelity to |psi>")
    ax.set_title("Teleportation Fidelity Under Thermal Relaxation")
    ax.set_xlim(float(result.scale_factors[0]), float(result.scale_factors[-1]))
    ax.set_ylim(0.0, 1.05)
    ax.grid(True, which="major", alpha=0.28)
    ax.grid(True, which="minor", alpha=0.12)
    ax.minorticks_on()
    ax.legend(loc="upper right", frameon=True, framealpha=0.95)

    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    if show_plot:
        plt.show()
    else:
        plt.close(fig)


# %%
def print_verification_summary(result: SweepResult) -> None:
    fidelity0 = float(result.fidelity[0])
    fidelity_end = float(result.fidelity[-1])
    descending_fraction = float(np.mean(np.diff(result.fidelity) <= 0.0)) if len(result.fidelity) > 1 else 1.0

    print(
        "Endpoint summary: "
        f"scale={result.scale_factors[0]:.1f} -> fidelity={fidelity0:.4f}; "
        f"scale={result.scale_factors[-1]:.1f} -> fidelity={fidelity_end:.4f}"
    )
    print(
        "Trend check: "
        f"fraction of non-increasing successive steps = {descending_fraction:.3f}"
    )
    print(
        "Classical threshold check: "
        f"fidelity@min-scale {'>' if fidelity0 > (2.0 / 3.0) else '<='} 2/3, "
        f"fidelity@max-scale {'>' if fidelity_end > (2.0 / 3.0) else '<='} 2/3"
    )


# %%
# Run full analysis and export outputs.
results_dir = Path("results")
plot_path = results_dir / "teleportation_thermal_relaxation_fidelity.png"
csv_path = results_dir / "teleportation_thermal_relaxation_metrics.csv"

save_teleportation_circuit_diagram()
result = run_sweep(SCALE_FACTORS, shots=SHOTS)
save_data_csv(result, csv_path)
make_report_plot(result, plot_path, show_plot=False)
print_verification_summary(result)

print(f"Saved circuit diagram to: {(Path('teleportation_circuit.png')).resolve()}")
print(f"Saved plot to: {plot_path.resolve()}")
print(f"Saved CSV  to: {csv_path.resolve()}")
