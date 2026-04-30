# %% [markdown]
# # Noisy Superdense Coding with Transmission-Only Depolarizing Channel
#
# This notebook-style Python script simulates superdense coding in the presence
# of realistic single-qubit depolarizing noise acting only during transmission
# of Alice's qubit to Bob.
#
# It computes:
# 1) Overall success probability of correctly decoding the intended 2-bit message
# 2) Empirical mutual information I(X;Y) (bits/use) as a capacity proxy

# %%
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
from qiskit import ClassicalRegister, QuantumCircuit, QuantumRegister
from qiskit_aer import AerSimulator
from qiskit_aer.noise import NoiseModel, depolarizing_error


# %%
# Simulation controls
SHOTS = 8192
N_P_POINTS = 101
P_VALUES = np.linspace(0.0, 1.0, N_P_POINTS)
SEED_SIMULATOR = 314159
SEED_TRANSPILE = 271828

# Message map: Alice applies Z^b1 X^b0 to encode two classical bits b1b0.
MESSAGES: List[str] = ["00", "01", "10", "11"]


@dataclass(frozen=True)
class SweepResult:
    p_values: np.ndarray
    success_probability: np.ndarray
    mutual_information_bits: np.ndarray


# %%
def build_superdense_circuit(message_bits: str) -> QuantumCircuit:
    """Build a single superdense coding circuit for one 2-bit message.

    The circuit includes an explicit identity gate on Alice's qubit to represent
    the physical transmission stage. Noise will be attached only to this gate.
    """
    if message_bits not in MESSAGES:
        raise ValueError(f"Unknown message '{message_bits}'. Choose from {MESSAGES}.")

    q = QuantumRegister(2, "q")
    c = ClassicalRegister(2, "c")
    qc = QuantumCircuit(q, c, name=f"msg_{message_bits}")

    # Create Bell pair |Phi+> shared between Alice (q0) and Bob (q1).
    qc.h(q[0])
    qc.cx(q[0], q[1])

    # Standard Pauli encoding: Z^b1 X^b0 on Alice's qubit.
    b1, b0 = int(message_bits[0]), int(message_bits[1])
    if b1 == 1:
        qc.z(q[0])
    if b0 == 1:
        qc.x(q[0])

    # Dedicated transmission marker (noise attached only here).
    qc.id(q[0])

    # Bob decodes in Bell basis.
    qc.cx(q[0], q[1])
    qc.h(q[0])

    # Measure both qubits. Qiskit bitstrings are returned as c1c0.
    qc.measure(q, c)
    return qc


# %%
def build_noise_model_transmission_only(p: float) -> NoiseModel:
    """Create a noise model with depolarizing noise only on Alice's transmission."""
    if p < 0.0 or p > 1.0:
        raise ValueError("Depolarizing parameter p must be in [0, 1].")

    noise_model = NoiseModel()
    error = depolarizing_error(p, 1)

    # Apply error only to the id gate used as transmission marker on qubit 0.
    noise_model.add_quantum_error(error, ["id"], [0])
    return noise_model


# %%
def infer_expected_outputs_noiseless() -> Dict[str, str]:
    """Infer the ideal output bitstring for each message at p=0.

    This avoids endianness/convention mistakes and gives a robust correctness map.
    """
    sim = AerSimulator(seed_simulator=SEED_SIMULATOR)
    circuits = [build_superdense_circuit(m) for m in MESSAGES]
    result = sim.run(circuits, shots=SHOTS).result()

    mapping: Dict[str, str] = {}
    for msg, circ in zip(MESSAGES, circuits):
        counts = result.get_counts(circ)
        # Most frequent output in the noiseless setting is the expected decode.
        mapping[msg] = max(counts, key=counts.get)
    return mapping


# %%
def build_message_circuits_once() -> List[QuantumCircuit]:
    """Build the four message circuits once for reuse across p-sweep."""
    return [build_superdense_circuit(m) for m in MESSAGES]


# %%
def save_superdense_circuit_diagram() -> None:
    """Save a clean Matplotlib circuit diagram for publication/report use."""
    circuit = build_superdense_circuit("11")
    fig = circuit.draw(
        "mpl",
        filename="superdense_circuit.png",
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
def run_sweep(p_values: np.ndarray, shots: int = SHOTS) -> SweepResult:
    expected_output = infer_expected_outputs_noiseless()
    circuits = build_message_circuits_once()

    success_vals: List[float] = []
    mutual_info_vals: List[float] = []

    for idx, p in enumerate(p_values):
        sim = AerSimulator(
            noise_model=build_noise_model_transmission_only(float(p)),
            seed_simulator=SEED_SIMULATOR,
        )
        result = sim.run(circuits, shots=shots).result()

        # Conditional distribution P(y|x), with uniform inputs P(x)=1/4.
        p_y_given_x: Dict[str, Dict[str, float]] = {m: {} for m in MESSAGES}

        # Success probability averaged over equiprobable messages.
        per_message_success = []

        for msg, circ in zip(MESSAGES, circuits):
            counts = result.get_counts(circ)
            total = sum(counts.values())

            for out, n in counts.items():
                p_y_given_x[msg][out] = n / total

            correct = counts.get(expected_output[msg], 0) / total
            per_message_success.append(correct)

        p_success = float(np.mean(per_message_success))
        success_vals.append(p_success)

        # Mutual information I(X;Y) = sum_x,y P(x)P(y|x) log2(P(y|x)/P(y)).
        p_x = 1.0 / len(MESSAGES)
        all_outputs = sorted({out for m in MESSAGES for out in p_y_given_x[m].keys()})

        p_y: Dict[str, float] = {
            out: sum(p_x * p_y_given_x[m].get(out, 0.0) for m in MESSAGES)
            for out in all_outputs
        }

        mi = 0.0
        for m in MESSAGES:
            for out in all_outputs:
                pyx = p_y_given_x[m].get(out, 0.0)
                py = p_y[out]
                if pyx > 0.0 and py > 0.0:
                    mi += p_x * pyx * np.log2(pyx / py)

        mutual_info_vals.append(float(mi))

        if (idx + 1) % 10 == 0 or (idx + 1) == len(p_values):
            print(f"Completed {idx + 1}/{len(p_values)} p-points")

    return SweepResult(
        p_values=np.array(p_values, dtype=float),
        success_probability=np.array(success_vals, dtype=float),
        mutual_information_bits=np.array(mutual_info_vals, dtype=float),
    )


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

    fig, ax1 = plt.subplots(figsize=(9.2, 5.6))

    # Primary axis: success probability
    line1 = ax1.plot(
        result.p_values,
        result.success_probability,
        color="#0B4F6C",
        linewidth=2.4,
        label="Success probability",
    )[0]
    ax1.set_xlabel("Depolarizing probability p")
    ax1.set_ylabel("Success probability", color="#0B4F6C")
    ax1.tick_params(axis="y", labelcolor="#0B4F6C")
    ax1.set_xlim(0.0, 1.0)
    ax1.set_ylim(0.0, 1.05)

    # Secondary axis: empirical mutual information (bits/use)
    ax2 = ax1.twinx()
    line2 = ax2.plot(
        result.p_values,
        result.mutual_information_bits,
        color="#E36414",
        linewidth=2.2,
        linestyle="--",
        label="Empirical I(X;Y)",
    )[0]
    ax2.set_ylabel("Empirical mutual information I(X;Y) [bits/use]", color="#E36414")
    ax2.tick_params(axis="y", labelcolor="#E36414")
    ax2.set_ylim(0.0, 2.05)

    ax1.set_title("Noisy Superdense Coding: Transmission-Only Depolarizing Channel")
    ax1.grid(True, which="major", alpha=0.28)
    ax1.grid(True, which="minor", alpha=0.12)
    ax1.minorticks_on()

    # Unified legend across both axes
    handles = [line1, line2]
    labels = [h.get_label() for h in handles]
    ax1.legend(handles, labels, loc="upper right", frameon=True, framealpha=0.95)

    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    if show_plot:
        plt.show()
    else:
        plt.close(fig)


# %%
def save_data_csv(result: SweepResult, out_csv: Path) -> None:
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    arr = np.column_stack(
        [result.p_values, result.success_probability, result.mutual_information_bits]
    )
    header = "p,success_probability,mutual_information_bits"
    np.savetxt(out_csv, arr, delimiter=",", header=header, comments="")


# %%
# Run the full sweep and generate output artifacts.
results_dir = Path("results")
plot_path = results_dir / "superdense_noisy_metrics.png"
csv_path = results_dir / "superdense_noisy_metrics.csv"

save_superdense_circuit_diagram()
result = run_sweep(P_VALUES, shots=SHOTS)
save_data_csv(result, csv_path)
make_report_plot(result, plot_path, show_plot=False)

print(f"Saved circuit diagram to: {(Path('superdense_circuit.png')).resolve()}")
print(f"Saved plot to: {plot_path.resolve()}")
print(f"Saved CSV  to: {csv_path.resolve()}")
print(
    "Endpoint summary: "
    f"p=0 -> success={result.success_probability[0]:.4f}, I={result.mutual_information_bits[0]:.4f} bits; "
    f"p=1 -> success={result.success_probability[-1]:.4f}, I={result.mutual_information_bits[-1]:.4f} bits"
)
