from dataclasses import dataclass
import numpy as np
from qiskit import QuantumCircuit
from qiskit_aer.primitives import SamplerV2 as Sampler

@dataclass
class BB84Results:
    """Data class to hold results of a BB84 protocol run."""
    alice_bits: np.ndarray
    alice_bases: np.ndarray
    bob_bases: np.ndarray
    bob_bits: np.ndarray
    sifted_indices: np.ndarray
    sample_indices: np.ndarray
    qber: float
    num_sampled_bits: int
    alice_key_final: np.ndarray
    bob_key_final: np.ndarray
    key_length: int
    match_count: int
    fidelity: float


class Alice:
    """
    Alice generates random bits and bases, prepares qubits accordingly, and sends them to Bob.
    
    Encoding scheme:

    Basis   Bit   State
    -----  -----  -----
      Z      0      |0>
      Z      1      |1> (applies X to |0>)
      X      0      |+> (applies H to |0>)
      X      1      |-> (applies X and H to |0>)

    """
    def __init__(self, num_bits, rng):
        self.num_bits = num_bits
        self.rng = rng
        self.bits = rng.integers(0, 2, size=num_bits)  # Random bits (0 or 1)
        self.bases = rng.integers(0, 2, size=num_bits)  # Random bases (0 for Z, 1 for X)
    
    def encode_qubits(self):
        """Alice prepares qubits encoding the bits in the given bases."""
        circuits = []
        for bit, basis in zip(self.bits, self.bases):
            circuit = QuantumCircuit(1, 1)
            if bit == 1:
                circuit.x(0)  # Encode bit value
            if basis == 1:
                circuit.h(0)  # Rotate to correct basis
            circuits.append(circuit)
        return circuits

class Bob:
    """
    Bob receives qubits from Alice and measures them in randomly chosen bases.
    
    """
    def __init__(self, num_bits, rng):
        self.num_bits = num_bits
        self.rng = rng
        self.bases = rng.integers(0, 2, size=num_bits)  # Random bases (0 for Z, 1 for X)
    
    def measure_qubits(self, circuits, channel):
        """Bob measures the received qubits in his randomly chosen bases using the specified channel."""
        measurements = []
        for qc, basis in zip(circuits, self.bases):
            qc_bob = qc.copy()
            if basis == 1:  # X basis
                qc_bob.h(0)  # Rotate to X basis before measurement
            qc_bob.measure(0, 0)
            measurements.append(qc_bob)

        job = channel.simulator.run(measurements, shots=1)
        results = job.result()
        
        return np.array([
            int(list(res.keys())[0])
            for res in results.get_counts()])

def sift_indices(alice, bob):
    """Alice and Bob compare bases."""
    return np.where(alice.bases == bob.bases)[0]

def estimate_qber(alice_key, bob_key, sample_fraction, rng):
    """Alice and Bob estimate QBER by comparing a random subset of their sifted bits."""
    n = len(alice_key)
    sample_size = max(1, int(sample_fraction * n))
    sample_indices = rng.choice(n, size=sample_size, replace=False)
    qber = np.mean(alice_key[sample_indices] != bob_key[sample_indices])
    return float(qber), sample_indices
   
def run_bb84(num_bits, channel, sample_qber_fraction, rng):
    """Run the full BB84 protocol."""

    alice = Alice(num_bits, rng)
    bob = Bob(num_bits, rng)

    # Step 1: Alice prepares qubits and sends to Bob through the specified channel
    qubits = alice.encode_qubits()
    transmitted_qubits = channel.transmit(qubits)

    # Step 2: Bob measures received qubits
    bob_results = bob.measure_qubits(transmitted_qubits, channel)

    # Step 3: Compare bases and sift keys
    sifted_indices = sift_indices(alice, bob)
    alice_key = alice.bits[sifted_indices]
    bob_key = bob_results[sifted_indices]

    # Step 4: Estimate QBER
    qber, sample_indices = estimate_qber(alice_key, bob_key, sample_qber_fraction, rng)

    # Remove sampled bits from final key
    alice_key_final = np.delete(alice_key, sample_indices)
    bob_key_final = np.delete(bob_key, sample_indices)

    key_length = len(alice_key_final)
    match_count = np.sum(alice_key_final == bob_key_final)
    fidelity = match_count / key_length if key_length > 0 else 0


    return BB84Results(
        alice_bits=alice.bits,
        alice_bases=alice.bases,
        bob_bases=bob.bases,
        bob_bits=bob_results,
        sifted_indices=sifted_indices,
        sample_indices=sample_indices,
        qber=qber,
        num_sampled_bits=len(sample_indices),
        alice_key_final=alice_key_final,
        bob_key_final=bob_key_final,
        key_length=key_length,
        match_count=match_count,
        fidelity=fidelity)

