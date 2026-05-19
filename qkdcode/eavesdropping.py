from abc import ABC, abstractmethod
from dataclasses import dataclass, field
import numpy as np
from qiskit import QuantumCircuit
from qiskit_aer import AerSimulator
from qkdcode.channel import QuantumChannel

@dataclass
class BB84AttackRecord:
    """
    Data class to hold details of an eavesdropping attack on the BB84 protocol.
    
    """
    eve_measurements: np.ndarray
    eve_bits: np.ndarray
    attack_type: str

class EveAttack(QuantumChannel, ABC):
    """
    Abstract base class for all eavesdropping attacks.
    Inherits from QuantumChannel so that an EveAttack instance can be passed as a channel to the protocol code.
    
    """

    def __init__(self, rng, seed=42):
        super().__init__(noise_model=None, seed=seed)  
        self.rng = rng
        self.attack_record = None  # To store details of the attack for analysis

    @abstractmethod
    def transmit(self, circuits):
        """
        Simulate the eavesdropping attack on the transmitted qubits.
        Must be implemented by subclasses to define specific attack strategies.
        
        Should return the modified circuits to be passed on to Bob, and record details of the attack in self.attack_record.
        
        """
        pass

class RandomInterceptResendEve(EveAttack):
    """
    Intercept-resend eavesdropper using randomly chosen bases.

    For each intercepted qubit, Eve:
        1. Randomly chooses Z or X basis with equal probability.
        2. Measures the qubit in her chosen basis.
        3. Re-encodes her measurement outcome in the same basis.
        4. Forwards the re-encoded qubit to Bob.

    """

    def transmit(self, circuits):
        num_qubits = len(circuits)
        eve_bases = self.rng.integers(0, 2, size=num_qubits)  # Z=0, X=1
        eve_measurements = np.zeros(num_qubits, dtype=int)
        eve_bits = np.zeros(num_qubits, dtype=int)
        resend_qubits = []

        for i, qc in enumerate(circuits):
            qc_eve = qc.copy()
            if eve_bases[i] == 1:  # X basis
                qc_eve.h(0)
            qc_eve.measure(0, 0)
            job = self.simulator.run(qc_eve, shots=1)
            result = job.result()
            measurement = int(list(result.get_counts().keys())[0])
            eve_measurements[i] = measurement
            eve_bits[i] = measurement  # In this simple attack, Eve's bit guess is the same as her measurement

            # Re-encode the qubit based on Eve's measurement
            qc_resend = QuantumCircuit(1, 1)
            if measurement == 1:
                qc_resend.x(0)  # Prepare same bit that Eve measured
            if eve_bases[i] == 1:  # Encode in same basis that Eve measured
                qc_resend.h(0)
            resend_qubits.append(qc_resend)

        self.attack_record = BB84AttackRecord(
            eve_measurements=eve_measurements,
            eve_bits=eve_bits,
            attack_type="Random Intercept-Resend"
        )

        return resend_qubits
        
class BreidbartEve(EveAttack):
    """
    Intercept-resend eavesdropper using the Breidbart basis.

    For each intercepted qubit, Eve:
        1. Measures in the Breidbart basis (22.5° rotated clockwise from Z).
        2. Re-encodes her bit guess in the Breidbart basis.
        3. Forwards the re-encoded qubit to Bob.

    Breidbart basis states:
        |b0⟩ = cos(π/8)|0⟩ + sin(π/8)|1⟩ (halfway between |0⟩ and |+⟩)
        |b1⟩ = - sin(π/8)|0⟩ + cos(π/8)|1⟩ (halfway between |1⟩ and |-⟩)

    To measure in the Breidbart basis, Eve applies Ry(-π/4) before measuring in Z basis:
        Ry(-pi/4)|b0> = |0>  
        Ry(-pi/4)|b1> = |1> 

    If Eve measures |b0⟩, she guesses bit 0. If she measures |b1⟩, she guesses bit 1. 
    This is because |b0⟩ has greater overlap with the states encoding bit 0 (|0⟩ and |+⟩), while |b1⟩ has greater overlap with the states encoding bit 1 (|1⟩ and |-⟩).
    She then re-encodes the qubit in the Breidbart basis according to her guess and forwards it to Bob.

    """

    def transmit(self, circuits):   
        num_qubits = len(circuits)
        eve_measurements = np.zeros(num_qubits, dtype=int)
        eve_bits = np.zeros(num_qubits, dtype=int)
        resend_qubits = []

        for i, qc in enumerate(circuits):
            qc_eve = qc.copy()
            qc_eve.ry(-np.pi/4, 0)  # Rotate to Z basis
            qc_eve.measure(0, 0)
            job = self.simulator.run(qc_eve, shots=1)
            result = job.result()
            measurement = int(list(result.get_counts().keys())[0])
            eve_measurements[i] = measurement
            eve_bits[i] = measurement  

            # Re-encode the qubit in Breidbart basis according to measurement
            qc_resend = QuantumCircuit(1, 1)
            if measurement == 1:
                qc_resend.x(0)  # Prepare same bit that Eve measured
            qc_resend.ry(np.pi/4, 0)  # Rotate to Breidbart basis
            resend_qubits.append(qc_resend)

        self.attack_record = BB84AttackRecord(
            eve_measurements=eve_measurements,
            eve_bits=eve_bits,
            attack_type= "Breidbart Intercept-Resend"
        )

        return resend_qubits
    
def compare_bb84_attacks(alice_bits, Random_attackrecord, Breidbart_attackrecord):
    """
    Compare and return metrics for two different eavesdropping attacks on the BB84 protocol,
    Takes the results of two attack records run on the same alice_bits and returns a dict of metrics for use in the analysis notebook.

    Returns a dictionary with the following keys:
    - random_p_correct     : empirical P(Eve correct) for random attack
    - breidbart_p_correct. : empirical P(Eve correct) for Breidbart attack
    - random_mutual_info   : empirical mutual information (bits)
    - breidbart_mutual_info: empirical mutual information (bits)  

    """

    def p_correct(attack_record):
        return np.mean(alice_bits == attack_record.eve_bits)

    def mutual_info(p):
        if p == 0 or p == 1:
            return 0.0
        return 1 - (-p * np.log2(p) - (1 - p) * np.log2(1 - p)) 
    
    p_random = p_correct(Random_attackrecord)
    p_breidbart = p_correct(Breidbart_attackrecord)

    return {
        'random_p_correct': p_random,
        'breidbart_p_correct': p_breidbart,
        'random_mutual_info': mutual_info(p_random),
        'breidbart_mutual_info': mutual_info(p_breidbart)
    }       


    
