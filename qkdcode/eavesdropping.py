from abc import ABC, abstractmethod
from dataclasses import dataclass
import numpy as np
from qiskit import QuantumCircuit
from qiskit.transpiler import generate_preset_pass_manager
from qiskit_aer.primitives import SamplerV2 as Sampler
from qkdcode.channel import QuantumChannel

@dataclass
class BB84AttackRecord:
    """
    Data class to hold details of an eavesdropping attack on the BB84 protocol.
    
    """
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
        resend_qubits = []
        pm = generate_preset_pass_manager(3, self.simulator)
        sampler = Sampler(options=dict(backend_options=dict(noise_model=self.noise_model)))
        for i, qc in enumerate(circuits):
            qc_eve = qc.copy()
            if eve_bases[i] == 1:  # X basis
                qc_eve.h(0)
            qc_eve.measure(0, 0)
            isa_qc_eve = pm.run(qc_eve)
            job = sampler.run([(isa_qc_eve, None, 1)])
            eve_result = job.result()[0].data.c.get_counts()
            eve_measurement = int(list(eve_result.keys())[0])
            eve_measurements[i] = eve_measurement
            
            # Re-encode the qubit based on Eve's measurement
            qc_resend = QuantumCircuit(1, 1)
            if eve_measurement == 1:
                qc_resend.x(0)  # Prepare same bit that Eve measured
            if eve_bases[i] == 1:  # Encode in same basis that Eve measured
                qc_resend.h(0)
            resend_qubits.append(qc_resend)

        eve_bits = eve_measurements  #Eve's bit guess is the same as her measurement outcome
        
        self.attack_record = BB84AttackRecord(
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
        resend_qubits = []
        pm = generate_preset_pass_manager(3, self.simulator)
        sampler = Sampler(options=dict(backend_options=dict(noise_model=self.noise_model)))
        for i, qc in enumerate(circuits):
            qc_eve = qc.copy()
            qc_eve.ry(-np.pi/4, 0)  # Rotate to Z basis
            qc_eve.measure(0, 0)
            isa_qc_eve = pm.run(qc_eve)
            job = sampler.run([(isa_qc_eve, None, 1)])
            eve_result = job.result()[0].data.c.get_counts()
            eve_measurement = int(list(eve_result.keys())[0])
            eve_measurements[i] = eve_measurement

            # Re-encode the qubit in Breidbart basis according to measurement
            qc_resend = QuantumCircuit(1, 1)
            if eve_measurement == 1:
                qc_resend.x(0)  # Prepare same bit that Eve measured
            qc_resend.ry(np.pi/4, 0)  # Rotate to Breidbart basis        
            resend_qubits.append(qc_resend)

        eve_bits = eve_measurements  #Eve's bit guess is the same as her measurement outcome

        self.attack_record = BB84AttackRecord(
            eve_bits=eve_bits,
            attack_type= "Breidbart Intercept-Resend"
        )

        return resend_qubits

