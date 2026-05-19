from qiskit_aer import AerSimulator
from qiskit_aer.noise import NoiseModel, depolarizing_error, phase_damping_error, amplitude_damping_error
from qiskit import transpile

"""
Quantum channel models for QKD simulation.

Provides three physically motivated noise channels:
- Depolarising
- Phase Damping
- Amplitude Damping 
These can be applied individually or composed into a composite channel. 
All channels are implemented as Qiskit Aer noise models applied to single-qubit circuits.

"""

def build_depolarising_model(p):
    """
    Depolarising noise model with error rate p applied to all single-qubit gates.
    At each gate, with probability p, one of {X, Y, Z} is applied uniformly at random.
    
    """
    if not 0 <= p <= 1:
        raise ValueError("Depolarising error rate p must be in [0, 1]")
    
    noise_model = NoiseModel()
    error = depolarizing_error(p, 1)
    noise_model.add_all_qubit_quantum_error(error, ['h', 'x', 'z', 'id'])
    return noise_model

def build_phase_damping_model(p):
    """
    Phase damping (phase-flip) noise model with error rate p applied to all single-qubit gates.
    At each gate, applies Z with probability p.
    
    """
    if not 0 <= p <= 1:
        raise ValueError("Phase damping error rate p must be in [0, 1]")
    
    noise_model = NoiseModel()
    error = phase_damping_error(p)
    noise_model.add_all_qubit_quantum_error(error, ['h', 'x', 'z', 'id'])
    return noise_model

def build_amplitude_damping_model(gamma):
    """
    Amplitude damping noise model with decay parameter gamma applied to all single-qubit gates.
    State |1> decays to |0> with probability gamma.
    
    """
    if not 0 <= gamma <= 1:
        raise ValueError("Decay parameter must be in [0, 1]")

    noise_model = NoiseModel()
    error = amplitude_damping_error(gamma)
    noise_model.add_all_qubit_quantum_error(error, ['h', 'x', 'z', 'id'])
    return noise_model


class QuantumChannel:
    """
    Model for quantum channel between Alice and Bob with configurable noise.
    Applies specified noise model to list of single-qubit circuits representing the physically transmitted qubits.
    
    """

    def __init__(self, noise_model=None, seed=42):
        self.noise_model = noise_model
        self.simulator = AerSimulator(noise_model=noise_model, seed_simulator=seed)
    
    def transmit(self, circuits):
        """
        Simulate transmission of qubits through the channel.
        
        For a noiseless channel, returns copies of the input circuits
        unchanged. For a noisy channel, the noise model is applied
        by AerSimulator during any subsequent execution.

        Note this method does not run the circuits. It returns them
        ready for execution when measured. The noise model
        is applied at execution time.
        
        """
        return [transpile(qc, self.simulator) for qc in circuits]
    
    @property
    def is_noiseless(self):
        """True if no noise model is applied."""
        return self.noise_model is None

    @classmethod
    def noiseless(cls):
        """Factory method for noiseless channel."""
        return cls(noise_model=None)
    
    @classmethod
    def depolarising(cls, p, seed=42):
        """Factory method for depolarising channel."""
        return cls(build_depolarising_model(p), seed=seed)
    
    @classmethod
    def phase_damping(cls, p, seed=42):
        """Factory method for phase damping channel."""
        return cls(build_phase_damping_model(p), seed=seed)
    
    @classmethod
    def amplitude_damping(cls, gamma, seed=42):
        """Factory method for amplitude damping channel."""
        return cls(build_amplitude_damping_model(gamma), seed=seed) 
    
