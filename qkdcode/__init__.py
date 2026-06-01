"""

qkdcode - BB84 QKD protocol implementations with eavesdropping and noise models.

    This package provides a modular implementation of the BB84 quantum key distribution protocol, including:
    - Alice and Bob classes representing protocol participants.
    - QuantumChannel class with optional noise models.
    - Eavesdropping attack classes for two interception-resend attacks.

The code is structured to allow extension to other protocols, noise models, and attack strategies in the future.
Future updates will include E91 protocol implementation with entanglement-based eavesdropping attacks.

Typical usage
-------------
    from qkdcode import Alice, Bob, QuantumChannel, run_bb84
    from qkdcode import InterceptResendEve, BreidtBartEve

Modules
-------
    channel         Quantum channel and noise models
    bb84            BB84 protocol — Alice, Bob, sifting, QBER
    eavesdropping   Eve attack models

"""

__version__ = "0.1.0"
__author__  = "NicholasFulford"

from qkdcode.channel       import QuantumChannel
from qkdcode.bb84          import Alice, Bob, sift_indices, estimate_qber, run_bb84
from qkdcode.eavesdropping import RandomInterceptResendEve, BreidbartEve

