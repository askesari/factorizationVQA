"""
This module contains the the classes related to mitigation
Author(s): Amit S. Kesari.
"""
from mthree import M3Mitigation
from mthree.utils import counts_to_vector, vector_to_quasiprobs, expval_and_stddev, expval
from mthree.classes import QuasiCollection, ProbDistribution
import numpy as np
from logconfig import get_logger

## initialize logger
log = get_logger(__name__)

class M3MitigationDerived(M3Mitigation):
    """
    This class enables in setting up the mitigation calibration counts.
    """

    def __init__(self, backend, num_qubits=None):
        """
        Description: initialize the base class and setup calibration counts
        """
        super().__init__(system=backend)
        ## Identify the maximum number of qubits supported by the backend to ensure
        ## that the calibration is not done beyond that; otherwise it will results in
        ## an exception
        max_num_qubits = backend.configuration().n_qubits
        if num_qubits is not None:
            try:
                if num_qubits > max_num_qubits:
                    raise Exception(f"Number of qubits {num_qubits} greater than allowed number of qubits {max_num_qubits} for the device {backend}")
                else:
                    self.num_qubits = num_qubits
            except Exception as e:
                log.exception(e, stack_info=True)
                raise

        self.qubit_list = np.asarray(range(self.num_qubits))
        log.info(f"Qubit List for calibration: {self.qubit_list}")

        # Setup calibration data
        self.cals_from_system(self.qubit_list)
    
    def apply_m3_mitigation(self, counts):
        """
        Description: Apply mitigation to the counts and return a Quasi-collection i.e. collection
        of quasi-probabilities
        Input Arguments:
        counts (dict, list): Input counts dict or list of dicts.
        qubits (array_like): Qubits over which to correct calibration data. Default is all.
        """
        log.info(f"Inside apply_mitigation method")
        qubits = self.qubit_list

        ## m3 mitigation returns quasi-probabilities, whereas we need a probability distribution.
        ## For this, we use an existing method "nearest_probability_distribution".
        quasis = self.apply_correction(counts, qubits=qubits)
        prob_data = quasis.nearest_probability_distribution()

        return(prob_data)
    
    @classmethod
    def get_expval(cls, hamiltonian, data_distr, mitigation_applied=False):
        """ 
        Description: Compute the expectation value and standard deviation with the given 
                     probability distribution. The measurement operators would be part of the 
                     Hamiltonian along with its coefficients
        """
        log.info("Start of expval i.e. expectation computation ...")

        # step 1: identify the measurement operators and the corresponding coefficients
        log.info(f"Extracting pauli operator strings and coefficients from qubit hamiltonian ...")
        pauli_op_list = hamiltonian.primitive.to_list()
        op_list = [item[0] for item in pauli_op_list]
        coeffs = np.array([np.real(item[1]) for item in pauli_op_list], dtype=float)
        log.debug(f"Operator list: {op_list}")
        log.debug(f"Coefficient array: {coeffs}")

        # step 2: compute expectation
        if mitigation_applied == False:
            log.info(f"Computing expectation for non-mitigated data ...")
            computed_expval = np.sum(coeffs*ProbDistribution(data_distr).expval(exp_ops=op_list))
        else:
            log.info(f"Computing expectation for mitigated data ...")
            computed_expval = np.sum(coeffs*data_distr.expval(exp_ops=op_list))
            
        
        return(computed_expval)