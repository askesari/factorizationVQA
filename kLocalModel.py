"""
This module implements an k-local Hamiltonian model by converting an algebraic expression consisting of 
binary variables into an k-local Hamiltonian model based on the mapping x_i <=> 0.5(I - Z), where 
x_i is a binary variable, Z is the Pauli-Z operator and I is the identity operator.
"""
# import basic and core python modules
import numpy as np
import math
import matplotlib.pyplot as plt
plt.rcParams['text.usetex'] = True

# import qiskit modules
from qiskit.opflow import PauliSumOp
from qiskit.quantum_info import SparsePauliOp
from sympy.tensor.indexed import Indexed
from sympy import IndexedBase
from qiskit.circuit.library import TwoLocal, QAOAAnsatz
from qiskit.algorithms.minimum_eigen_solvers import VQE
from qiskit.algorithms.optimizers import L_BFGS_B, COBYLA, SPSA, NELDER_MEAD
from qiskit import QuantumCircuit

# import custom modules
from logconfig import get_logger

## initialize logger
log = get_logger(__name__)

class kLocalModel():
    # following are the variables to store intermediate results of VQE operation
    intermediate_eval_count = []
    intermediate_mean = []
    intermediate_params = []
    intermediate_std = []

    def __init__(self, sym_expr):
        """
        The class takes a symbolic expression, identifies it's symbols/variables and 
        replaces them by named symbols
        """
        self.expr = sym_expr

        ## let us identify the symbols/variables used in the expression
        ## since, the expresssion contains indexed symbols, we only include those in our list
        ## e.g. p[1] is an indexed variable, whereas p is a symbol.
        self.var_list = list()
        temp_var_list = list(self.expr.free_symbols)
        for v in temp_var_list:
            if v.func == Indexed:
                self.var_list.append(v)

        # define number of qubits
        self.num_qubits = len(self.var_list)

        # let us define the Hamiltonian variable which will be initialised using the 
        # build_hamiltonian method
        self.hamiltonian = None

    def build_hamiltonian(self, normalize_hamiltonian=None):
        """
        This method builds the Ising Hamiltonian for the desired expression.
        """

        # Step 1: let us replace the existing variables/symbols by either 0.5(1 - Z), where 
        # Z is a variable and will be replaced later by Pauli-Z operator.
        # If a Z variable is missing for a specific index, it indicates an Identity operator I.
        z_sym = IndexedBase('Z')
        my_expr = self.expr
        for i in range(len(self.var_list)):
            v = self.var_list[i]
            my_expr = my_expr.subs({v:0.5*(1-z_sym[i+1])})
        my_expr = my_expr.expand()
        log.info(f"The updated expression consisting of Z variables is: {my_expr}")

        # Step 2: now, we identify the individual terms and coefficients of the expression
        terms_coeff_dict = kLocalModel.get_terms_and_coeffs(my_expr)

        # Step 3: check if the hamiltonian needs to be normalized and find the corresponding norm
        hamiltonian_norm = 0
        if normalize_hamiltonian == "L1":
            for term, coeff in terms_coeff_dict.items():
                hamiltonian_norm = max(hamiltonian_norm, abs(coeff))
        elif normalize_hamiltonian == "L2":
            for term, coeff in terms_coeff_dict.items():
                hamiltonian_norm = hamiltonian_norm + pow(coeff, 2)
            hamiltonian_norm = math.sqrt(hamiltonian_norm)

        # Step 4: we evaluate each term in the dictionary and replace the variables
        #   by the corresponding Pauli operators and add them to the Hamiltonian
        # IMP: When constructing the Pauli string, it is important to use the appropriate sequence 
        # of qubits. e.g. 'ZII' indicates that the 1st and 2nd qubits are identity and 3rd is the 
        # Pauli Z operator. Thus, the first qubit operator is the last letter in the Pauli string and the (n)th qubit operator is the first letter in the Pauli string.

        for term, coeff in terms_coeff_dict.items(): 
            # if normalize hamiltonian, then don't process insignificant entries
            if normalize_hamiltonian in ["L1","L2"]:
                new_coeff = round(coeff / hamiltonian_norm, 3)
                #if abs(new_coeff) < 0.001:
                #    log.debug(f"Normalize Hamiltonian: Ignored entry {term}:{coeff}:{new_coeff}")
                #    continue
            else:
                new_coeff = coeff

            # following variable position_dict is used to identify missing positions from the terms. The missing
            # positions have to be replaced by identity (I) operator as part of the string.
            position_dict = dict()

            temp_var_list = list(term.free_symbols)
            for i in range(len(temp_var_list)):
                v = temp_var_list[i].args
                if v and v[0] == z_sym:
                    # we have located a Z variable and hence we include the position in the dictionary
                    # along with the Z string
                    position_dict[v[1]] = 'Z'
            
            # we now identify missing position values and include them in the position_dict along with 
            # I string
            all_positions_set = set(range(1, len(self.var_list)+1))
            missing_positions_set = all_positions_set - set(position_dict.keys())
            for pos in missing_positions_set:
                position_dict[pos] = 'I'

            # subsequently, we sort the dictionary based on keys that will help us generate the pauli
            # string. The sorting is in reverse order since the 1st qubit position is the last in the
            # pauli string.
            sorted_position_dict = dict(sorted(position_dict.items(), reverse=True))
            log.debug(f"The sorted position dict is: {sorted_position_dict}")

            # now, we are in the position to generate the pauli string and assign the coefficient
            # following 2 variables are used to generate the Pauli string e.g. 'IZZ' and the 
            # corresponding coefficient
            pauli_coeff = new_coeff
            pauli_string = ""

            for pos, p_str in sorted_position_dict.items():
                pauli_string += p_str
            log.debug(f"The Pauli string and coefficient are: {pauli_string, pauli_coeff}")

            # include the pauli string and coefficient in the hamiltonian
            if self.hamiltonian is None:
                self.hamiltonian = PauliSumOp(SparsePauliOp(pauli_string, pauli_coeff))
            else:
                self.hamiltonian = self.hamiltonian.add(PauliSumOp(SparsePauliOp(pauli_string, pauli_coeff)))

        log.info(f"The Hamiltonian is: {self.hamiltonian}")

        log.info(f"The Hamiltonian is now built.")

    def set_initial_state(self, alpha=1, beta=0):
        """
        Signature: set_initial_state(alpha=1, beta=0)
        This method creates a superposition state based on coefficients alpha and beta.
        If alpha = 1 and beta = 0, then state is |0>.
        If alpha = 0 and beta = 1, then state is |1>.
        Note: it is important that the state is normalised i.e. |alpha|^2+|beta|^2=1.
        """
        initial_state = QuantumCircuit(self.num_qubits)

        if alpha == 1 and beta == 0:
            ## return just the base quantum circuit
            None  
        elif alpha == 0 and beta == 1:
            ## apply X gate to all qubits
            for i in range(self.num_qubits):
                initial_state.x(i)
        elif alpha == 1/math.sqrt(2) and beta == 1/math.sqrt(2):
            ## apply hadamard gate to all qubits
            for i in range(self.num_qubits):
                initial_state.h(i)
        else:
            ## use the initialize function to setup custom coefficients
            ## first check if the norm is 1
            if math.pow(abs(alpha),2) + math.pow(abs(beta),2) == 1:
                for i in range(self.num_qubits):
                    initial_state.initialize([alpha, beta],i)
        
        return initial_state
    
    def set_qaoa_ansatz(self, reps=3, initial_state=None):
        """
        Signature: set_qaoa_ansatz(reps=3, initial_state=None)
        reps: no. of repetitions/layers
        initial_state: if none, default Hadamard gate is applied to each qubit by QAOA Ansatz itself
        The cost hamiltonian to be used is already set at the object level.
        """
        log.info(f"Setting up the QAOA ansatz with arguments: reps={reps}")

        if initial_state is not None:
            ansatz = QAOAAnsatz(cost_operator=self.hamiltonian, reps=reps,
                            initial_state=initial_state, flatten=False)
            log.info(f"QAOA Ansatz setup with intial state.")
        else:
            ansatz = QAOAAnsatz(cost_operator=self.hamiltonian, reps=reps, flatten=False)
            log.info(f"QAOA Ansatz setup with initial state as None.")

        log.debug(f"Number of QAOA ansatz settable parameters: {ansatz.num_parameters_settable}")
        
        self.num_ansatz_params = ansatz.num_parameters
        log.info(f"Number of QAOA ansatz parameters: {self.num_ansatz_params}")

        return(ansatz)
    
    def set_custom_ansatz(self, rotations=["ry"], entanglement="cx", entanglement_type="linear", reps=3, final_rotation_layer = True, initial_state=None):
        """
        Signature: set_custom_ansatz(rotation_blocks, entanglement_blocks, entanglemnt_type, reps)
        reps: no. of repetitions/layers
        rotation_block: either ["ry] or ["ry","rz"]
        entanglement_block: ["cx"] or ["cz"]
        """

        log.info(f"Setting up the custom ansatz with arguments: num_qubits: {self.num_qubits}, rotations:{rotations}, entanglement:{entanglement}, entanglement_type:{entanglement_type}, reps={reps}, final_rotation_layer={final_rotation_layer}")

        self.num_ansatz_params = self.num_qubits * len(rotations) * (reps + final_rotation_layer)
        if entanglement in ["cry","crx","crz"]:
            if entanglement_type in ["linear", "reverse_linear"]:
                self.num_ansatz_params = self.num_ansatz_params + (self.num_qubits - 1) * reps
            elif entanglement_type == "full":
                self.num_ansatz_params = self.num_ansatz_params + int(self.num_qubits * (self.num_qubits - 1) * reps / 2)
        log.info(f"Number of custom ansatz parameters: {self.num_ansatz_params}")

        if entanglement_type is None or entanglement_type == "":
            ansatz = TwoLocal(num_qubits=self.num_qubits,
                  rotation_blocks=rotations, 
                  #entanglement_blocks=entanglement,
                  #entanglement=entanglement_type, 
                  reps=reps,
                  skip_final_rotation_layer=not(final_rotation_layer),
                  insert_barriers=True)
        else:
            ansatz = TwoLocal(num_qubits=self.num_qubits,
                    rotation_blocks=rotations, 
                    entanglement_blocks=entanglement,
                    entanglement=entanglement_type, 
                    reps=reps,
                    skip_final_rotation_layer=not(final_rotation_layer),
                    insert_barriers=True)

        if initial_state is not None:
            ansatz.compose(initial_state, front = True, inplace=True)
            log.info(f"Custom Ansatz setup with intial state.")
        else:
            log.info(f"Custom Ansatz setup with initial state as None.")

        return(ansatz)
    
    def set_optimizer(self, optimizer_label='L_BFGS_B', maxiter=3000, tol=1e-3):
        """
        This method identifies the optimiser to be used based on label.
        Signature: set_optimiser(optimiser_label='L_BFGS_B')
        """
        try:
            if optimizer_label == 'L_BFGS_B':
                self.optimizer = L_BFGS_B(ftol=tol, maxfun=maxiter, maxiter=maxiter)
                #self.optimiser.set_options(bounds=[(-math.pi,math.pi) for i in self.num_ansatz_parms])
            elif optimizer_label == 'COBYLA':
                self.optimizer = COBYLA(tol=tol, maxiter=maxiter)
            elif optimizer_label == 'SPSA':
                self.optimizer = SPSA(maxiter=maxiter)
            elif optimizer_label == 'NELDER_MEAD':
                self.optimizer = NELDER_MEAD(tol=tol, maxfev=maxiter, maxiter=maxiter)
            else:
                raise ValueError
        except ValueError:
            log.exception(f'Incorrect optimizer label : {optimizer_label}. It should be one of L_BFGS_B, COBYLA, SPSA or NELDER_MEAD.')
            raise
        
    def setup_vqe(self, ansatz, backend, initial_point, optimizer='COBYLA', maxiter=3000, tol=1e-3):
        """
        Signature: setup_vqe()
        """
        # first set the optimiser
        self.set_optimizer(optimizer_label=optimizer,maxiter=maxiter, tol=tol)

        log.info(f"Initial point size: {initial_point.size}")
        if initial_point.size == 0:
            vqe = VQE(ansatz,optimizer=self.optimizer,quantum_instance=backend)
        else:
            vqe = VQE(ansatz,optimizer=self.optimizer,quantum_instance=backend, initial_point=initial_point, callback=self.store_intermediate_results)

        log.debug(f"VQE optimizer setup as {vars(vqe.optimizer)}")

        log.info(f"VQE instance setup.")
        return(vqe)
    
    def store_intermediate_results(self, eval_count, parameters, mean, std):
        self.intermediate_eval_count.append(eval_count)
        self.intermediate_mean.append(mean)
        self.intermediate_params.append(parameters)
        self.intermediate_std.append(std)

    def plot_intermediate_results_graph(self, ansatz_init_param_label='RANDOM', debug=False):
        """
        Description: Plot the count vs expectation values of intermediate results
        """
        log.info("Inside plot_intermediate_results_graph method ===========")

        if ansatz_init_param_label == 'RANDOM':
            plot_title = r': Initial Parameter values = $\{ \vec \theta | \theta_i \sim U[0,2\pi] \}$'
        elif ansatz_init_param_label == 'ZERO':
            plot_title = r': Initial Parameter values = $\{ \vec \theta | \theta_i = 0 \}$'
        else:
            plot_title = ''
        
        intermediate_results_plot = \
            kLocalModel.single_plot_graph(x_values=self.intermediate_eval_count, 
                                         y_values=self.intermediate_mean, 
                                         plot_title='Optimizer Convergence' + plot_title, 
                                         x_label='Evaluation Count', y_label='Expectation value',
                                         debug=debug)

        log.info("End of plot_intermediate_results_graph method ===========")

        return(intermediate_results_plot)

    @classmethod
    def single_plot_graph(cls, x_values, y_values, plot_title, x_label, y_label, debug=False):
        log.info("Inside single_plot_graph method ===========")
        if debug == True:
            log.debug("Printing X and Y values ...")
            log.debug(x_values)
            log.debug(y_values)

        plt.plot(x_values, y_values)
        plt.xlabel(x_label)
        plt.ylabel(y_label)
        plt.title(plot_title)
        #plt.yticks(np.arange(np.floor(min(y_values)), np.ceil(max(y_values)), 0.5))
        plt.yticks(np.linspace(start=np.floor(min(y_values)), stop=np.ceil(max(y_values)), num=10))
        #plt.legend()
        #plt.show()
        #plt.close()

        log.info("End of single_plot_graph method ===========")
        
        return(plt)

    @classmethod
    def get_terms_and_coeffs(cls, expr):
        """
        This method returns the individual terms and coefficients in the form of a dictionary 
        with the term as key and the coefficient as value.
        """
        # let us identify the terms and coefficients in the expression 
        # and return the dictionary
        return(expr.as_coefficients_dict())
            
    def get_expression(self):
        return(self.expr)
    
    def get_var_list(self):
        return(self.var_list)

