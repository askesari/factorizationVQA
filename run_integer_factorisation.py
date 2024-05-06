"""
This module executes the integer factorisation program.
Author: Amit S. Kesari
"""
# import the basic and core python modules 
import os, yaml, math, time
import numpy as np
from matplotlib import pyplot as plt
plt.rcParams['text.usetex'] = True
import tikzplotlib # converts matplotlib image to tikz tex content
from sympy import IndexedBase

# import qiskit modules
from qiskit import QuantumCircuit, QuantumRegister, Aer, IBMQ
from qiskit.compiler import transpile
from qiskit.algorithms.minimum_eigensolvers import NumPyMinimumEigensolver
from qiskit.algorithms.eigensolvers import NumPyEigensolver
from qiskit.visualization import plot_histogram
from qiskit.providers.fake_provider import FakeAthens, FakeAlmaden, FakeMumbai, FakeManhattan
from qiskit_aer.backends import AerSimulator, QasmSimulator
from qiskit_aer.noise import NoiseModel
from qiskit.providers.ibmq import least_busy
from qiskit.algorithms import VarQITE
from qiskit.algorithms.time_evolvers import TimeEvolutionProblem
from qiskit.primitives import Estimator

# import custom modules
from Factorisation import IntegerFactorisation
from DirectFactorisation import DirectFactorisation
from ColumnFactorisation import ColumnFactorisation
from kLocalModel import kLocalModel
from logconfig import get_logger
from mitigation import M3MitigationDerived as m3_mit

## define some global variables
curr_dir_path = os.path.dirname(os.path.realpath(__file__))
outputfilepath = curr_dir_path + "/output"

if not os.path.exists(outputfilepath):
    os.makedirs(outputfilepath)

# define global dictionary for backend devices
"""
IMP: 
1. QasmSimulator.from_backend does not currently support V2 nosiy sumulator devices and hence have not been included in the devices dictionary.
2. Some noisy simulator devices e.g. FakeManila throw 'tkinter' error during transpilation and hence are not included as well. Most probably, this is becauase transpiled errors are generated in a multi-threaded manner. 
"""
devices = {'FAKEATHENS': FakeAthens(),
           'FAKEALMADEN': FakeAlmaden(),
           'FAKEMUMBAI': FakeMumbai(),
           'FAKEMANHATTAN': FakeManhattan(),
          }

## initialize logger
log = get_logger(__name__)

### following fix found online at 
### https://stackoverflow.com/questions/75900239/attributeerror-occurs-with-tikzplotlib-when-legend-is-plotted and the author of the soln is
### https://github.com/st--
### the bug is already filed at https://github.com/nschloe/tikzplotlib/pull/558 
def tikzplotlib_fix_ncols(obj):
    """
    workaround for matplotlib 3.6 renamed legend's _ncol to _ncols, which breaks tikzplotlib
    """
    if hasattr(obj, "_ncols"):
        obj._ncol = obj._ncols
    for child in obj.get_children():
        tikzplotlib_fix_ncols(child)

def is_folder_path_exists(folderpath):
    """
    Check if the folder path location exists and return True, if yes, otherwise False
    """
    ## initialize to False
    folder_exists = False

    try:
        if folderpath is not None:
            #v_file_dir = filename_with_path.rpartition("/")[0]
            if not os.path.exists(folderpath):
                log.info(f"Folder path {folderpath} does not exist. Please ensure to create it.")
            else:
                folder_exists = True
        else:
            raise NameError("Folder path not passed as input: " + folderpath)          
    except NameError as ne1:
        log.exception(ne1, stack_info=True)
        raise
    
    return(folder_exists)

# get noise model and coupling map for the requested noisy simulator
def get_noise_model(noise_model_device):
    """
    Method name: get_noisy_device(noise_model_device)
    This method returns the backend, noise model and the coupling map corresponding to the
    input device model string. 
    """
    log.info(f"Start of method get_noisy_device ...")
    try:
        if noise_model_device != None:
            device_backend = devices.get(noise_model_device.upper())
            if device_backend is None:
                raise Exception(f"Noise model device label can only be one of {list(devices.keys())}")
            
            device = AerSimulator.from_backend(device_backend)
            device.set_options(precision='single')
            coupling_map = device.configuration().coupling_map
            noise_model = NoiseModel.from_backend(device)
        else:
            raise Exception("No noise model defined.")

    except Exception as e:
        log.exception(e, stack_info=True)
        raise
    else:
        return(device_backend, noise_model, coupling_map)    
            
# get the appropriate backend, either the ideal simulator, noisy simulator or the
# actual hardware backend            
def get_backend(num_qubits, is_simulator=True, simulator_type='AER_STATEVECTOR', noise_model_device=None):        
    """
    Method Name: get_backend(num_qubits, is_simulator=True, simulator_type='AER', noise_model_device=None)
    """
    try:
        noise_model = None
        coupling_map = None
        
        if is_simulator == False:
            # Get quantum computer backend
            provider = IBMQ.get_provider(hub = 'ibm-q')
            #provider.backends()
            #backend = provider.get_backend('ibm_nairobi')
            backend = least_busy(provider.backends
                                (filters = lambda x:x.configuration().n_qubits >= num_qubits and not x.configuration().simulator
                                    and x.status().operational==True)
                        )
            if backend is None:
                raise Exception("No IBM Q backend avaiable for the required criteria.")

        elif is_simulator == True:
            if simulator_type == 'AER_STATEVECTOR':
                backend = Aer.get_backend('aer_simulator_statevector')
            elif simulator_type == 'AER':
                if noise_model_device is None:
                    backend = Aer.get_backend('aer_simulator')
                else:
                    backend, noise_model, coupling_map = get_noise_model(noise_model_device)
            elif simulator_type == 'QASM':
                backend = Aer.get_backend('qasm_simulator') 
    except Exception as e:
        log.exception(e, stack_info=True)
        print(e)
        raise
    else:
        return(backend, noise_model, coupling_map)

def get_decimal_output(var_list, bit_strings, var_values_dict, p_num_vars, q_num_vars):
    """
    This method returns the decimal output for the binary output
    """
    # step 1: Let us maintain a map in dictionary format {variale:bit location} where 
    #         variable -> p, q or c variable with appropriate index and bit location -> location of 
    #         variable in the bit string.
    #         Also, we maintain the list of 'p' and 'q' variables found so that missing variables
    #         can be included in the final processing.
    
    log.info(f"Getting decimal output ...")
    log.debug(var_list)
    log.debug(bit_strings)
    log.debug(var_values_dict)
    log.debug(p_num_vars)
    log.debug(q_num_vars)
    
    pq_final_value_list = list()
    found_dict = dict()
    p_found_list = list() # required to find missing p variables
    q_found_list = list() # required to find missing q variables
    for i in range(len(var_list)):
        if var_list[i].args[0] == IndexedBase('p'):
            p_found_list.append(var_list[i].args[1])
        elif var_list[i].args[0] == IndexedBase('q'):
            q_found_list.append(var_list[i].args[1])
        # include the variable and i i.e. bit location
        found_dict[var_list[i]] = i
    log.debug(f"p found list: {p_found_list}")
    log.debug(f"q found list: {q_found_list}")

    # step 2: find the missing variables - required for column factorisation
    my_pn_set = set(range(p_num_vars)) - {0}
    my_qn_set = set(range(q_num_vars)) - {0}
    p_missing = my_pn_set - set(p_found_list)
    q_missing = my_qn_set - set(q_found_list)
    log.debug(f"Missing p variables: {p_missing}")
    log.debug(f"Missing q variables: {q_missing}")

    # step 3: find the bit location from bit_string of the missing p and q variables
    pq_missing_dict = dict()

    # 3 scenarios possible: x_i = x_j or x_i = 1/0 or x_i = 1 - x_j
    # processing missing p values
    if len(p_missing) > 0 and len(var_values_dict) > 0:
        # find the missing variables in var_values_dict
        for i in p_missing:
            missing_val = var_values_dict.get(IndexedBase('p')[i])
            if missing_val is not None:
                if isinstance(missing_val, int) == True:
                    pq_missing_dict[IndexedBase('p')[i]] = (missing_val, None)
                else:
                    my_key = None
                    my_coeff = None
                    is_recursive = False
                    missing_val_terms_dict = missing_val.as_coefficients_dict()
                    for key, coeff in missing_val_terms_dict.items(): ## maximum 2 terms
                        if key == 1:
                            my_coeff = coeff
                        elif found_dict.get(key) is not None:
                            my_key = (coeff, found_dict.get(key))
                        ### if the term corresponding to the missing value is not in found list,
                        ### but instead in the missing dictionary, then it needs to be handled
                        ### appropriately.
                        elif pq_missing_dict.get(key) is not None:
                            my_key = pq_missing_dict.get(key)
                            is_recursive = True
                        elif key in p_missing:
                            my_key = key
                            is_recursive = True
                        elif key in q_missing:
                            my_key = key
                            is_recursive = True

                    if is_recursive == True and my_coeff is None:
                        pq_missing_dict[IndexedBase('p')[i]] = my_key
                    elif is_recursive == True and my_coeff is not None:
                        (my_new_coeff, my_new_key) = my_key
                        my_new_coeff = my_new_coeff + my_coeff
                        pq_missing_dict[IndexedBase('p')[i]] = (my_new_coeff, my_new_key)

                    else:            
                        pq_missing_dict[IndexedBase('p')[i]] = (my_coeff, my_key)
   
    # processing missing q values
    if len(q_missing) > 0 and len(var_values_dict) > 0:
        # find the missing variables in var_values_dict
        for i in q_missing:
            missing_val = var_values_dict.get(IndexedBase('q')[i])
            if missing_val is not None:
                if isinstance(missing_val, int) == True:
                    pq_missing_dict[IndexedBase('q')[i]] = (missing_val, None)
                else:
                    my_key = None
                    my_coeff = None
                    is_recursive = False
                    missing_val_terms_dict = missing_val.as_coefficients_dict()
                    for key, coeff in missing_val_terms_dict.items(): ## maximum 2 terms
                        if key == 1:
                            my_coeff = coeff
                        elif found_dict.get(key) is not None:
                            my_key = (coeff, found_dict.get(key))
                        ### if the term corresponding to the missing value is not in found list,
                        ### but instead in the missing dictionary, then it needs to be handled
                        ### appropriately.
                        elif pq_missing_dict.get(key) is not None:
                            my_key = pq_missing_dict.get(key)
                            is_recursive = True
                        elif key in p_missing:
                            my_key = key
                            is_recursive = True
                        elif key in q_missing:
                            my_key = key
                            is_recursive = True

                    if is_recursive == True and my_coeff is None:
                        pq_missing_dict[IndexedBase('q')[i]] = my_key
                    elif is_recursive == True and my_coeff is not None:
                        (my_new_coeff, my_new_key) = my_key
                        my_new_coeff = my_new_coeff + my_coeff
                        pq_missing_dict[IndexedBase('q')[i]] = (my_new_coeff, my_new_key)
                    else:
                        pq_missing_dict[IndexedBase('q')[i]] = (my_coeff, my_key)

    log.debug(f"PQ missing dictionary with location map: {pq_missing_dict}")

    # step 4: perform the decimal value computation. 
    #         Include the computation of missing variables, if found, into the existing decimal
    #         values of p and q

    for bit_str_value in bit_strings:
        p_val = 1
        q_val = 1
        for i in range(len(var_list)):
            if var_list[i].args[0] == IndexedBase('p'):
                p_val = p_val + int(bit_str_value[i]) * math.pow(2,var_list[i].args[1])
            elif var_list[i].args[0] == IndexedBase('q'):
                q_val = q_val + int(bit_str_value[i]) * math.pow(2,var_list[i].args[1])

        for key, loc_map in pq_missing_dict.items():
            log.debug(f"{key}:{loc_map}")
            if key.args[0] == IndexedBase('p'):
                if loc_map[0] is None and loc_map[1] is None:
                    log.exception(f"p: {loc_map} .... Oops... there seems to be some problem ... ")
                    raise
                elif loc_map[0] is not None and loc_map[1] is None: # only constant present
                    p_val = p_val + int(loc_map[0]) * math.pow(2,key.args[1])
                elif loc_map[0] is None and loc_map[1] is not None:
                    bit_loc = loc_map[1]
                    p_val = p_val + (bit_loc[0] * int(bit_str_value[bit_loc[1]])) * math.pow(2,key.args[1])
                else:
                    log.debug(f"Const: {const_coeff}, bit_loc: {bit_loc}, {bit_loc[0]}, {bit_str_value[bit_loc[1]]}, Value: {(const_coeff + bit_loc[0] * int(bit_str_value[bit_loc[1]])) * math.pow(2,key.args[1])}")
                    const_coeff = loc_map[0]
                    bit_loc = loc_map[1]
                    p_val = p_val + (const_coeff + bit_loc[0] * int(bit_str_value[bit_loc[1]])) * math.pow(2,key.args[1])
            elif key.args[0] == IndexedBase('q'):
                if loc_map[0] is None and loc_map[1] is None:
                    log.exception(f"q: {loc_map} .... Oops... there seems to be some problem ... ")
                    raise
                elif loc_map[0] is not None and loc_map[1] is None: # only constant present
                    q_val = q_val + int(loc_map[0]) * math.pow(2,key.args[1])
                elif loc_map[0] is None and loc_map[1] is not None:
                    bit_loc = loc_map[1]
                    q_val = q_val + (bit_loc[0] * int(bit_str_value[bit_loc[1]])) * math.pow(2,key.args[1])
                else:
                    const_coeff = loc_map[0]
                    bit_loc = loc_map[1]
                    log.debug(f"Const: {const_coeff}, bit_loc: {bit_loc}, {bit_loc[0]}, {bit_str_value[bit_loc[1]]}, Value: {(const_coeff + bit_loc[0] * int(bit_str_value[bit_loc[1]])) * math.pow(2,key.args[1])}")
                    q_val = q_val + (const_coeff + bit_loc[0] * int(bit_str_value[bit_loc[1]])) * math.pow(2,key.args[1])

        pq_final_value_list.append(str((int(p_val),int(q_val))))

    log.debug(f"Final Decimal Soln: {pq_final_value_list}")

    return(pq_final_value_list)


# start of main function
def main():
    log.info("=============================================")
    log.info(f"Start of program ...")
    log.info(f"Checking if output path exists ...")

    ## load account ##
    log.info("Loading Qiskit account. This may take some time, so please hang on ...")
    IBMQ.load_account()
    log.info("Qiskit account loaded successfully.")

    log.info(f"Loading parameter file ...")
    ## load the parameter.yaml file
    skip_lines=15
    try:
        with open("parameters.yaml", 'r') as param_stream:
            for i in range(skip_lines):
                _ = param_stream.readline()
            parameters = yaml.safe_load(param_stream)
    except FileNotFoundError as fnf:
        raise
    finally:
        param_stream.close()
    
    log.info(f"paramaters: {parameters}")
    log.info(f"Parameter file read successfully.")

    #### set the variables to appropriate parameter values
    N = parameters['N'] # initialise the integer to be factorised
    method = parameters['method'] # set the method 'C' or 'D'
    bit_length_type = parameters['bit_length_type'] # set the bit_length_type to 'EQUAL' or 'GENERAL'
    normalize_hamiltonian = parameters['normalize_hamiltonian'] # set the value to "L2" or "L1"
    is_simulator = parameters['is_simulator'] # set is_simulator = True or False
    noise_model_device = parameters['noise_model_device']
    init_state_label = parameters['init_state']

    # Set the parameters for optimizer
    optimizer_label = parameters['optimizer_label']
    optimizer_maxiter = parameters['optimizer_maxiter']
    optimizer_tol = parameters['optimizer_tol']

    # Set the parameters for the ansatz
    ansatz_type = parameters['ansatz_type']
    ansatz_init_param_label = parameters['ansatz_init_param_values']
    ansatz_reps = parameters['ansatz_num_layers']
    rotation_layer = parameters['custom_ansatz']['rotations']
    entanglement = parameters['custom_ansatz']['entanglement']
    entanglement_type = parameters['custom_ansatz']['entanglement_type']
    final_rotation_layer = parameters['custom_ansatz']['final_rotation_layer']

    # additional parameters
    is_exact_solver = parameters['is_exact_solver']

    # mitigation related parameters
    is_apply_m3_mitigation = parameters['is_apply_m3_mitigation']

    # create output folder within the main output folder for each N
    myoutputfilepath = outputfilepath + "/N" + str(N)
    new_subdir_num = 0
    log.info(f"Checking if output path exists ...")
    outputpath_exists = is_folder_path_exists(myoutputfilepath)
    if not outputpath_exists:
        ## main folder related to N doesn't exist and hence create it as well as the sub-folder 001
        log.info(f"Creating output folder {myoutputfilepath} ...")
        os.makedirs(myoutputfilepath)
        new_subdir_num = new_subdir_num + 1
        myoutputfilepath = myoutputfilepath + "/" + str(new_subdir_num).zfill(3)
        log.info(f"Creating output folder {myoutputfilepath} ...")
        os.makedirs(myoutputfilepath)
    else:
        ## main folder related to N exists and hence create the next sub-folder based on sorted list
        ## of sub-directories
        log.info(f"Main Output folder {myoutputfilepath} already exists.")
        subdir_list = sorted(os.listdir(myoutputfilepath), reverse=True)
        new_subdir_num = int(subdir_list[0])+1
        if new_subdir_num < 1000:
            myoutputfilepath = myoutputfilepath + "/" + str(new_subdir_num).zfill(3)
            log.info(f"Creating output folder {myoutputfilepath} ...")
            os.makedirs(myoutputfilepath)
        else:
            ## it seems that we have reached the limit of 999 subfolders for a specific N.
            ## so, its better to clean the current directory and start fresh for this specific 
            ## value of N
            log.exception(f"IMP!!!!! The limit of 999 subfolders for N={N} seems to be exhausted. Please clean up the subfolders for N={N} and rerun the process.")
            raise

    # depending on the method, identify the expression whose value is to be minimized
    if method == 'D':
        # initialise direct factorisation object
        log.info(f"Starting Direct method ...")
        df = DirectFactorisation(N, bit_length_type=bit_length_type)
        binary_N = df.get_binary_N()
        df_expr = df.get_expression()
        my_p = df.get_p()
        my_q = df.get_q()
        df_norm_expr = df.get_norm_expression()
        log.info(f"Binary value of {N}: {binary_N}")
        log.info(f"p: {my_p}")
        log.info(f"q: {my_q}")
        log.debug(f"N-p*q expression: {df_expr}")

        df_var_list = IntegerFactorisation.get_var_list(df_expr)

        log.info(f"==============================================")
        log.info(f"N: {N}")
        log.info(f"Direct: No. of variables: {len(df_var_list)}; Variable list: {df_var_list}")
        log.info(f"Direct Norm Expression: {df_norm_expr}")
        log.info(f"==============================================")

    elif method == 'C':
        # initialise column factorisation object
        log.info(f"Starting Column-based method ... ")
        cf = ColumnFactorisation(N, bit_length_type=bit_length_type)
        binary_N = cf.get_binary_N()
        my_column_clauses = cf.get_column_clauses()
        my_p = cf.get_p()
        my_q = cf.get_q()
        log.info(f"Binary value of {N}: {binary_N}")
        log.info(f"p: {my_p}")
        log.info(f"q: {my_q}")

        for i, clause in enumerate(my_column_clauses):
            log.debug(f"Column clause C{i+1}: {clause}")

        num_iterations = cf.classical_preprocessing(num_iterations = 10)
        log.info(f"No. of iterations executed by classical preprocessor: {num_iterations}")
        
        #num_special_iterations = cf.special_classical_preprocessing(num_iterations = 5)
        #log.info(f"No. of iterations executed by special preprocessor: {num_special_iterations}")

        my_column_clauses = cf.get_column_clauses()
        for i, clause in enumerate(my_column_clauses):
            log.debug(f"Column clause C{i+1}: {clause}")
        my_var_values_dict = cf.get_var_values_dict()
        log.debug(my_var_values_dict)

        cf_norm_expr = cf.get_norm_expression()
        cf_var_list = IntegerFactorisation.get_var_list(cf_norm_expr)

        log.info(f"==============================================")
        log.info(f"N: {N}")
        log.info(f"Column: No. of variables: {len(cf_var_list)}; Variable list: {cf_var_list}")
        log.info(f"Column Norm Expression: {cf_norm_expr}")
        log.info(f"==============================================")

    elif method == 'H':
        log.info(f"Starting Hybrid method ... Direct factorization for expression and column-based for classical pre-processing ...")
        # initialise direct factorisation object
        hf = DirectFactorisation(N, bit_length_type=bit_length_type)
        binary_N = hf.get_binary_N()
        my_p = hf.get_p()
        my_q = hf.get_q()
        log.info(f"Binary value of {N}: {binary_N}")
        log.info(f"p: {my_p}")
        log.info(f"q: {my_q}")

        hf_expr = hf.get_expression()
        log.debug(f"original N-p*q expression: {hf_expr}")

        # Since, we will be using ColumnFactorisation object only for classical pre-processing,
        # the object has been named as "_temp"
        cf_temp = ColumnFactorisation(N, bit_length_type=bit_length_type)
        my_column_clauses = cf_temp.get_column_clauses()
        for i, clause in enumerate(my_column_clauses):
            log.debug(f"Column clause C{i+1}: {clause}")

        num_iterations = cf_temp.classical_preprocessing(num_iterations = 10)
        log.info(f"No. of iterations executed by classical preprocessor: {num_iterations}")
        
        #num_special_iterations = cf.special_classical_preprocessing(num_iterations = 5)
        #log.info(f"No. of iterations executed by special preprocessor: {num_special_iterations}")

        my_column_clauses = cf_temp.get_column_clauses()
        for i, clause in enumerate(my_column_clauses):
            log.debug(f"Column clause C{i+1}: {clause}")
        my_var_values_dict = cf_temp.get_var_values_dict()
        log.debug(f"Variable-value mapping: {my_var_values_dict}")

        hf_norm_expr = hf.get_norm_expression(my_var_values_dict)
        hf_var_list = IntegerFactorisation.get_var_list(hf_norm_expr)

        log.info(f"==============================================")
        log.info(f"N: {N}")
        log.info(f"Hybrid: No. of variables: {len(hf_var_list)}; Variable list: {hf_var_list}")
        log.info(f"Hybrid Norm Expression: {hf_norm_expr}")
        log.info(f"==============================================")
    
    else:
        log.info("Incorrect Method Type. Value can be only one of D, C or H")
        raise
    
    if method == 'D':
        my_expr = df_norm_expr
        p_num_vars = df.get_p_num_vars() 
        q_num_vars = df.get_q_num_vars()
    elif method == 'C':
        my_expr = cf_norm_expr
        p_num_vars = cf.get_p_num_vars() 
        q_num_vars = cf.get_q_num_vars()
    elif method == 'H':
        my_expr = hf_norm_expr
        p_num_vars = hf.get_p_num_vars() 
        q_num_vars = hf.get_q_num_vars()

    ### check if the norm expression exists so that vqe process can be initiated.
    ### otherwise, the classical pre-processing itself gives us the result.
    if method in ['C','H'] and my_expr == 0:
        # get the decimal output directly as nothing to process further
        log.info("Classical pre-processing is sufficient to get the output and hence VQE not initiated.")
        exit()

    # set up ising model
    klocal_model = kLocalModel(my_expr)
    log.info(f"Variable list: {klocal_model.var_list}")

    # now, build the k-local Hamiltonian
    klocal_model.build_hamiltonian(normalize_hamiltonian)

    ### start setup of VQE
    
    # setup initial state
    if init_state_label == 'ALL_ZEROS':
        init_state = klocal_model.set_initial_state(alpha=1, beta=0)
    elif init_state_label == 'ALL_ONES':
        init_state = klocal_model.set_initial_state(alpha=0, beta=1)
    elif init_state_label == 'EQUAL_SUPERPOS':    
        init_state = klocal_model.set_initial_state(alpha=1/math.sqrt(2), beta=1/math.sqrt(2))
    else:
        log.exception(f"Invalid value for init_state. Value can be one of ALL_ZEROS, ALL_ONES, EQUAL_SUPERPOS")
    
    # setup ansatz
    if ansatz_type == 'CUSTOM':
        ansatz = klocal_model.set_custom_ansatz(rotations = rotation_layer, entanglement = entanglement, entanglement_type = entanglement_type, reps=ansatz_reps, final_rotation_layer=final_rotation_layer, initial_state=init_state)
    elif ansatz_type == 'QAOA': # applicable only for algorithm VQE
        ansatz = klocal_model.set_qaoa_ansatz(reps=ansatz_reps, initial_state=init_state)
    else:
        log.exception(f"Incorrect Ansatz Type set. Value can be one of 'CUSTOM' or 'QAOA'.")
        raise

    if is_simulator == True:
        if noise_model_device is None or noise_model_device == "":
            if klocal_model.num_qubits > 10:
                # initialise backend (ideal simulator)
                backend, noise_model, coupling_map = get_backend(num_qubits=klocal_model.num_qubits, simulator_type='QASM')
            else: 
                # initialise backend (ideal simulator)
                backend, noise_model, coupling_map = get_backend(num_qubits=klocal_model.num_qubits, simulator_type='AER_STATEVECTOR')
        else:
            # using the noisy simulator
            backend, noise_model, coupling_map = get_backend(num_qubits=klocal_model.num_qubits, simulator_type='AER', noise_model_device=noise_model_device)
    else:
        log.exception("IMP!! Directly invoking on hardware not enabled as of now. Set is_simulator = True")
        raise

    backend_literal = backend.__class__.__name__
    log.info(f"The backend device is: {backend}")
    log.debug(f"The noise model is: {noise_model}")
    log.debug(f"The coupling map is: {coupling_map}")
    log.info(f"The backend device literal is: {backend_literal}")

    ## print ansatz
    if outputpath_exists == True:
        if ansatz_type == "QAOA" and ansatz_reps > 1:
            log.info(f"QAOA ansatz too deep to be printed!! Ignoring the exception for now ... ")
        else:
            ansatz.decompose().draw(output='latex',filename=myoutputfilepath + '/' + method + '_' + str(N) + '_' + str(new_subdir_num).zfill(3) + '_' + init_state_label + '_' + ansatz_type + '_' + optimizer_label + '_' + backend_literal + '.png')
            log.info(f"Ansatz printed. Check for file {method}_{str(N)}_{str(new_subdir_num).zfill(3)}_{init_state_label}_{ansatz_type}_{optimizer_label}_{backend_literal}.png in the output directory.")

    # initialize ansatz parameters
    #initial_point = np.array([])
    if ansatz_init_param_label == "ZERO":
        initial_point = np.zeros(klocal_model.num_ansatz_params)
    elif ansatz_init_param_label == "RANDOM":
        rand_num_generator = np.random.default_rng(seed=108)
        initial_point = rand_num_generator.uniform(low=0,high=2*np.pi,size=(klocal_model.num_ansatz_params,))
    else:
        log.exception(f"Incorrect value of parameter {ansatz_init_param_label}. Value can only be one of ZERO or RANDOM")
        raise

    log.info(f"Initial Point for ansatz: {initial_point}")

    ### execute the VQE algorithm
    vqe = klocal_model.setup_vqe(ansatz=ansatz, backend=backend, initial_point=initial_point, 
                            optimizer = optimizer_label, maxiter=optimizer_maxiter, tol=optimizer_tol)
        
    # let us print the expectation estimation circuits
    operator_fn = vqe.construct_expectation(parameter=initial_point,operator=klocal_model.hamiltonian)
    log.info(f"The expectation operator is: ")
    log.info(operator_fn)
    
    # let us compute the minimum eigen value
    # !! IMP !!! we will also time the result to get an approximate idea regarding time taken per
    #            iteration. This is primarily helpful for noisy simulator and most probably QPU as well.
    start_time_vqe = time.time()
    result = vqe.compute_minimum_eigenvalue(klocal_model.hamiltonian)
    vqe_time_taken = time.time() - start_time_vqe
    log.info(f"Print result: {result}")
    log.info(f"Time taken for execution: {vqe_time_taken}")

    vqe_optimal_params = result.optimal_point
    vqe_eigen_value = result.eigenvalue
    vqe_func_evals = result.cost_function_evals

    iteration_delta = abs(vqe_eigen_value) ### since minimum value of Hamiltonian/cost function
                                            ### must be zero.
    log.info(f"VQE eigen value: {vqe_eigen_value}")
    log.info(f"VQE optimal point: {vqe_optimal_params}")
    #num_parameters = ansatz.num_parameters

    ## we check if the energy value of the ground state / minimized value
    ## of the cost function is within 0.1 value of the expected value 0.0. Accordingly, we know, whether
    ## VQE terminated within the expected threshold value or not.
    ## 
    threshold = 0.1
    if abs(iteration_delta) < threshold:
        log.info(f"VQE converged with eigen value: {vqe_eigen_value}")
    else:
        log.info(f"Suboptimal output: VQE converged with eigen value: {vqe_eigen_value}")

    ## print/plot intermediate results to check for convergence
    if outputpath_exists == True:
        intermediate_plot_location=myoutputfilepath + '/' + method + '_convergence_' + str(N) + '_' + str(new_subdir_num).zfill(3) + '_' + init_state_label + '_' + ansatz_type + '_' + optimizer_label + '_' + backend_literal
        intermediate_plot = klocal_model.plot_intermediate_results_graph \
                                (ansatz_init_param_label=ansatz_init_param_label)
        tikzplotlib.save(intermediate_plot_location + '.tex')
        intermediate_plot.savefig(intermediate_plot_location + '.png')
        log.info(f"Intermediate results printed. Check for file {method}_convergence_{str(N)}_{str(new_subdir_num).zfill(3)}_{init_state_label}_{ansatz_type}_{optimizer_label}_{backend_literal}.png in the output directory.")

    if is_exact_solver == True:
        log.info(f"Starting exact solver ...")

        ## getting the classical result
        npme = NumPyMinimumEigensolver()
        exact_result = npme.compute_minimum_eigenvalue(operator=klocal_model.hamiltonian)
        log.info(f"The exact result using NumPy minimum eigen solver: {exact_result}")
        # now, let us get the eigen state i.e. state vector in dictionary form
        #my_min_eigenstate = exact_result.eigenstate.primitive.to_dict()
        my_min_eigenstate = exact_result.eigenstate.to_dict()
        log.info(f"Eigenstate for min. eigen value as dictionary: {my_min_eigenstate}")

        ## now, let us analyse classically again but for 'k' eigen values
        npe = NumPyEigensolver(k=klocal_model.num_qubits)
        exact_eigenvalues = npe.compute_eigenvalues(operator=klocal_model.hamiltonian)
        log.debug(f"The exact result using NumPy eigen solver: {exact_eigenvalues}")

        ## let us find out the eigen vectors corresponding to eigen value of 0.+0.j
        for i in range(len(exact_eigenvalues.eigenvalues)):
            #if exact_eigenvalues.eigenvalues[i] == 0:
            my_eigenvalue = exact_eigenvalues.eigenvalues[i]
            my_eigenstate = exact_eigenvalues.eigenstates[i].to_dict()
            log.info(f"Eigen state for eigen value {my_eigenvalue}: {my_eigenstate}")

        # now, let us sample the exact result, which is a VectorStateFn
        #exact_counts = exact_result.eigenstate.sample()
        exact_counts = exact_result.eigenstate.sample_counts(shots=4096)
        log.info(f"Exact processing complete.")

    ## now, that we have identified the optimal point, we use it to bind the ansatz circuit.
    ## But, before that we append the measurement part and run the circuit to get the counts
    ## Step 1: Compose full circuits by appending measurement circuits to the Ansatz
    ## IMP: Since our Hamiltonian only involves Pauli 'Z' operator and Identity, we don't need the 
    ##      post-rotation measurements to handle X and Y operators.
    try:
        full_circ = ansatz.measure_all(inplace=False)
    except Exception as e:
        log.exception(e, stack_info=True)
        raise
    # let us print the full circuit
    if outputpath_exists == True:
        if ansatz_type == "QAOA" and ansatz_reps > 1:
            log.info(f"QAOA full circuit ansatz too deep to be printed!! Ignoring the exception for now ... ")
        else:
            full_circ.decompose().draw(output='latex', filename=myoutputfilepath + '/' + method + '_FullCirc_' + str(N) + '_' + str(new_subdir_num).zfill(3) + '_' + init_state_label + '_' + ansatz_type + '_' + optimizer_label + '_' + backend_literal + '.png')
    
            log.info(f"Full circuit printed. Check for file {method}_FullCirc_{str(N)}_{str(new_subdir_num).zfill(3)}_{init_state_label}_{ansatz_type}_{optimizer_label}_{backend_literal}.png in the output directory.")

    ## Step 2: Generate transpiled circuits for ideal simulator
    try:
        trans_circ = transpile(full_circ, backend)
    except Exception as e:
        log.exception(e, stack_info=True)
        raise
    log.info(f"Transpiled circuit generated for the backend: {backend}")

    ## Step 3: Attach parameters to the transpiled circuit variables
    bound_circ = trans_circ.assign_parameters(vqe_optimal_params)
    
    ## Step 4: Submit the job and get the resultant counts back
    counts_dict = backend.run(bound_circ, shots=4096).result().get_counts()

    my_var_list = klocal_model.get_var_list()
    my_var_list.reverse()
    log.debug(f"VQE: Variables: {my_var_list}; counts: {counts_dict}")
    if is_exact_solver == True:
        log.debug(f"The exact counts: Variables: {my_var_list};{exact_counts}")

    
    ## let us compute the expectation value based on the raw count data
    expval_no_mit = m3_mit.get_expval(hamiltonian=klocal_model.hamiltonian, 
                                        data_distr=counts_dict)
    log.info(f"Expectation value from raw count distribution - no mitigation: {expval_no_mit}")

    # Step 5: let us process the output to get the bit strings, convert them into decimal values
    #         and get the corresponding probability distribution

    # let us get the individual bit strings and count values
    bit_strings = list(counts_dict.keys()) # need to call the methods keys() and values()
    counts = list(counts_dict.values()) # simultaneously to ensure ordering is same
    #print(counts)
    """
    decimal_output = get_decimal_output(my_var_list, bit_strings=['110101010010111100'],        
                                        var_values_dict = my_var_values_dict, 
                                        p_num_vars=p_num_vars, q_num_vars=q_num_vars)
    log.info(f"Test decimal output is: {decimal_output}")
    """
    # let us get the decimal values
    if method == 'C':
        decimal_output = get_decimal_output(my_var_list, bit_strings, my_var_values_dict, 
                                            p_num_vars, q_num_vars)
        # let us identify if any of the factors have multiple entries and let's consolidate them
        # it is applicable only for 'C' method as we have introduced additional carry variables
        # that may have different values, but the original p and q variables have the same values
        ignore_indices = dict()
        counts_unique = list()
        decimal_output_unique = list()
        counts_index = -1
        for i, val in enumerate(decimal_output):
            for j, check_val in enumerate(decimal_output):
                if ignore_indices.get(i) is None and ignore_indices.get(j) is None:
                    if i == j:
                        # we add the entry to the list
                        counts_unique.append(counts[i])
                        counts_index = counts_index + 1
                        decimal_output_unique.append(val)
                        #print(f"{i}:{counts[i]}: {len(counts_unique)}: {counts_index}")
                    elif i != j and val == check_val:
                        # add the probability values and merge the decimal values into one
                        #print(f"{i},{j}:{counts[j]}, {check_val}")
                        counts_unique[counts_index] += counts[j]
                        ignore_indices[j] = check_val

    elif method == 'D':
        decimal_output = get_decimal_output(my_var_list, bit_strings, None, p_num_vars, 
                                            q_num_vars)
        # for 'D' method, we should never have repeating factors
        decimal_output_unique = decimal_output
        counts_unique = counts
    elif method == 'H':
        decimal_output = get_decimal_output(my_var_list, bit_strings, my_var_values_dict, 
                                            p_num_vars, q_num_vars)
        # for 'H' method, we should never have repeating factors
        decimal_output_unique = decimal_output
        counts_unique = counts
    log.info(f"The decimal output of VQE result without mitigation is: {decimal_output_unique}")

    # let us sort the dictionary in descending order based on count of individual factors
    sort_index = np.argsort(np.asarray(counts_unique))
    sorted_decimal_output = list(np.asarray(decimal_output_unique)[sort_index])
    sorted_counts = list(np.asarray(counts_unique)[sort_index])
    log.debug(f"Sorted Decimal Output: {sorted_decimal_output}; sorted counts: {sorted_counts}")

    ## compute probability distribution
    total_count = sum(val for val in sorted_counts)
    prob_data_no_mit = [val/total_count for val in sorted_counts]
    log.debug(f"Probability distribution with no mitigation: {prob_data_no_mit}")

    ## let us remove entries where probability is less than some threshold value e.g. 0.01
    no_mit_filter_cond = np.where(np.asarray(prob_data_no_mit) > 0.01)
    prob_data_no_mit_curated = list(np.asarray(prob_data_no_mit)[no_mit_filter_cond])
    decimal_output_curated = list(np.asarray(sorted_decimal_output)[no_mit_filter_cond])
    log.info(f"Decimal Output after curating i.e. removing low probability (<= 0.01) values: {decimal_output_curated}")
    log.info(f"Probability distribution after curating i.e. removing low probability (<= 0.01) values: {prob_data_no_mit_curated}")

    #plt = plot_histogram(counts, title=plot_title, color='midnightblue')
    #ax = plt.axes[0]
    #ax.set_xticklabels(counts.keys(), fontsize=8, rotation=45)
    ## Now, plot the relevant distribution along with the factors
    fig_no_mit = plt.figure(figsize=(15,20))
    plt.barh(decimal_output_curated, prob_data_no_mit_curated)
    plt.title("(p,q) for N="+str(N))
    plt.xlabel('Probability')
    plt.ylabel('Factors')
    # save the plot
    hist_plot_location = myoutputfilepath + '/' + method + '_barplot_' + str(N) + '_' + str(new_subdir_num).zfill(3) + '_' + init_state_label + '_' + ansatz_type + '_' + optimizer_label + '_' + backend_literal + '_no_mit'
    tikzplotlib.save(hist_plot_location + '.tex')
    plt.savefig(hist_plot_location + '.png')
    log.info(f"Bar plot generated. Check for file {method}_barplot_{str(N)}_{str(new_subdir_num).zfill(3)}_{init_state_label}_{ansatz_type}_{optimizer_label}_{backend_literal}_no_mit.png in the output directory.")
    plt.close(fig_no_mit)

    if is_apply_m3_mitigation == True and noise_model_device is not None and noise_model_device != "":
        # first apply m3 mitigation
        m3mit = m3_mit(backend=backend, num_qubits=klocal_model.num_qubits)
        m3mit_prob = m3mit.apply_m3_mitigation(counts=counts_dict)
        log.info(f"Probability distribution data post M3 mitigation on raw data: {m3mit_prob}")

        ## let us compute the expectation value based on the prob. distribution post mitigation
        expval_mit = m3_mit.get_expval(hamiltonian=klocal_model.hamiltonian, 
                                        data_distr=m3mit_prob, mitigation_applied=True)
        log.info(f"Expectation value from raw count distribution - M3 mitigation: {expval_mit}")

        mit_bit_strings = list(m3mit_prob.keys()) # need to call the methods keys() and values()
        mit_prob = list(m3mit_prob.values()) # simultaneously to ensure ordering is same

        # let us get the decimal values
        if method == 'C':
            m3_decimal_output = get_decimal_output(my_var_list, mit_bit_strings, my_var_values_dict, 
                                                    p_num_vars, q_num_vars)
            # let us identify if any of the factors have multiple entries and let's consolidate them
            # it is applicable only for 'C' method as we have introduced additional carry variables
            # that my have different values, but the original p and q variables have the same values
            ignore_indices = dict()
            m3_prob_unique = list()
            m3_decimal_output_unique = list()
            counts_index = -1
            for i, val in enumerate(m3_decimal_output):
                for j, check_val in enumerate(m3_decimal_output):
                    if ignore_indices.get(i) is None and ignore_indices.get(j) is None:
                        if i == j:
                            # we add the entry to the list
                            m3_prob_unique.append(mit_prob[i])
                            counts_index = counts_index + 1
                            m3_decimal_output_unique.append(val)
                            #print(f"{i}:{mit_counts[i]}: {len(m3_counts_unique)}: {counts_index}")
                        elif i != j and val == check_val:
                            # add the probability values and merge the decimal values into one
                            #print(f"{i},{j}:{mit_counts[j]}, {check_val}")
                            m3_prob_unique[counts_index] += mit_prob[j]
                            ignore_indices[j] = check_val

        elif method == 'D' or method == 'H':
            m3_decimal_output = get_decimal_output(my_var_list, mit_bit_strings, None, 
                                                    p_num_vars,  q_num_vars)
            # for 'D' method, we should never have repeating factors
            m3_decimal_output_unique = m3_decimal_output
            m3_prob_unique = mit_prob
        elif method == 'H':
            m3_decimal_output = get_decimal_output(my_var_list, mit_bit_strings, my_var_values_dict, 
                                                    p_num_vars,  q_num_vars)
            # for 'H' method, we should never have repeating factors
            m3_decimal_output_unique = m3_decimal_output
            m3_prob_unique = mit_prob
        log.info(f"The decimal output of VQE result post M3 mitigation is: {m3_decimal_output_unique}")

        # let us arrange the data as per the sorting order of the non-mitigated counts
        # let us get the relevant counts and probability distribution along with decimal output
        # Change: Only curated probability data without mitigation is being considered
        
        m3_output_dict = dict(zip(m3_decimal_output_unique, m3_prob_unique))
        prob_data_with_m3mit_curated = list()
        for key in decimal_output_curated:
            if m3_output_dict.get(key) is not None:
                #m3_decimal_output_curated.append(key)
                prob_data_with_m3mit_curated.append(m3_output_dict.get(key))
            else: ## after mitigation, some results might be excluded and hence the original  
                    ## decimal output may not be found in the post mitigation result
                #m3_decimal_output_curated.append(key)
                prob_data_with_m3mit_curated.append(0)
        log.info(f"Probability distribution with M3 mitigation after curated decimal output mapping: {prob_data_with_m3mit_curated}")

        ## Now, plot the relevant distribution along with the factors
        fig_m3_mit = plt.figure(figsize=(15,20))
        bar_width = 0.25
        axis_num = np.arange(len(decimal_output_curated)) 
        plt.barh(axis_num, prob_data_no_mit_curated, bar_width, label='No Mitigation')
        plt.barh(axis_num+bar_width, prob_data_with_m3mit_curated, bar_width, label='M3 Mitigation')
        plt.yticks(ticks=axis_num+bar_width, labels=decimal_output_curated)
        plt.title("(p,q) for N="+str(N))
        plt.xlabel('Probability')
        plt.ylabel('Factors')
        plt.legend(ncol=2)
        # save the plot
        hist_plot_location = myoutputfilepath + '/' + method + '_barplot_' + str(N) + '_' + str(new_subdir_num).zfill(3) + '_' + init_state_label + '_' + ansatz_type + '_' + optimizer_label + '_' + backend_literal + '_m3mit'
        plt.savefig(hist_plot_location + '.png')
        tikzplotlib_fix_ncols(fig_m3_mit) ## fix found online
        tikzplotlib.save(hist_plot_location + '.tex')
        log.info(f"Bar plot generated. Check for file {method}_barplot_{str(N)}_{str(new_subdir_num).zfill(3)}_{init_state_label}_{ansatz_type}_{optimizer_label}_{backend_literal}_m3mit.png in the output directory.")
        plt.close(fig_m3_mit)

    ## now, let us write to an output file, the key findings of this run
    ## define a list to hold all data
    myoutarray = list()
    seconds = time.time()
    myline = f"Current time: {time.ctime(seconds)}\n"
    myoutarray.append(myline)

    ## writing output expressions
    myline = f"N={N}\n"
    myoutarray.append(myline)
    myline = f"Norm expression: {my_expr}\n"
    myoutarray.append(myline)
    myline = f"Hamiltonian Normalized?: {normalize_hamiltonian}\n"
    myoutarray.append(myline)
    myline = f"Hamiltonian: {klocal_model.hamiltonian}\n"
    myoutarray.append(myline)
    myline = f"Variable list corresponding to Hamiltonian: {my_var_list}\n"
    myoutarray.append(myline)

    ## classical output
    myline = f"The exact result using NumPy minimum eigen solver: {exact_result}\n"
    myoutarray.append(myline)
    myline = f"Eigenstate for min. eigen value using NumPy minimum eigen solver: {my_min_eigenstate}\n"
    myoutarray.append(myline)

    ## writing parameters
    myline = f"Initial state label: {init_state_label}\n"
    myoutarray.append(myline)
    myline = f"Ansatz Type: {ansatz_type}\n"
    myoutarray.append(myline)
    if ansatz_type == "CUSTOM":
        myline = f"Rotation blocks: {rotation_layer}\n"
        myoutarray.append(myline)
        myline = f"Entanglement blocks: {entanglement}\n"
        myoutarray.append(myline)    
    myline = f"Ansatz reps/layers: {ansatz_reps}\n"
    myoutarray.append(myline)
    myline = f"Ansatz initial parameter values: {initial_point}\n"
    myoutarray.append(myline)
    myline = f"Optimizer: {optimizer_label}\n"
    myoutarray.append(myline)
    myline = f"VQE eigen value: {vqe_eigen_value}\n"
    myoutarray.append(myline)
    myline = f"VQE optimal parameter values: {vqe_optimal_params}\n"
    myoutarray.append(myline)
    myline = f"VQE Function evaluations: {vqe_func_evals}\n"
    myoutarray.append(myline)
    myline = f"VQE Evaluation time: {vqe_time_taken}\n"
    myoutarray.append(myline)
    myline = f"VQE Evaluation time per function evaluation: {vqe_time_taken/vqe_func_evals}\n"
    myoutarray.append(myline)
    myline = f"Curated VQE factors (no mitigation): X-axis: {decimal_output_curated}\n"
    myoutarray.append(myline)
    myline = f"Curated VQE Probability (no mitigation): Y-axis: {prob_data_no_mit_curated}\n"
    myoutarray.append(myline)
    myline = f"Expectation value from prob. distribution (no mitigation): {expval_no_mit}\n"
    myoutarray.append(myline)
    if is_apply_m3_mitigation == True and noise_model_device is not None and noise_model_device != "":
        myline = f"Noise model used is: {noise_model}\n"
        myoutarray.append(myline)
        myline = f"Coupling map used is: {coupling_map}\n"
        myoutarray.append(myline)
        myline = f"Probability distribution after M3 mitigation: {m3mit_prob}\n"
        myoutarray.append(myline)
        myline = f"VQE factors (post M3 mitigation): X-axis: {m3_decimal_output}\n"
        myoutarray.append(myline)
        myline = f"VQE Probability (post M3 mitigation): Y-axis: {prob_data_with_m3mit_curated}\n"
        myoutarray.append(myline)
        myline = f"Expectation value from prob. distribution (M3 mitigation): {expval_mit}\n"
        myoutarray.append(myline)

    ## start writing now
    with open(myoutputfilepath+"/out_" + str(N) + '_' + str(new_subdir_num).zfill(3) + ".txt","w+") as myoutfile:        
        myoutfile.writelines(myoutarray)

    """
    ### execute the following piece of code for VarQITE algorithm
    if algorithm == 'VARQITE':
        # setup the time evolution problem
        aux_ops = [klocal_model.hamiltonian]
        evolution_problem = TimeEvolutionProblem(hamiltonian=klocal_model.hamiltonian, time=1, aux_operators=aux_ops)
        log.info(f"Evolution problem set for time 1 sec.")
        log.debug(vars(evolution_problem))

        # setup variational QITE
        var_qite = VarQITE(ansatz=ansatz,initial_parameters=initial_point, estimator=Estimator())
        log.info(f"VarQITE set.")
        log.debug(vars(var_qite))

        # now, we run the circuit
        varqite_result = var_qite.evolve(evolution_problem=evolution_problem)
        log.info(f"VarQITE Evolution complete.")
        log.debug(f"VarQITE Result =====> {varqite_result}")

        h_exp_val = np.array([ele[0][0] for ele in varqite_result.observables])
        times = varqite_result.times

        fig_qite = plt.figure(figsize=(15,20))
        plt.plot(times, h_exp_val)
        plt.title("(p,q) for N="+str(N))
        plt.xlabel('Time')
        plt.ylabel('Expectation Value')
        # save the plot
        plot_location = myoutputfilepath + '/' + is_direct_or_column + '_qite_plot_' + str(N) + '_' + str(new_subdir_num).zfill(3) + '_' + init_state_label + '_' + ansatz_type + '_' + optimizer_label + '_' + backend_literal + '_no_mit'
        plt.savefig(plot_location + '.png')
        log.info(f"Plot generated for quantum imaginary-time evolution. Check for file {is_direct_or_column}_qite_plot_{str(N)}_{str(new_subdir_num).zfill(3)}_{init_state_label}_{ansatz_type}_{optimizer_label}_{backend_literal}_no_mit.png in the output directory.")
        plt.close(fig_qite)
    """

if __name__ == '__main__':
    main()    
