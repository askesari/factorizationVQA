# Factorisation of bi-prime integers using Variational Quantum Algorithm

## Description
This project is part of my Master's thesis at the Indian Institute of Technology - Madras. The objective of the project is to leverage binary optimization to factorise bi-prime integers using a variational quantum algorithm.

As part of the project, I have explored the following two methods for creating the cost function for optimization:
1. Direct factorisation method
2. Column-based multiplication method

Further, the cost function is transformed into a Hamiltonian, whose minimum eigenvalue can be estimated using Variational Quantum Eigensolver (VQE). For executing VQE, the following ansatze have been explored:
1. Custom hardware-inspired ansatz
2. Quantum Approximate Optimization Algorithm (QAOA)

The classical optimizer to identify the optimal parameters of the variational circuit can be one of:
1. L-BFGS-B
2. COBYLA
3. SPSA

Finally, if a noise-simulated backend device is chosen, we can apply M3 miigation technique to analyze its impact.

## Pre-requisites
The pre-requisites for installing the package are:

### Python==3.8.13
It is advisable to create a new environment using either pip or conda to deploy the project. 
If using conda, the following command can be used where \<envname> needs to be replaced with the appropriate name during execution. 
    
    conda create --name <envname> python==3.8.13 

### Qiskit packages
- qiskit==0.45.1
    - qiskit-aer==0.11.0
    - qiskit-ibm-experiment==0.2.6
    - qiskit-ibmq-provider==0.19.2
    - qiskit-terra==0.45.1
- qiskit_experiments==0.4.0

### PySCF library
- pyscf==2.1.1

### YAML library
- PyYAML==6.0

### MTHREE library
- mthree==1.1.0

### Matplotlib library
- matplotlib==3.6.0
- pylatexenc==2.10

### tikzplotlib library
- tikzplotlib==0.10.1

Some additional dependencies such as "cm-super" and "dvipng" may need to be installed separately.

One can install all the necessary prerequisite packages and libraries by executing the following command. The requirements.txt file is provided in the repository.

    pip install -r requirements.txt

> Note: The qiskit account credentials must be stored on the machine that contains the program to be executed. Refer to IBM Qiskit help to identify how qiskit account credentials can be stored locally.

## Usage
The program requires several parameters that need to be set before execution. These parameters are present in the "parameters.yaml" file provided along with the package. The parameters are:

    # Allowed N values: a product of 2 prime numbers e.g. 143, 15, 35.
    # Allowed method values: "C" - for column-based multiplication method, "D" - for direct method, "H" - for hybrid method i.e. direct method with classical preprocessor
    # Allowed bit_length_type values: "EQUAL" - if p and q factors have same number of bits, "GENERAL" - no specific assumption
    # Allowed noise_model_device: "" - for ideal simulator, if is_simulator=True, "FAKEATHENS", "FAKEALMADEN", "FAKEMUMBAI", "FAKEMANHATTAN"
    # Allowed init_state: "EQUAL_SUPERPOS" - for all qubits in equal superposition, "ALL_ZEROS" - for all qubits in |0>, "ALL_ONES" - for all qubits in |1>
    # Allowed optimizer_label values: "COBYLA", "L-BFGS-B, "SPSA", "NELDER_MEAD"
    # Allowed optimizer_maxiter: any positive integer for maximum number of iterations/evaluations of the optimizer e.g. 1000, 1500, etc.
    # Allowed optimizer_tol: tolerance value for convergence e.g. 0.01, 1e-4, etc.
    # Allowed ansatz_type values: "CUSTOM" - for hardware-inspired ansatz, "QAOA" - applicable only for algorithm VQE
    # Allowed ansatz_init_param_values: "ZERO" - for all initial parameter values to be 0, "RANDOM" - for all initial parameters to be chosen randomly in [0,2*pi]
    # Allowed ansatz_num_layers: 2 (default), 3, 4, etc. Layers of rotation and entanglement gates for CUSTOM ansatz or alternate repeating layers for QAOA Ansatz
    # Allowed rotations values: ["ry","rz"] (default), "ry"
    # Allowed entanglement values: "cx" - for CNOT gate, "cz" - for Control-Z gate
    # Allowed entanglement_type values: "linear" (default), "full", "reverse_linear", "" - for no entanglement

As an example, refer the following:

    N: 3127
    method: "C"
    bit_length_type: "EQUAL"
    normalize_hamiltonian: "L2"
    is_simulator: True
    noise_model_device: ""
    init_state: "EQUAL_SUPERPOS"
    optimizer_label: "L_BFGS_B"
    optimizer_maxiter: 300
    optimizer_tol: 0.00001
    ansatz_type: "CUSTOM"
    ansatz_init_param_values: "RANDOM"
    ansatz_num_layers: 2
    custom_ansatz:
      rotations: ["ry","rz"]
      entanglement: "cx"
      entanglement_type: "reverse_linear"
      final_rotation_layer: True
    is_exact_solver: True
    is_apply_m3_mitigation: True

To run the program, execute the following command:

    python3 run_integer_factorisation.py


