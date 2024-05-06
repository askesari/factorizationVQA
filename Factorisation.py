"""
This module helps in defining the IntegerFactorisation parent class.
Since, we are using 2 approaches to solve the integer factorisation problem, we have 2 derived classes, namely DirectFactorisation and ColumnFactorisation.
The problem at hand is:
Given a positive integer N with biprime factors, p and q, we need to identify p and q.
For this, we represent N, p and q in binary format. 
Let 'n_m' is the binary length of the integer N. Thus, 'n_p' can be defined as floor(log N) and 'n_q' as (ceil(1/2*floor(log N))+1) - 1.
We leverage 'sympy' to define the symbols and the symbolic expresssion.
Futher, we apply classical preprocesing rules, to reduce the number of variables/symbols used in the expression.
"""

from sympy import Pow, IndexedBase
from sympy.tensor.indexed import Indexed
import math
# import custom modules
from logconfig import get_logger

## initialize logger
log = get_logger(__name__)

class IntegerFactorisation:
    def __init__(self, N, bit_length_type):
        ## initialise the input number both as an integer and binary list
        self.N = N
        self.binary_N = [int(x) for x in list('{0:0b}'.format(N))]
        self.bit_length_type = bit_length_type
        
        ## initialise the number of bits for each integer i.e. N, p and q
        ## IMP: while the last bit is considered '1' for both p and q, in our definition,
        ##      we will be considering p0 and q0 as part of the total number of bits 
        ##      and then replace them by 1 during the expression evaluation
        self.nm = math.floor(math.log2(N)) + 1
        if N == 15:
            self.np = 3
            self.nq = 3
        else:
            if bit_length_type == 'EQUAL':
                self.np = math.ceil(0.5*math.log2(N))
                self.nq = math.ceil(0.5*math.log2(N))
            elif bit_length_type == 'GENERAL':
                self.np = math.floor(math.log2(N))
                self.nq = math.ceil(0.5 * math.floor(math.log2(N)) + 1)
        ## initialise the symbolic variables for p and q
        self.p_sym=IndexedBase('p')
        self.q_sym=IndexedBase('q')

        ## list of symbols/variables required
        ## IMP: we don't include p0 and q0 as they have a default value of 1.
        ## 
        """
        self.var_list = list()
        for i in range(1, self.np):
            self.var_list.append(self.p_sym[i])
        
        for j in range(1, self.nq):
            self.var_list.append(self.q_sym[j])
        """
        ## define the p and q expressions
        self.p=1 # since p0=1
        for i in range(1, self.np):
            self.p = self.p + Pow(2,i) * self.p_sym[i] 
        
        self.q=1 # since q0=1
        for i in range(1,self.nq):
            self.q = self.q + Pow(2,i) * self.q_sym[i] 

        log.info(f"Base integer factorisation class initialized with np = {self.np}, nq = {self.nq}")

    def get_N(self):
        return(self.N)

    def get_binary_N(self):
        return(self.binary_N)
    
    def get_p(self):
        return(self.p)

    def get_q(self):
        return(self.q)

    def get_p_num_vars(self):
        return self.np
    
    def get_q_num_vars(self):
        return self.nq
    
    @classmethod
    def get_var_list(cls, expr):
        """
        this class method returns the variables used in the input expression
        """
        var_list = list()
        for v in list(expr.free_symbols):
            if v.func == Indexed:
                var_list.append(v)
        
        return(var_list)


