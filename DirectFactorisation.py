"""
Direct Factorisation class
"""
from sympy import Pow, IndexedBase
import numpy as np
import math
# import custom modules
from logconfig import get_logger
from Factorisation import IntegerFactorisation

## initialize logger
log = get_logger(__name__)

class DirectFactorisation(IntegerFactorisation):
    def __init__(self, N, bit_length_type):
        super().__init__(N, bit_length_type=bit_length_type)
        
        temp_expr = self.N - (self.p * self.q)
        self.direct_expr = temp_expr.expand()

        log.info(f"Direct Factorisation initialized with np = {self.np}, nq = {self.nq}")

    def get_expression(self):
        return(self.direct_expr)

    def get_norm_expression(self, var_values_dict=None):
        """
        This method returns the squared norm of the expression.
        The squared norm is essential for the Hamiltonian formation.
        """
        # generate the squared norm of the expression in expanded form
        temp_expr = Pow(self.direct_expr,2).expand()

        # now, apply the simplification rule i.e. x**2 = x
        temp_expr = DirectFactorisation.apply_rule_simplify(temp_expr)
        log.debug(f"After x**m=x simplify rule: {temp_expr}")

        ## Now, let us apply any values identified as part of classical pre-processing e.g. p1.q2=0, etc. so as to reduce the norm expression further
        if var_values_dict is not None:
            for rule_key, val in var_values_dict.items():
                temp_expr = temp_expr.subs({rule_key:val}).expand()
            log.debug(f"After var-values replacement: {temp_expr}")

        ## Finally, we simplify the expression further by reducing the higher powers e.g. x**2=x
        temp_expr = DirectFactorisation.apply_rule_simplify(temp_expr)
        log.debug(f"After x**m=x simplify rule: {temp_expr}")

        self.simplified_norm_expr = temp_expr
        log.info(f"Norm expression processing complete.")

        return(self.simplified_norm_expr)
    
    @classmethod
    def get_terms_and_coeffs(cls, expr):
        """
        This method returns the individual terms and coefficients in the form of a dictionary 
        with the term as key and the coefficient as value.
        """
        # let us identify the terms and coefficients for each clause S_i
        terms_list = list()
        terms_list.append(expr.as_coefficients_dict())
        
        return(terms_list)
    
    @classmethod
    def apply_rule_simplify(cls, expr):
        """
        Rule: This rule simplifies an expression e.g. x**2 = x, x**3=x, etc.
        """
        log.info(f"Apply Rule simplify begins ...")

        # first, let us get the individual terms and coefficients of the clauses
        terms_list = DirectFactorisation.get_terms_and_coeffs(expr)
        
        # now, let us identify clauses satisfying the rule
        rule_keys = list()
        for term in terms_list:
            found_keys = list()
            for key, _ in term.items():
                if key.func == Pow and key != 1:
                    found_keys.append(key)
                    
            if len(found_keys) > 0:
                rule_keys.extend(found_keys)
                log.debug(f"Found keys: {found_keys}")

        if len(rule_keys) == 0:
            log.info(f"Rule x**m=x simplify: No keys found")

        # now, let us apply this rule to the input expression
        my_expr = expr
        for rule_key in rule_keys:
            ## IMP: Executing the same statement below twice is just a hack so that higher powers
            ##      are also handled effectievely.
            my_expr = my_expr.subs({rule_key:rule_key.args[0]})
            my_expr = my_expr.subs({rule_key:rule_key.args[0]})
        
        log.debug(f"Rule simplify: {my_expr}")

        log.info(f"Rule simplify processing complete.")

        return(my_expr)