"""
Column Based Factorisation class
"""
from sympy import Pow, IndexedBase, Mul, factor_list
from sympy.tensor.indexed import Indexed
import numpy as np
import math
from itertools import product
# import custom modules
from logconfig import get_logger
from Factorisation import IntegerFactorisation

## initialize logger
log = get_logger(__name__)

class ColumnFactorisation(IntegerFactorisation):
    def __init__(self, N, bit_length_type):
        super().__init__(N, bit_length_type=bit_length_type)

        # define the carry variables s[i,j]
        # IMP: these carry variables also need to be included in the self.var_list
        self.s_sym = IndexedBase('s')

        # set up individual column as a clause
        self.C_list = list()
        # also, let us define a dictionary to hold the values of the variables obtained via
        # classical preprocessing
        self.var_values_dict = dict()

        # we define a matrix M to indicate the position of carry terms generated
        matM = np.zeros((self.np + self.nq + 1, self.np + self.nq + 1))
        
        # also, we need to have a reverse copy of self.binary_N to ensure that
        # the correct clauses have the appropriate bit
        reverse_binary_N = self.binary_N.copy()
        reverse_binary_N.reverse()

        for i in range(1, self.np + self.nq + 1):
            column_expr = 0
            # define the p*q terms
            # the starting point is li=max(i-np+1,0) assuming np>=nq 
            l = max(i - self.np + 1, 0)
            nl = 0 # number of p*q terms
            for j in range(l, min(self.nq,i+1)):
                column_expr = column_expr - self.q_sym[j] * self.p_sym[i-j]
                nl = nl + 1

            # define the input carry terms 
            # we know C_1 clause does not have an input carry.
            # so, we only consider the input carry from C_2 clause
            # also, we leverage the matrix M (matM) used for identifying the output carries
            # from previous clauses to determine the input carries for the subsequent clauses
            ns = 0 # number of input carry terms
            if i > 1:
                for j in range(1, i):
                    if matM[j][i] == 1:
                        column_expr = column_expr - self.s_sym[j,i]
                        ns = ns + 1
                   
            # define the output carry terms
            # we leverage the matrix M defined outside the loop for identifying the generated 
            # output carry positions
            if (nl + ns) > 0: # if no input terms, then output is also zero
                if i>1:
                    # the starting point is j=1, but end point is mi = ceil(log2(nl + ns))
                    temp_m = math.log2(nl + ns)
                    if temp_m.is_integer():
                        m = int(temp_m) + 1
                    else:
                        m = math.ceil(temp_m)

                    for j in range(1, m):
                        column_expr = column_expr + Pow(2,j) * self.s_sym[i,i+j]
                        matM[i][i+j] = 1
                        # include the carry term in the variable list self.var_list
                        #self.var_list[self.s_sym[i,i+j]] = {'ising':1}
                else: # a carry can be generated from S_1 clause
                    # we manually set the matM[1][2] value as 1
                    column_expr = column_expr + Pow(2,i) * self.s_sym[i,i+1]
                    matM[i][i+1] = 1
                    # include the carry term in the variable list self.var_list
                    #self.var_list[self.s_sym[i,i+1]] = {'ising':1}

            # finally define the output Ni term i.e. bit in ith position of N
            if(i<self.nm):
                column_expr = column_expr + reverse_binary_N[i]

            if column_expr != 0:
                # now, set p0=q0=1
                column_expr = column_expr.subs({self.p_sym[0]:1, self.q_sym[0]:1})

                # also, if bit_length_type = EQUAL, set the value at max position to 1
                if self.bit_length_type == 'EQUAL' and self.N != 15:
                    column_expr = column_expr.subs({self.p_sym[self.np-1]:1, self.q_sym[self.nq-1]:1})
                    self.var_values_dict[self.p_sym[self.np-1]] = 1
                    self.var_values_dict[self.q_sym[self.nq-1]] = 1
                self.C_list.append(column_expr)

        log.info(f"Column Based Factorisation initialized with np = {self.np}, nq = {self.nq}, ns= {np.count_nonzero(matM)} and Ci clauses = {len(self.C_list)}")

    def get_var_values_dict(self):
        """
        Returns the values/expressions set for the variables as part of classical pre-processing
        """
        return self.var_values_dict
    
    def get_column_clauses(self):
        """
        returns the column clauses/expressions
        """
        return(self.C_list)
    
    def get_terms_and_coeffs(self):
        """
        This method breaks down each clause C_i into individual terms and coefficients and 
        returns in the form of a dictionary with the term as key and the coefficient as value.
        """
        # let us identify the terms and coefficients for each clause S_i
        clause_terms_list = list()
        for clause in self.C_list:
            clause_terms_list.append(clause.as_coefficients_dict())
        
        return(clause_terms_list)

    @classmethod
    def apply_rule_1(cls, terms_dict):
        """
        Rule: x1+x2+...+xn = n => x_i = 1 
           or -x1-x2-...-xn = -n => x_i = 1
           or a1.x1+a2.x2+..+am.xm = n, where sum(am) = n => x_i=1
        """
        log.debug(f"Apply Rule 1 begins for {terms_dict} ...")

        # now, identify terms satisfying the rule
        rule_keys = list()
        # initialise the appropriate values
        positive_coeff = 0
        negative_coeff = 0
        const_coeff = 0
        is_found = True
        found_keys = list()
        for key, coeff in terms_dict.items():
            if key == 1:
                const_coeff = const_coeff + coeff
            elif coeff < 0:
                found_keys.append((key,1))
                negative_coeff = negative_coeff + abs(coeff)
            elif coeff > 0:
                found_keys.append((key,1))
                positive_coeff = positive_coeff + coeff
            else:
                is_found = False
                break

        if len(found_keys) > 0:
            if (is_found == True and positive_coeff == 0 and negative_coeff == const_coeff) or \
                (is_found == True and negative_coeff == 0 and positive_coeff == (-1)*const_coeff):
                rule_keys = found_keys
                log.info(f"Rule 1: Found keys {found_keys}")

        return(rule_keys)

    @classmethod
    def apply_rule_2(cls, terms_dict):
        """
        Rule: x1 + x2 + x3 + ... = 0 => all x_i = 0 OR
              -x1 -x2 -x3 -... = 0 => all x_i = 0 
        """
        log.debug(f"Apply Rule 2 begins for {terms_dict} ...")

        # now, identify clauses satisfying the rule
        rule_keys = list()
        # initialise the appropriate values    
        positive_coeff = 0
        negative_coeff = 0
        const_coeff = 0
        is_found = True
        found_keys = list()
        for key, coeff in terms_dict.items():
            if key == 1:
                const_coeff = const_coeff + coeff
            elif coeff == -1:
                negative_coeff = negative_coeff + abs(coeff)
                found_keys.append((key,0))
            elif coeff == 1:
                positive_coeff = positive_coeff + coeff
                found_keys.append((key,0))
            else:
                is_found = False
                break

        if (is_found == True and negative_coeff > 0 and positive_coeff == 0 and const_coeff == 0) or \
            (is_found == True and positive_coeff > 0 and negative_coeff == 0 and const_coeff == 0):
            rule_keys = found_keys
            log.info(f"Rule 2: Found keys {found_keys}")

        return(rule_keys)

    @classmethod
    def apply_rule_3(cls, terms_dict):
        """
        Rule: x1 + x2 = 2*x3 => all x1=x2=x3
              Special: 2*x1 = 2*x3
        """
        log.debug(f"Apply Rule 3 begins for {terms_dict} ...")

        # now, identify clauses satisfying the rule
        rule_keys = list()
        if len(terms_dict) == 3 or len(terms_dict) == 2:
            # initialise appropriate values
            positive_coeff = 0
            positive_coeff_terms = 0 
            negative_coeff = 0
            negative_coeff_terms = 0
            is_found = True
            found_keys = list()
            for key, coeff in terms_dict.items():
                if key == 1:
                    is_found = False
                    break
                elif coeff < 0:
                    negative_coeff = negative_coeff + abs(coeff)
                    negative_coeff_terms += 1
                    found_keys.append(key)
                else:
                    positive_coeff = positive_coeff + coeff
                    positive_coeff_terms += 1
                    found_keys.append(key)

            if (is_found == True) and (negative_coeff == positive_coeff) and \
                (positive_coeff_terms == 1 or negative_coeff_terms == 1) and len(found_keys) != 0:
                main_key = found_keys[0]
                for i in range(1, len(found_keys)):
                    rule_keys.append(tuple([found_keys[i],main_key]))
                log.info(f"Rule 3: Found keys {rule_keys}")

        return(rule_keys)

    @classmethod
    def apply_rule_4(cls, terms_dict):
        """
        Rule: -x1-x2-...-xn + c1*y1 + c2*y2 + ... cm*ym + c0, c_i's are constants 
                        => if c_i > abs(negative_coeff) - c0, then y_i = 0
              OR
              -c1*x1-c2*x2-...-cn*xn + y1 + y2 + ... ym + c0, c_i's are constants 
                        => if c_i > positive_coeff + c0, then x_i = 0
              
              A specific case: x1 + x2 = 2*x3 + 1 => x3 = 0
        """
        log.debug(f"Apply Rule 4 begins for {terms_dict} ...")

        # now, identify clauses satisfying the rule
        rule_keys = list()
        # initialise appropriate values
        pos_const_coeff = 0
        neg_const_coeff = 0
        negative_coeff = 0
        positive_coeff = 0
        found_keys = list()
        for key, coeff in terms_dict.items():
            if key == 1:
                if coeff > 0:
                    pos_const_coeff = coeff
                else:
                    neg_const_coeff = abs(coeff)
            elif coeff < 0:
                negative_coeff = negative_coeff + abs(coeff)
            elif coeff > 0:
                positive_coeff = positive_coeff + coeff

        # compute the value to verify. 
        # Here, negative_coeff has to be greater than pos_const_coeff; otherwise it is an invalid condition
        for key, coeff in terms_dict.items():
            if key != 1 and coeff > max(0, (negative_coeff + neg_const_coeff - pos_const_coeff)):
                found_keys.append((key,0))
            elif key != 1 and abs(coeff) > (positive_coeff + pos_const_coeff):
                found_keys.append((key,0))

        if len(found_keys) > 0:
            rule_keys = found_keys
            log.info(f"Rule 4: Found keys {found_keys}")

        return(rule_keys)    

    @classmethod
    def apply_rule_5(cls, terms_dict):
        """
        Rule: x1 + x2 = 1 => x1.x2 = 0 or x2 = 1-x1
        """
        log.debug(f"Apply Rule 5 begins for {terms_dict} ...")

        # now, identify clauses satisfying the rule
        rule_keys = list()
        # initialise appropriate values
        if len(terms_dict) == 3:
            const_coeff = 0
            found_coeff = 0
            is_found = True
            found_keys = list()
            for key, coeff in terms_dict.items():
                if key == 1:
                    const_coeff = const_coeff + coeff
                elif coeff == -1:
                    found_coeff = found_coeff + abs(coeff)
                    found_keys.append(key)
                else:
                    is_found = False
                    break

            if (is_found == True) and (found_coeff == 2) and (const_coeff == 1):
                #my_key = 1
                #for key in found_keys:
                #    my_key = my_key * key
                #rule_keys = [my_key]
                main_key = found_keys[0]
                subs_key = found_keys[1]
                rule_keys.append((subs_key,(1-main_key)))
                log.info(f"Rule 5: Found keys {rule_keys}")

        return(rule_keys)

    
    @classmethod
    def apply_rule_6(cls, terms_dict):
        """
        Rule: -x1-x2-x3 ... -xn - c0 + c1.y1 + c2.y2, , c_i's are constants 
                => if c1 + c2 > abs(negative_coeff) + c0 and c0 !=0, then y2=1-y1
        """
        log.debug(f"Apply Rule 6 begins for {terms_dict} ...")

        # now, identify clauses satisfying the rule
        rule_keys = list()
        # initialise appropriate values
        const_coeff = 0
        negative_coeff = 0
        is_found = False
        positive_coeff_keys = list()
        for key, coeff in terms_dict.items():
            if key == 1:
                const_coeff = const_coeff + coeff
            elif coeff < 0:
                negative_coeff = negative_coeff + abs(coeff)
            elif coeff > 0 and key.func == Indexed:
                log.debug(f"Check 6: {key.args}")
                positive_coeff_keys.append((key, coeff))
                is_found = True

        # now, identify if sum of positive coeffs of 2 single variables is more than the 
        # check_coeff value and const_coeff is not zero

        if (is_found == True) and const_coeff < 0 and len(positive_coeff_keys) == 2:
            key_coeff1 = positive_coeff_keys[0]
            key_coeff2 = positive_coeff_keys[1]
            if key_coeff1[1] + key_coeff2[1] > (negative_coeff - const_coeff):
                main_key = key_coeff1[0]
                subs_key = key_coeff2[0]
                rule_keys.append((subs_key, (1-main_key)))
            if len(rule_keys) > 0:
                log.info(f"Rule 6: Found keys {rule_keys}")

        return(rule_keys)


    @classmethod
    def apply_rule_parity(cls, terms_dict):
        """
        Rule: -a1.x1 - a2.x2 - a3.x3 ... - an.xn + b1.y1 + b2.y2 + b3.y3 + .. + bm.ym + b0
                => Parity of neg coeff terms = Parity of pos coeff terms
        """
        log.debug(f"Apply Rule Parity begins for {terms_dict} ...")

        # now, identify clauses satisfying the rule
        rule_keys = list()
        # initialise appropriate values
        neg_parity = 0
        neg_even_parity_count = 0
        pos_parity = 0
        pos_even_parity_count = 0
        check_neg_parity_terms = list()
        check_pos_parity_terms = list()
        for key, coeff in terms_dict.items():
            if key == 1:
                if coeff > 0:
                    pos_parity = (pos_parity + coeff)%2
                else:
                    neg_parity = (neg_parity + abs(coeff))%2

            elif coeff < 0:
                if (abs(coeff))%2 == 0:
                    neg_parity = (neg_parity + abs(coeff))%2
                    neg_even_parity_count += 1
                else:
                    check_neg_parity_terms.append(key)
            
            elif coeff > 0:
                if coeff%2 == 0:
                    pos_parity = (pos_parity + coeff)%2
                    pos_even_parity_count += 1
                else:
                    check_pos_parity_terms.append(key)

        # now, identify if we can locate 2 terms (both with neg coeff. or both with pos coeff.)
        # with parity 0 or 1
        # IMP: It is essential that we apply the rule to expression where terms containing both
        #      negative and positive coefficients are present (excluding constant terms)
        if (len(check_pos_parity_terms) == 0 and pos_even_parity_count > 0 and 1 <= len(check_neg_parity_terms) <= 2):
            if abs(pos_parity - neg_parity) == 0: # even parity
                if len(check_neg_parity_terms) == 1:
                    my_key = (check_neg_parity_terms[0],0)
                else: # there are 2 terms
                    my_key = (check_neg_parity_terms[1],check_neg_parity_terms[0])
            else: # odd parity
                if len(check_neg_parity_terms) == 1:
                    my_key = (check_neg_parity_terms[0],1)
                else: # there are 2 terms
                    my_key = (check_neg_parity_terms[1], (1-check_neg_parity_terms[0]))
            rule_keys.append(my_key)
        elif (1 <= len(check_pos_parity_terms) <= 2  and len(check_neg_parity_terms) == 0 and neg_even_parity_count > 0):
            if abs(pos_parity - neg_parity) == 0: # even parity
                if len(check_pos_parity_terms) == 1:
                    my_key = (check_pos_parity_terms[0],0)
                else: # there are 2 terms
                    my_key = (check_pos_parity_terms[1],check_pos_parity_terms[0])
            else: # odd parity
                if len(check_pos_parity_terms) == 1:
                    my_key = (check_pos_parity_terms[0],1)
                else: # there are 2 terms
                    my_key = (check_pos_parity_terms[1], (1-check_pos_parity_terms[0]))
            rule_keys.append(my_key)

        if len(rule_keys) > 0:
            log.info(f"Rule Parity: Found keys {rule_keys}")

        return(rule_keys)


    @classmethod
    def apply_rule_simplify(cls, terms_dict):
        """
        Rule: This rule simplifies an expression e.g. x**2 = x, x**3=x, etc.
        """
        log.debug(f"Apply Rule simplify begins for {terms_dict} ...")

        # now, let us check if the expression satsifies the rule
        rule_keys = list()
        for term, coeff in terms_dict.items():
            if term.func == Pow and term != 1:
                rule_keys.append(term)
            elif term.func == Mul and term != 1:
                ## power may be present as part of a term which would be a tuple
                for subterm in term.args:
                    if subterm.func == Pow and subterm != 1:
                        rule_keys.append(subterm)
                    
        if len(rule_keys) > 0:
                log.info(f"Found keys: {rule_keys}")

        return(rule_keys)
    

    @classmethod
    def apply_rule_special_1a(cls, terms_dict, clause):
        """ 
        Rule Special 1a. 2xy - x - y => value is either -1 or 0
                      b. xy - y => value is -y if x = 0 else 0
        If the above rules gives us additional constraints irrespective of values of x and y,
        then we can simplify the expressions further.
        """
        log.debug(f"Apply rule special 1a begins for clause {clause} and {terms_dict}...")

        # now, let us check if the expression satisfies the rule (a)
        rule_keys = list()
        found_keys = list()
        donotrepeat_keys_dict = dict()
        for term, coeff in terms_dict.items():
            log.debug(f"check special 1a: {term.func}, {term.args}, {len(term.args)}")
            if term.func == Mul and len(term.args) == 2 and coeff == 2: 
                # if product of 2 terms with positive coeff. = 2
                key1 = term.args[0]
                key2 = term.args[1]
                if donotrepeat_keys_dict.get(key1) is None and donotrepeat_keys_dict.get(key2) is None:
                    found_key1 = False 
                    found_key2 = False
                    # check if both -key1 and -key2 is present in the entire clause
                    for check_key, check_coeff in terms_dict.items():
                        if check_coeff == -1 and check_key == key1:
                            found_key1 = True
                        elif check_coeff == -1 and check_key == key2:
                            found_key2 = True
                        if found_key1 == True and found_key2 == True:
                            log.debug(f"Found keys {(key1,key2)}")
                            found_keys.append((key1,key2))
                            donotrepeat_keys_dict[key1] = 1
                            donotrepeat_keys_dict[key2] = 1
                            break
        
        # if keys found for rule (a), then process it for the clause to derive possible constraints
        if len(found_keys) > 0:
            my_iterator = list(product(np.array([0,-1]), repeat=len(found_keys)))
            log.debug(f"My iterator: {my_iterator}")

            for ind, iter_val in enumerate(my_iterator):
                verify_keys = list()
                my_expr = clause
                # replace 2xy -x - y by appropriate values in the iterator
                for i, t_key in enumerate(found_keys):
                    replace_expr = 2*t_key[0]*t_key[1] - t_key[0] - t_key[1]
                    log.debug(f"replace_expr for iterator {iter_val[i]}: {replace_expr}")
                    # build expr for replace_expr = iterator value
                    my_expr = my_expr.subs({replace_expr:iter_val[i]})
                    log.debug(f"my_expr: {my_expr}")

                my_expr_terms = my_expr.as_coefficients_dict()

                # now, apply each of the existing regular rules and see if some common pattern or 
                # constraint can be derived.
                verify_keys.extend(ColumnFactorisation.apply_rule_1(terms_dict=my_expr_terms))
                verify_keys.extend(ColumnFactorisation.apply_rule_2(terms_dict=my_expr_terms))
                verify_keys.extend(ColumnFactorisation.apply_rule_3(terms_dict=my_expr_terms))
                verify_keys.extend(ColumnFactorisation.apply_rule_4(terms_dict=my_expr_terms))
                verify_keys.extend(ColumnFactorisation.apply_rule_5(terms_dict=my_expr_terms))
                verify_keys.extend(ColumnFactorisation.apply_rule_6(terms_dict=my_expr_terms))
                verify_keys.extend(ColumnFactorisation.apply_rule_parity(terms_dict=my_expr_terms))

                ## now compare the results with the prior results of the iterator
                if ind == 0 and len(verify_keys) > 0:
                    rule_keys = verify_keys
                elif ind == 0 and len(verify_keys) == 0:
                    log.debug(f"No keys found at ind: {ind}")
                    rule_keys = list()
                    break
                elif ind > 0 and len(verify_keys) > 0:
                    # compare the current iteration results with the previous one
                    rule_keys = list(set(rule_keys).intersection(verify_keys))
                    if len(rule_keys) == 0:
                        log.debug(f"No keys found after intersection at ind: {ind}")
                        break
                else:
                    log.debug(f"No keys found at ind: {ind}")
                    rule_keys = list()
                    break

        if len(rule_keys) > 0:
            log.info(f"Found final keys: {rule_keys}")
                
        return(rule_keys)
    

    @classmethod
    def apply_rule_special_1b(cls, terms_dict, clause):
        """ 
        Rule Special 1a. 2xy - x - y => value is either -1 or 0
                     1b. xy - y => value is -y if x = 0 else 0
        If the above rule gives us additional constraints irrespective of values of x and y,
        then we can simplify the expressions further.
        """
        log.debug(f"Apply rule special 1b begins for clause {clause} and {terms_dict}...")

        # now, let us check if the expression satisfies the rule (b)
        rule_keys = list()
        found_keys = list()
        donotrepeat_keys_dict = dict()
        for term, coeff in terms_dict.items():
            log.debug(f"check special 1b: {term.func}, {term.args}, {len(term.args)}")
            if term.func == Mul and len(term.args) == 2 and coeff == 1: 
                # if product of 2 terms with positive coeff. = 1
                key1 = term.args[0]
                key2 = term.args[1]
                product_key = key1*key2
                if donotrepeat_keys_dict.get(product_key) is None:
                    found_key1 = False 
                    found_key2 = False
                    # check if either -key1 or -key2 is present in the entire clause
                    for check_key, check_coeff in terms_dict.items():
                        if check_coeff == -1 and check_key == key1:
                            found_key1 = True
                        elif check_coeff == -1 and check_key == key2:
                            found_key2 = True
                            
                        if found_key1 == True or found_key2 == True:
                            if found_key1 == True: 
                                # capture (xy,x) for the expression xy - y
                                found_keys.append((product_key,key2))
                                log.debug(f"Found keys {(product_key,key2)}")
                            else: # implies found_key2 is True
                                # capture (xy,x) for the expression xy - y
                                found_keys.append((product_key,key1))
                                log.debug(f"Found keys {(product_key,key1)}")
                            donotrepeat_keys_dict[product_key] = 1
                            break
        
        # if keys found for rule (b), then process it for the clause to derive possible constraints
        if len(found_keys) > 0:
            my_iterator = list(product(np.array([0,1]), repeat=len(found_keys)))
            log.debug(f"My iterator: {my_iterator}")

            for ind, iter_val in enumerate(my_iterator):
                log.debug(f"Executing for iterator: {iter_val}")
                verify_keys = list()
                my_expr = clause
                # replace y from (xy -x) by appropriate values in the iterator
                for i, t_key in enumerate(found_keys):
                    replace_expr = t_key[1]
                    log.debug(f"replace_expr for iterator {iter_val[i]}: {replace_expr}")
                    # build expr for replace_expr = 0
                    my_expr = my_expr.subs({replace_expr:iter_val[i]})
                    log.debug(f"my_expr: {my_expr}")

                my_expr_terms = my_expr.as_coefficients_dict()
                # now, apply each of the existing regular rules and see if some common pattern or 
                # constraint can be derived.
                verify_keys.extend(ColumnFactorisation.apply_rule_1(terms_dict=my_expr_terms))
                verify_keys.extend(ColumnFactorisation.apply_rule_2(terms_dict=my_expr_terms))
                verify_keys.extend(ColumnFactorisation.apply_rule_3(terms_dict=my_expr_terms))
                verify_keys.extend(ColumnFactorisation.apply_rule_4(terms_dict=my_expr_terms))
                verify_keys.extend(ColumnFactorisation.apply_rule_5(terms_dict=my_expr_terms))
                verify_keys.extend(ColumnFactorisation.apply_rule_6(terms_dict=my_expr_terms))
                verify_keys.extend(ColumnFactorisation.apply_rule_parity(terms_dict=my_expr_terms))

                ## now compare the results with the prior results of the iterator
                if ind == 0 and len(verify_keys) > 0:
                    rule_keys = verify_keys
                elif ind == 0 and len(verify_keys) == 0:
                    log.debug(f"No keys found at ind: {ind}")
                    rule_keys = list()
                    break
                elif ind > 0 and len(verify_keys) > 0:
                    # compare the current iteration results with the previous one
                    rule_keys = list(set(rule_keys).intersection(verify_keys))
                    if len(rule_keys) == 0:
                        log.debug(f"No keys found after intersection at ind: {ind}")
                        break
                else:
                    log.debug(f"No keys found at ind: {ind}")
                    rule_keys = list()
                    break

        if len(rule_keys) > 0:
            log.info(f"Found final keys: {rule_keys}")
                
        return(rule_keys)
    

    def apply_rule_special_final(self):
        """ 
        Rule Special final: Here we try to replace carry variable 's' that has coefficient
                            either +1 or -1 by the rest of the clause expression
        """
        log.info(f"Start of special rule for final processing ...")
        continue_iter = 1
        iter_no = 1
        while(continue_iter == 1):

            log.info(f"Special final rule iteration: {iter_no} ...")
            continue_iter = 0 # reset to 0 so that it can set to 1 if appropriate key is found

            ## Step 1: get the terms and coefficients for each clause and check if there is
            ##         any carry 's' variable with coefficient +1 / -1 and that 's' variable
            ##         is present in some other clause
            clause_terms_list = self.get_terms_and_coeffs()
            for clause_no, clause_term_dict in enumerate(clause_terms_list):
                is_found = False
                is_found_valid = False

                for key, coeff in clause_term_dict.items():
                    if key != 1 and key.args[0] == IndexedBase('s') and coeff in [1,-1]:
                        log.debug(f"Key {key} for clause term: {clause_term_dict}")
                        found_key = key
                        found_coeff = coeff
                        is_found = True
                        break
            
                # Step 2: check if the found key is present in some other clause
                #         we only need to check for terms with found key as single variable,
                #         as we shouldn't get 's' variables in product form ----> Check !!!!
                if is_found == True:
                    for t_clause_no, t_clause_term_dict in enumerate(clause_terms_list):
                        if clause_no != t_clause_no:
                            for key, coeff in t_clause_term_dict.items():
                                if key != 1 and key == found_key:
                                    log.debug(f"Found {key} is valid.")
                                    is_found_valid = True
                                    break
                            if is_found_valid == True:
                                break
                
                # Step 3: if found key is valid, replace the key by the rest of the clause and
                #         apply the substitution to all clauses
                if is_found_valid == True:
                    if found_coeff == -1:
                        replace_expr = (self.C_list[clause_no] + found_key)
                        log.debug(f"replace_expr is {replace_expr}")
                        for i in range(len(self.C_list)):
                            my_expr = self.C_list[i]
                            self.C_list[i] = my_expr.subs({found_key:replace_expr}).expand()
                    elif found_coeff == 1:
                        replace_expr = -1 * (self.C_list[clause_no] - found_key)
                        log.debug(f"replace_expr is {replace_expr}")
                        for i in range(len(self.C_list)):
                            my_expr = self.C_list[i]
                            self.C_list[i] = my_expr.subs({found_key:replace_expr}).expand()
                    else:
                        log.exception(f"Oops!! Something seems to be wrong here ...")
                    
                    continue_iter = 1
                    iter_no += 1

                    # print debug statements
                    for s_no, clause in enumerate(self.C_list):
                        log.debug(f"Clause {s_no+1}: {clause}")
                    
                    break


    def classical_preprocessing(self, num_iterations=20):
        """
        This method applies classical pre-processing rules to the individual columns so as to reduce the number of variables and hence the number of qubits.
        num_iterations: There can be multiple passes over the clauses using the same rules to ensure that rules applied in previous iteration can reduce the expression variables in the subsequent passes.
        """
        
        ## following variable used to check if the processing has to continue or not
        ## in short, if no changes possible, we can stop the iterative process.
        continue_iter = 0

        for iter in range(num_iterations):
            log.info(f"Pass {iter+1} of the classical processing rules ... ")

            ### apply rule 1 i.e. (x1+x2+..+xn = n or -x1-x2-...-xn = -n) => x_i = 1
            # first, let us get the individual terms and coefficients of the clauses
            rule_keys = list()
            clause_terms_list = self.get_terms_and_coeffs()
            for clause_terms in clause_terms_list:
                rule_keys.extend(ColumnFactorisation.apply_rule_1(terms_dict=clause_terms))
                
            # now, let us apply this rule to all the clauses
            if len(rule_keys) == 0 :
                log.info(f"Rule 1: No new keys found")
            else:
                for rule_key in rule_keys:
                    for i in range(len(self.C_list)):
                        my_expr = self.C_list[i]
                        log.debug(f"Check 1: {rule_key[0].args}, {rule_key}, {rule_key[0].func}")
                        if rule_key[0].func == Indexed: # implies single variable
                            self.C_list[i] = my_expr.subs({rule_key[0]:rule_key[1]})
                            self.var_values_dict[rule_key[0]] = rule_key[1]
                        else: # implies product of 2 variables
                            my_key1 = rule_key[0].args[0]
                            my_key2 = rule_key[0].args[1]
                            self.C_list[i] = my_expr.subs({my_key1:rule_key[1],my_key2:rule_key[1]})
                            self.var_values_dict[my_key1] = rule_key[1]
                            self.var_values_dict[my_key2] = rule_key[1]
                continue_iter = 1 # ensure that iteration continues
            log.info(f"Rule 1 processing complete.")

            # print debug statements
            for s_no, clause in enumerate(self.C_list):
                log.debug(f"Clause {s_no+1}: {clause}")

            ### apply rule 2 i.e. (x1+x2+x3+...=0 or -x1-x2-x3-..=0) => x_i = 0 
            # again, let us get the individual terms and coefficients of the clauses
            rule_keys = list()
            clause_terms_list = self.get_terms_and_coeffs()
            for clause_terms in clause_terms_list:
                rule_keys.extend(ColumnFactorisation.apply_rule_2(terms_dict=clause_terms))
                
            # now, let us apply this rule to all the clauses
            if len(rule_keys) == 0 :
                log.info(f"Rule 2: No new keys found")
            else:
                for rule_key in rule_keys:
                    for i in range(len(self.C_list)):
                        self.C_list[i] = self.C_list[i].subs({rule_key[0]:rule_key[1]})
                    self.var_values_dict[rule_key[0]] = rule_key[1]
                continue_iter = 1 # ensure that iteration continues
            log.info(f"Rule 2 processing complete.")
            
            # print debug statements
            for s_no, clause in enumerate(self.C_list):
                log.debug(f"Clause {s_no+1}: {clause}")
                

            ### apply rule 3 i.e. (x1+x2+x3+...+xn = n.y or -x1-x2-x3-..-xn = -n.y) => 
            ###                                                   x_1 = x_2 = ... = y 
            ### IMP: currently this rule is applied only for 3 terms i.e. x1+x2=2.y => x1=x2=y
            ###      and 2 terms i.e. a1.x1=a1.x2
            # again, let us get the individual terms and coefficients of the clauses
            rule_keys = list()
            clause_terms_list = self.get_terms_and_coeffs()
            for clause_terms in clause_terms_list:
                found_keys = ColumnFactorisation.apply_rule_3(terms_dict=clause_terms)
                # ensure that empty lists are not added to rule_keys list
                if len(found_keys) > 0:
                    rule_keys.extend(found_keys)
                
            if len(rule_keys) == 0:
                log.info(f"Rule 3: No new keys found.")
            else:
                # now, let us apply this rule to all the clauses
                log.debug(f"Rule 3: {rule_keys}")
                for rule_key in rule_keys:
                    log.debug(f"Check 3: {rule_key}")
                    log.debug(f"Check 3: {rule_key[0]}, {rule_key[1]}")
                    for i in range(len(self.C_list)):
                        my_expr = self.C_list[i]
                        self.C_list[i] = my_expr.subs({rule_key[0]:rule_key[1]})
                        # we add each individual mapping to var_values_dict
                        self.var_values_dict[rule_key[0]] = rule_key[1]
                continue_iter = 1 # ensure that iteration continues
            log.info(f"Rule 3 processing complete.")

            # print debug statements
            for s_no, clause in enumerate(self.C_list):
                log.debug(f"Rule 3: Clause {s_no+1}: {clause}")

            #### apply rule 4
            # -x1-x2-...-xn + c1*y1 + c2*y2 + ... cn*yn + c0, c_i's are constants 
            # => if c_i > abs(negative_coeff) - c0, then y_i = 0
            # a specific case: (x+y=2z+1 => z=0)
            # IMP: here, y_i can be a single variable or a product of 2 variables, but still
            #      we cannot split the term.
            # again, let us get the individual terms and coefficients of the clauses
            rule_keys = list()
            clause_terms_list = self.get_terms_and_coeffs()
            for clause_terms in clause_terms_list:
                rule_keys.extend(ColumnFactorisation.apply_rule_4(terms_dict=clause_terms))

            if len(rule_keys) == 0:
                log.info(f"Rule 4: No new keys found.")
            else:
                # now, let us apply this rule to all the clauses
                for rule_key in rule_keys:
                    for i in range(len(self.C_list)):
                        my_expr = self.C_list[i]
                        #log.debug(f"Check 4: {rule_key.args}, {rule_key}, {rule_key.func}")
                        # if rule_key[0].func == Indexed: # implies single variable
                        #     self.C_list[i] = my_expr.subs({rule_key[0]:rule_key[1]})
                        # else: # implies product of 2 variables
                        #     my_key1 = rule_key[0].args[0]
                        #     my_key2 = rule_key[0].args[1]
                        #     self.C_list[i] = my_expr.subs({my_key2: (1-my_key1)}).expand()
                        self.C_list[i] = my_expr.subs({rule_key[0]:rule_key[1]}).expand()    
                    self.var_values_dict[rule_key[0]] = rule_key[1]
                continue_iter = 1 # ensure that iteration continues
            log.info(f"Rule 4 processing complete.")
            
            # print debug statements
            for s_no, clause in enumerate(self.C_list):
                log.debug(f"Rule 4: Clause {s_no+1}: {clause}")


            #### apply rule 5 i.e. (x1+x2=1 => x1.x2=0 or x2=1-x1)
            # again, let us get the individual terms and coefficients of the clauses
            rule_keys = list()
            clause_terms_list = self.get_terms_and_coeffs()
            for clause_terms in clause_terms_list:
                rule_keys.extend(ColumnFactorisation.apply_rule_5(terms_dict=clause_terms))

            if len(rule_keys) == 0:
                log.info(f"Rule 5: No new keys found.")
            else:
                # now, let us apply this rule to all the clauses
                # for rule_key in rule_keys:
                #     # check if the rule_key is not processed in the previous iteration(s)
                #     if self.var_values_dict.get(rule_key) is None:
                #         for i in range(len(self.C_list)):
                #             self.C_list[i] = self.C_list[i].subs({rule_key:0})
                #         self.var_values_dict[rule_key] = 0
                #         continue_iter = 1 # ensure that iteration continues
                for rule_key in rule_keys:
                    for i in range(len(self.C_list)):
                        my_expr = self.C_list[i]
                        self.C_list[i] = my_expr.subs({rule_key[0]:rule_key[1]}).expand()
                    self.var_values_dict[rule_key[0]] = rule_key[1]
                    continue_iter = 1 # ensure that iteration continues
            log.info(f"Rule 5 processing complete.")
            
            # print debug statements
            for s_no, clause in enumerate(self.C_list):
                log.debug(f"Rule 5: Clause {s_no+1}: {clause}")


            #### apply rule 6
            #
            # Rule: -x1-x2-x3 ... -xn -c0 + c1.y1 + c2.y2, c_i's are constants 
            #    => if c1 + c2 > abs(negative_coeff) - c0 and c0 != 0, then y2 = 1-y1
            #
            # again, let us get the individual terms and coefficients of the clauses
            rule_keys = list()
            clause_terms_list = self.get_terms_and_coeffs()
            for clause_terms in clause_terms_list:
                rule_keys.extend(ColumnFactorisation.apply_rule_6(terms_dict=clause_terms))

            if len(rule_keys) == 0:
                log.info(f"Rule 6: No new keys found.")
            else:
                # now, let us apply this rule to all the clauses 
                for rule_key in rule_keys:
                    # check if the rule_key is not processed in the previous iteration(s)
                    if self.var_values_dict.get(rule_key[0]) is None:
                        for i in range(len(self.C_list)):
                            my_expr = self.C_list[i]
                            self.C_list[i] = my_expr.subs({rule_key[0]:rule_key[1]})
                        self.var_values_dict[rule_key[0]] = rule_key[1]
                        continue_iter = 1 # ensure that iteration continues
                    else:
                        log.debug(f"Rule 6: Key {rule_key} already processed in prior iteration.")
            log.info(f"Rule 6 processing complete.")

            # print debug statements
            for s_no, clause in enumerate(self.C_list):
                log.debug(f"Rule 6: Clause {s_no+1}: {clause}")

            #### apply rule parity
            #
            # Rule: -a1.x1 - a2.x2 - a3.x3 ... -an.xn + b1.y1 + b2.y2 + ... + bm.ym + b0,
            #    => if parity of neg. coeff. terms = parity of pos. coeff. terms
            #
            # again, let us get the individual terms and coefficients of the clauses
            rule_keys = list()
            clause_terms_list = self.get_terms_and_coeffs()
            for clause_terms in clause_terms_list:
                rule_keys.extend(ColumnFactorisation.apply_rule_parity(terms_dict=clause_terms))

            if len(rule_keys) == 0:
                log.info(f"Rule Parity: No new keys found.")
            else:
                # now, let us apply this rule to all the clauses 
                for rule_key in rule_keys:
                    for i in range(len(self.C_list)):
                        my_expr = self.C_list[i]
                        self.C_list[i] = my_expr.subs({rule_key[0]:rule_key[1]}).expand()
                    self.var_values_dict[rule_key[0]] = rule_key[1]
                    continue_iter = 1 # ensure that iteration continues
            log.info(f"Rule Parity processing complete.")

            # print debug statements
            for s_no, clause in enumerate(self.C_list):
                log.debug(f"Rule Parity: Clause {s_no+1}: {clause}")

            #### apply rule simplify i.e. x**2 = x 
            # again, let us get the individual terms and coefficients of the clauses
            rule_keys = list()
            clause_terms_list = self.get_terms_and_coeffs()
            for clause_terms in clause_terms_list:
                rule_keys.extend(ColumnFactorisation.apply_rule_simplify(terms_dict=clause_terms))

            if len(rule_keys) == 0:
                log.info(f"Rule x**m=x: No keys found")
            else:
                # now, let us apply this rule to all the clauses    
                for rule_key in rule_keys:
                    for i in range(len(self.C_list)):
                        self.C_list[i] = self.C_list[i].subs({rule_key:rule_key.args[0]})
                continue_iter = 1 # ensure that iteration continues

            ### check if any constant factor for each clause can be eliminated
            for i in range(len(self.C_list)):
                my_expr = self.C_list[i]
                my_factor_list = factor_list(self.C_list[i])
                log.debug(f"Factor list: {my_factor_list}")
                if my_factor_list[0] not in [-1,0,1]:
                    self.C_list[i] = my_expr / abs(my_factor_list[0])
                    continue_iter = 1
            log.info(f"Rule simplify processing complete.")

            # print debug statements
            for s_no, clause in enumerate(self.C_list):
                log.debug(f"Rule simplify: Clause {s_no+1}: {clause}")

            
            ### apply special rules
            # special rule 1a
            rule_keys = list()
            for clause in self.C_list:
                clause_terms = clause.as_coefficients_dict()
                rule_keys.extend(ColumnFactorisation.apply_rule_special_1a(terms_dict=clause_terms, clause=clause))

            if len(rule_keys) == 0:
                log.info(f"Rule special 1a: No keys found")
            else:
                # now, let us apply this rule to all the clauses    
                for rule_key in rule_keys:
                    for i in range(len(self.C_list)):
                        self.C_list[i] = self.C_list[i].subs({rule_key[0]:rule_key[1]})
                continue_iter = 1 # ensure that iteration continues
            
            # print debug statements
            for s_no, clause in enumerate(self.C_list):
                log.debug(f"Rule special 1a: Clause {s_no+1}: {clause}")

            # special rule 1b
            rule_keys = list()
            for clause in self.C_list:
                clause_terms = clause.as_coefficients_dict()
                rule_keys.extend(ColumnFactorisation.apply_rule_special_1b(terms_dict=clause_terms, clause=clause))

            if len(rule_keys) == 0:
                log.info(f"Rule special 1b: No keys found")
            else:
                # now, let us apply this rule to all the clauses    
                for rule_key in rule_keys:
                    for i in range(len(self.C_list)):
                        self.C_list[i] = self.C_list[i].subs({rule_key[0]:rule_key[1]})
                continue_iter = 1 # ensure that iteration continues
            
            # print debug statements
            for s_no, clause in enumerate(self.C_list):
                log.debug(f"Rule special 1b: Clause {s_no+1}: {clause}")


            # check whether the next iteration has to be executed or not
            if continue_iter == 1:
                continue_iter = 0
            else:
                break
        
        # apply final special rule to process the carry 's' variables
        self.apply_rule_special_final()
        # print debug statements
        for s_no, clause in enumerate(self.C_list):
            log.debug(f"Rule special final: Clause {s_no+1}: {clause}")


        return(iter+1)

        
    def get_norm_expression(self):
        """
        This method returns the simplified squared norm of the expression.
        The squared norm is essential for the Hamiltonian formation.
        """
        log.info(f"Generating norm expression ...")
        
        # generate the squared value of the expression in expanded form
        # here, we square each individual clause and finally add them all together
        temp_expr = None
        for expr in self.C_list:
            if temp_expr is None:
                temp_expr = Pow(expr,2).expand()
            else:
                temp_expr = temp_expr + Pow(expr,2).expand()
        log.debug(f"Temp norm expression is: {temp_expr}")

        # now, we simplify the expression based on the simplification rule i.e. x**2=x1.
        # IMP: It is important to not apply any other simplification rules as they have already
        #      been covered in the individual column expressions. Adding the other rules, may also
        #      result in the Ising Hamiltonian having eigen value < 0.
        # first, let us get the terms and coefficients of the norm expression
        terms_and_coeff = temp_expr.as_coefficients_dict()
        
        rule_keys = ColumnFactorisation.apply_rule_simplify(terms_and_coeff)
        if len(rule_keys) == 0:
            log.info(f"Rule x**m=x simplify: No keys found")

        # now, let us apply this rule to all the clauses    
        for rule_key in rule_keys:
            ## IMP: Executing the same statement below twice is just a hack so that higher powers
            ##      are also handled effectievely.
            temp_expr = temp_expr.subs({rule_key:rule_key.args[0]})
            temp_expr = temp_expr.subs({rule_key:rule_key.args[0]})
        log.debug(f"After x**m=x simplify rule: {temp_expr}")

        """
        ## now, let us apply any values identified as part of classical pre-processing e.g. p1.q2=0, etc. so as to reduce the norm expression further
        for rule_key, val in self.var_values_dict.items():
            temp_expr = temp_expr.subs({rule_key:val})
        log.debug(f"After var-values replacement: {temp_expr}")

        # Finally, let us apply simplify rule i.e. x**2=x again
        # first, let us get the terms and coefficients of the norm expression
        terms_and_coeff = temp_expr.as_coefficients_dict()
        
        rule_keys = ColumnFactorisation.apply_rule_simplify(terms_and_coeff)
        if len(rule_keys) == 0:
            log.info(f"Rule x**m=x simplify: No keys found")

        # now, let us apply this rule to all the clauses    
        for rule_key in rule_keys:
            ## IMP: Executing the same statement below twice is just a hack so that higher powers
            ##      are also handled effectievely.
            temp_expr = temp_expr.subs({rule_key:rule_key.args[0]})
            temp_expr = temp_expr.subs({rule_key:rule_key.args[0]})
        log.debug(f"After x**m=x simplify rule: {temp_expr}")
        """
        
        self.simplified_norm_expr = temp_expr
        log.info(f"Norm expression processing complete.")
        
        return(self.simplified_norm_expr)
    
    