"""
File: ladders.py
Author: Ja-Eun Kim
Date: 2025-07-30
Description: Automates calculations involving multimode noncommutative Bosonic ladder operators.
"""

import warnings
import numpy as np


class Expression:
    """
    **Multimode noncommutative Bosonic ladder operator calculator
    with good ol' string manipulation**

    Adds, multiplies, and expands quantum mechanical expressions 
    written in terms of Bosonic ladder operators.
    Maintains normal ordering (Wick ordering) by 
    applying the Bosonic commutation relation [a, a+] = 1.

    Data structure: For example, by initializing an instance by Expression("2a+_a(+)b+_b(+)1")
    - Expression.expr_dict: 
        Dictionary that has strings which represent each term of an expression as keys, 
        and the (complex) coefficients as values. 
        - Example: {"a+_a":2, "b+_b":1, "":1}
        - Note: key "" is used for constants. 
            {"":1} means constant 1, where an empty dictionary, {}, means 0.
    - Expression.modes: a list of characters that represents the modes involved in the Expression.
        - Example: ['a', 'b']

    Syntax rules for initializing an Expression instance:
    An operator should be a single letter that's not j. (j is reserved for the imaginary number)
        - Example: "a_a+(+)3+4.ja(+)b"
        - Creation operators are in the form of "a+" (you can use any letter)
        - Annihilation operators are in the form of "a" (you can use any letter)
        - Multiplication of operators are expressed with "_"
        - Additions are written as "(+)" (because "+" is for dagger)
        - Coefficients are multiplied in front of the operators

    Algorithm: Exactly as how you would it with pen and paper. 
        - Step 1. Expand products of Expressions by multiplying each term
        - Step 2. Convert each term to normal order using the commutation relation
        - Step 3. Collect like terms (same term string) and sum their coefficients
    
    TODO:
        - Caching (memoization):
            - Since the return values (dictionaries) are mutable and unhashable, 
                memoization breaks the code. 
            - Implement efficient cache passing when creating a new Expression instance 
                by adding or multiplying Expression instances
        - Pretty printing
        - Implement getter and setter for 'expr_dict' and 'modes' to protect internal states
        - Handle mode-dependent coefficient factors (ex. 1/sqrt(omega_1) and 1/sqrt(omega_z))
    """

    def __init__(self, expr_string=""):
        self.expr_dict = self.parse_expr_string(expr_string)
        self.modes = self.find_modes(self.expr_dict)  # modes involved
        self.cache = None  # memoization for normal_order()
        self.LOGGING = False

    @classmethod
    def from_dict(cls, expr_dict, modes=None):
        """
        Alternative constructor: build an Expression straight from a
        {term string: coefficient} dictionary, bypassing string parsing.

        inputs:
          expr_dict: a {term string: coefficient} dictionary. It is copied, so
                  the caller keeps ownership of the dictionary they passed in.
          modes: (list or None) the modes involved. Recomputed from 'expr_dict'
                  unless given. Pass it only when the modes are already known to
                  be correct (e.g. scalar multiplication, which introduces no
                  new mode), to skip the scan over every term.

        returns:
          A new instance of 'cls', so subclasses stay their own type
        """
        new = cls("")
        new.expr_dict = dict(expr_dict)
        new.modes = list(modes) if modes is not None else new.find_modes(new.expr_dict)
        return new

    def copy(self):
        """
        A new Expression holding the same terms.

        Only the containers ('expr_dict' and 'modes') are duplicated; the
        coefficients themselves are shared. That is safe because every
        coefficient type in use is immutable (complex numbers and sympy
        expressions), and it keeps the copy cheap even for a large symbolic
        expression, where copy.deepcopy() would rebuild every sympy tree.

        'LOGGING' is carried over, being user configuration rather than
        expression data. 'cache' is shared on purpose: normal_order() is a pure
        function of the term string, so an entry computed for one Expression is
        valid for any other. Keep it that way -- if anything expression-specific
        is ever stored in 'cache', sharing it here becomes a bug.

        returns:
          A new Expression instance, of the same class as self
        """
        new = type(self).from_dict(self.expr_dict, self.modes)
        new.LOGGING = self.LOGGING
        new.cache = self.cache
        return new

    def __copy__(self):
        return self.copy()

    def __deepcopy__(self, memo):
        # A deep copy is indistinguishable from a shallow one here: the
        # coefficients are immutable, so sharing them cannot be observed.
        new = self.copy()
        memo[id(self)] = new  # repeated references to self map to a single copy
        return new

    def parse_expr_string(self, expr_string):
        """
        When initializing an Expression instance,
        parse the user input string to create an Expression instance 
        (= a dictionary with key as terms, values as coefficients)
        """
        terms = expr_string.split("(+)")
        expr_dict = {}

        if expr_string == "":  # empty initialization (corresponds to a constant 0)
            return expr_dict

        for term in terms:
            # separate the coefficient from the operators
            first_alphabet_idx = self.find_first_alphabet_index(term)  # index of the first operator
            operators = term[first_alphabet_idx:]
            if first_alphabet_idx == -1:  # constant term
                expr_dict[""] = complex(term)
            elif first_alphabet_idx == 0:
                expr_dict[operators] = 1
            else:
                coeff_string = term[:first_alphabet_idx]
                expr_dict[operators] = complex(coeff_string)

        return expr_dict

    def find_first_alphabet_index(self, term_string):
        """
        Find the index of the first operator.
        term_string[:first_alphabet_idx]: coefficient
        term_string[first_alphabet_idx:]: operators
        """
        first_alphabet_idx = -1
        for idx, char in enumerate(term_string):
            if char.isalpha() and char != "j":
                first_alphabet_idx = idx
                break
        return first_alphabet_idx

    def find_modes(self, expr_dict):
        """
        Given a dictionary representation of an expression, find the modes involved.
        """
        modes = []
        for term in expr_dict.keys():
            operators = term.strip("_+")  # keep the annihilation operators only
            for char in operators:
                if char.isalpha() and (char not in modes):
                    modes.append(char)
        return sorted(modes)  # alphabetical order

    def add(self, expr2):
        """
        Add itself and expr2

        returns:
          A new Expression instance with the result
        """
        result = Expression("")

        for term, coeff in self.expr_dict.items():
            result.expr_dict[term] = coeff

        for term, coeff in expr2.expr_dict.items():
            if term in result.expr_dict.keys():
                result.expr_dict[term] += coeff
            else:
                result.expr_dict[term] = coeff

        result.modes = result.find_modes(result.expr_dict)

        return result

    def __add__(self, other_expr):
        return self.add(other_expr)

    def __sub__(self, other_expr):
        return self.add(scalar_multiply(other_expr, -1))

    def multiply(self, other_expr):
        """
        Multiply other_expr to the RIGHT of this expression

        returns:
          A new Expression instance with the result
        """
        print(
            "Called multiply(): ", self.expr_dict, other_expr.expr_dict
        ) if self.LOGGING else None

        result = Expression()

        # combine modes of both Expressions
        result.modes = sorted(
            set(self.modes + other_expr.modes)
        )

        # expand each term in both Expressions
        for term1, coeff1 in self.expr_dict.items():
            for term2, coeff2 in other_expr.expr_dict.items():
                print(
                    "\n<<< Multiplying term1: ",
                    coeff1,
                    term1,
                    "  term2: ",
                    coeff2,
                    term2,
                    ">>>",
                ) if self.LOGGING else None

                # Step 1. Multiply two terms by simply concatenating the strings
                # Here, the order of operators are all mixed up yet:
                # e.g., "a_b+_a+_b" (operators are not grouped by mode, and are not in normal order)
                unordered_operators = self.clean_string(term1 + "_" + term2)
                original_coeff = coeff1 * coeff2

                # if both terms were constant, the rest of code won't execute
                # so manually add this constant term to the result
                if term1 == "" and term2 == "":
                    self.add_expr_dicts(result.expr_dict, {"": original_coeff})
                    continue

                # Step 2. Rearrange operators so that those of the same mode are grouped together
                mode_grouped_term = self.group_by_mode(unordered_operators)
                print(
                    "Grouped by modes: ", mode_grouped_term
                ) if self.LOGGING else None

                # Step 3. For each mode, convert to normal order using the commutation relation
                temp = {}
                # ex. single_mode_term = "a_a+_a+" (not in normal order yet)
                for single_mode_term in self.split_by_mode(mode_grouped_term):
                    print(
                        "- single_mode_term: ", single_mode_term
                    ) if self.LOGGING else None
                    mode = single_mode_term[0]  # the first character is the mode letter
                    normal_ordered_single_mode_dict = self.normal_order(
                        mode, single_mode_term, 1
                    )
                    print(
                        "-- normal_ordered_single_mode_dict: ",
                        normal_ordered_single_mode_dict,
                    ) if self.LOGGING else None

                    if not temp:
                        temp = normal_ordered_single_mode_dict
                    else:
                        temp = self._multiply_dicts(
                            temp, normal_ordered_single_mode_dict
                        )

                # temp now has the normal ordered terms for ALL modes
                # and the coefficients that resulted from the commutation relations
                # now multiply the original coefficient to each term
                for multimode_term, commutation_coeff in temp.items():
                    temp[multimode_term] = commutation_coeff * original_coeff

                self.add_expr_dicts(
                    result.expr_dict, temp
                )  # add the result to the final expression dictionary

        print(
            "Result after multiplying ",
            other_expr.expr_dict,
            " is: ",
            result.expr_dict,
        ) if self.LOGGING else None
        return result

    def __mul__(self, other_expr):
        return self.multiply(other_expr)

    def _multiply_dicts(self, dict1, dict2):
        """
        Multiply two expression dictionaries.
        This is a helper function for the multiply method.

        input:
          dict1: first dictionary
          dict2: second dictionary
        returns:
          a new dictionary with the product of the two dictionaries
        """
        result = {}
        for key1, value1 in dict1.items():
            for key2, value2 in dict2.items():
                new_key = self.clean_string(key1 + "_" + key2)  # combine keys
                if new_key in result:
                    result[new_key] += value1 * value2
                else:
                    result[new_key] = value1 * value2
        return result

    def group_by_mode(self, term_string):
        """
        Given a string representation of a SINGLE term consisting of multiple modes,
        collect the same modes together and rearrange them in alphabetical order.

        input: 
          term_string: key of expr_dict. (No coefficients, no (+).)
        
        example:
          group_by_mode("a_b+_a+_b"): "a_a+_b+_b"
          (The result is not yet in normal order. normal_order() takes care of it.)
        """
        mode_grouped_term = ""

        # go over each mode in term_string
        # (find_modes gives a list of mode letters in alphabetical order)
        for mode in self.find_modes({term_string: 1}):
            operators = term_string.split("_")
            for operator in operators:  # an operator could be either creation or annihilation
                if mode in operator:
                    # if this operator is in the mode that we're looking for,
                    # copy it to the mode grouped term string
                    mode_grouped_term += operator + "_"

        return mode_grouped_term[:-1]  # remove the trailing "_"

    def normal_order(self, mode, single_mode_term, coeff):
        """
        Given a string representation of a term consisting of a single mode,
        use the commutation relation to organize it in normal order
        (creation operators to the left, annihilation operators to the right).

        input:
          mode: the letter that represents the operator (annihilator)
          single_mode_term: string representation of a SINGLE mode term (without coeff)
          coeff: its coefficient

        returns:
          a dictionary with organized terms and coefficients
        """

        n = mode + "+_" + mode   # number operator in this mode
        n_dag = mode + "_" + mode + "+"

        idx = single_mode_term.find(n_dag)   # any occurrence of n_dag should be rewritten as (n+1)

        if idx == -1:  # already organized, return sonomamadae
            # print("Already organized: ", single_mode_term)
            return {single_mode_term: coeff}

        else:  # apply the commutation relation
            left = self.clean_string(single_mode_term[:idx])
            right = self.clean_string(single_mode_term[idx + len(n_dag) :])
            # print("left: ", left, "right: ", right)

            # commutation relation: replace n_dag with (n+1)
            term_with_n = self.clean_string(left + "_" + n + "_" + right)
            term_with_1 = self.clean_string(left + "_" + right)
            # print("term_with_n: ", term_with_n, "  term_with_1: ", term_with_1)

            result = self.add_expr_dicts(
                self.normal_order(mode, term_with_n, coeff),
                self.normal_order(mode, term_with_1, coeff),
            )
            return result

    def clean_string(self, term_string):
        """
        Clean the term_string by removing extra underscores.
        """
        if term_string.endswith("_"):
            term_string = term_string[:-1]
        if term_string.startswith("_"):
            term_string = term_string[1:]
        term_string = term_string.replace(
            "__", "_"
        )  # remove double underscores

        return term_string

    def add_expr_dicts(self, dict1, dict2):
        """
        Add two expression dictionaries together.
        If there is a common key, the coefficients are added together.

        input:
          dict1: first expression dictionary
          dict2: second expression dictionary
        returns:
          dict1: updated first expression dictionary with the sum
        """
        for key, value in dict2.items():
            if key in dict1:
                dict1[key] += value
            else:
                dict1[key] = value
        return dict1

    def split_by_mode(self, multi_mode_string):
        """
        Split a string with multiple modes into a list of terms w.r.t. each mode.

        inputs:
        multi_mode_string: a string containing multiple modes (e.g., "a_a+_b_b+").
             Should already be in alphabetical order (by running group_by_mode() beforehand)
        returns:
          list of term strings for each Hilbert space.
        """
        modes = self.find_modes({multi_mode_string: 1})
        # 1 is a dummy coefficient because find_modes takes an expression dictionary as input

        splitted_expressions = []
        for mode in modes:
            # find the first occurrence of the mode
            idx = multi_mode_string.find(mode)
            assert idx != -1

            # the substring ending at idx is a term string that consists of the PREVIOUS mode
            single_mode_string = self.clean_string(multi_mode_string[:idx])
            splitted_expressions.append(single_mode_string)

            # discard the part that's already copied to splitted_expressions
            multi_mode_string = self.clean_string(multi_mode_string[idx:])

        # append the remaining string (term string of the last mode)
        splitted_expressions.append(self.clean_string(multi_mode_string))
        splitted_expressions.remove("")  # remove any empty strings

        return splitted_expressions

    def count_order(self, mode, multi_mode_string):
        """
        Count the number of occurrences of a mode in a multi-mode string.

        inputs:
          mode: the character representing the mode (e.g., 'a')
          multi_mode_string: a string containing multiple modes (e.g., "a_a+_b_b+")
        returns:
          the count of the mode in the string
        """
        return multi_mode_string.count(mode)

    def unitary_transform(self, generator, order=25, dagger=False,
                          convergence_tol=1e-9):
        """
        Unitary transformation by U = e^S, where S is the given generator.
        Computed with the Hadamard lemma (BCH expansion), truncated at 'order':
            e^S A e^-S = A + [S, A] + [S, [S, A]]/2! + [S, [S, [S, A]]]/3! + ...

        Pass the EXPONENT of the unitary as the generator!

        Example: the squeezing operator of Crimin et al. 2021 (eq. 23),
            S(zeta) = exp( -(zeta/2) a+_a+ + (conj(zeta)/2) a_a ),  zeta = r e^{i phi},
        is built by squeeze_generator(mode, zeta), and
            A.unitary_transform(squeeze_generator("a", zeta))
        computes S A S+, e.g. a -> cosh(r) a + e^{i phi} sinh(r) a+ (their eq. 24).

        Note: the series is a Taylor truncation, so for squeezing the cosh/sinh
        coefficients are only accurate up to 'order' powers of r.
        For a quadratic generator the cost is LINEAR in 'order' (such a
        generator preserves operator degree, so the expression never grows),
        which is why the default is generous: order=25 reaches machine
        precision for squeezing parameters up to r ~ 5 in a few milliseconds.
        A non-quadratic generator does grow the expression, so raise 'order'
        with more care if the generator's order > 2.

        inputs:
          generator: an Expression instance, the exponent S of the unitary U = e^S
          order: (int) truncation order of the BCH series
          dagger: (bool) Notation! If False (default), computes U A U+ = e^S A e^-S
                  if True, computes U+ A U = e^-S A e^S
          convergence_tol: (float or None) warn if 
                  (the last term of the series) / result > convergence_tol.
                  Pass None to silence the check (e.g. when a low-order truncation 
                  is deliberate, as in perturbation theory). Skipped automatically 
                  when the coefficients are symbolic (cannot be compared numerically).

        returns:
          A new Expression instance with the result
        """
        S = scalar_multiply(generator, -1) if dagger else generator

        # a fresh copy of self, for the first term "A" in the series
        result = self.copy()

        # will hold the n-fold nested commutator [S, [S, ... [S, A]]]
        # start with a fresh copy of self, to get ready for the first commutator [S, A]
        nested = self.copy()
        factorial = 1
        last_term = None  # most recent term added to the series

        for n in range(1, order + 1):
            nested = commutator(S, nested)
            factorial *= n

            # For performance, prune terms whose coefficients cancelled to zero in the commutator.
            # Without this, dead terms are multiplied again on the next iteration
            # and their operator strings grow, making normal_order() blow up.
            nested.expr_dict = {
                term: coeff for term, coeff in nested.expr_dict.items() if coeff != 0
            }

            # the series terminates once a nested commutator vanishes (e.g. [S, A] = 0)
            # this is an EXACT result, so no convergence warning is warranted
            if not nested.expr_dict:
                last_term = None
                break

            last_term = scalar_multiply(nested, 1 / factorial)
            result = result.add(last_term)

        if convergence_tol is not None and last_term is not None:
            self._warn_if_unconverged(last_term, result, order, convergence_tol)

        return result

    def _warn_if_unconverged(self, last_term, result, order, convergence_tol):
        """
        Warn if the BCH series of unitary_transform() was still contributing
        significantly when it was cut off at 'order'.

        Compares the size of the last term in series against the size of the whole
        result. Silently does nothing if the coefficients are symbolic, since
        they cannot then be compared numerically.
        """
        tail = largest_coefficient(last_term)
        total = largest_coefficient(result)

        if tail is None or total is None or total == 0:
            return  # symbolic coefficients, or an identically zero result

        ratio = tail / total
        if ratio > convergence_tol:
            warnings.warn(
                f"unitary_transform(): the BCH series may not have converged by "
                f"order={order}. The last term in the series is {ratio:.2e} of the result "
                f"(tolerance {convergence_tol:.0e}), so the answer is likely "
                f"inaccurate at about that relative level. Increase 'order', or "
                f"pass convergence_tol=None if this truncation is deliberate.",
                stacklevel=3,
            )

def squeeze_generator(mode, zeta):
    """
    Build the exponent of the single-mode squeezing operator S(zeta),
    in the convention of Crimin et al. 2021 (eq. 23):
        S(zeta) = exp( -(zeta/2) a+_a+ + (conj(zeta)/2) a_a ),  zeta = r exp(i phi)

    Pass the result to Expression.unitary_transform(), which then computes
    S A S+, reproducing their eq. (24):
        a -> cosh(r) a + exp(i phi) sinh(r) a+

    Beware the sign convention: the creation term carries -zeta/2 and the
    annihilation term +conj(zeta)/2. Flipping them squeezes the other quadrature.

    inputs:
      mode: (str) the single letter naming the mode, e.g. "a"
      zeta: the squeezing parameter r * exp(1j * phi).
            May be real, complex, or a sympy expression.

    returns:
      An Expression instance, to be used as the 'generator' argument
      of Expression.unitary_transform()
    """
    # works for Python numbers, numpy scalars and sympy expressions alike
    zeta_conj = zeta.conjugate() if hasattr(zeta, "conjugate") else np.conj(zeta)

    return scalar_multiply(Expression(mode + "_" + mode), zeta_conj / 2) + \
           scalar_multiply(Expression(mode + "+_" + mode + "+"), -zeta / 2)

def largest_coefficient(expr):
    """
    Largest absolute coefficient in an Expression, as a float.
    A rough measure of the size of an expression.

    Returns None if any coefficient is symbolic (a sympy expression with free
    symbols), because those cannot be compared numerically. Callers should
    treat None as "no numeric comparison possible" rather than as zero.
    """
    magnitude = 0.0
    for coeff in expr.expr_dict.values():
        try:
            magnitude = max(magnitude, abs(complex(coeff)))
        except (TypeError, ValueError):
            return None  # symbolic coefficient
    return magnitude

def power(expr, exponent):
    """
    Power.

    inputs:
        expr: an Expression instance
        exponent: (int)
    
    returns a new Expression instance
    """
    result = expr.copy()
    for _ in range(exponent - 1):
        result = result.multiply(expr)

    # no need to find_modes() because no new mode is introduced

    return result


def scalar_multiply(expr, scalar):
    """
    Returns a fresh Expression instance after scalar multiplication.
    """
    # modes are passed straight through: scalar multiplication introduces no new mode
    return type(expr).from_dict(
        {term: coeff * scalar for term, coeff in expr.expr_dict.items()},
        modes=expr.modes,
    )


def print_nonzero_terms(expr):
    """
    Print non-zero terms of an Expression instance.
    """
    print("Non-zero terms:")
    for key, value in expr.expr_dict.items():
        if value != 0:
            print(key, ": \t", value) if key != "" else print(
                "constant: \t", value
            )


def compare_expr(expr1, expr2, tolerance=1e-9):
    """
    Compares two Expressions, checking if values are close.
    Prints terms that are in one dictionary but not in the other, or terms
    where the complex values differ by more than the specified tolerance.
    """
    IS_EQUAL = True

    dict1 = expr1.expr_dict
    dict2 = expr2.expr_dict

    keys1 = set(dict1.keys())
    keys2 = set(dict2.keys())

    # Check for keys in dict1 but not in dict2
    for key in keys1 - keys2:
        print(f"Key '{key}' in first dictionary but not in second.")
        IS_EQUAL = False

    # Check for keys in dict2 but not in dict1
    for key in keys2 - keys1:
        print(f"Key '{key}' in second dictionary but not in first.")
        IS_EQUAL = False

    # Check for keys in both dictionaries
    for key in keys1.intersection(keys2):
        value1 = dict1[key]
        value2 = dict2[key]

        # Use numpy's isclose for comparing complex numbers
        if not np.isclose(value1, value2, atol=tolerance):
            print(
                f"Values differ for key '{key}': {value1} (expr1) vs {value2} (expr2)"
            )
            IS_EQUAL = False

    return IS_EQUAL


def print_kerr(expr):
    """
    Identify and print the self-Kerr and cross-Kerr terms.
    Currently works for hard-coded a,b,z modes.

    TODO add "modes" argument (type str) and auto generate the corresponding Kerr strings
    """

    print(
        "a mode self-Kerr: ", expr.expr_dict["a+_a+_a_a"]
    ) if "a+_a+_a_a" in expr.expr_dict.keys() else print(
        "a mode self-Kerr: ", 0
    )
    print(
        "b mode self-Kerr: ", expr.expr_dict["b+_b+_b_b"]
    ) if "b+_b+_b_b" in expr.expr_dict.keys() else print(
        "b mode self-Kerr: ", 0
    )
    print(
        "magnetron-cyclotron cross-Kerr: ", expr.expr_dict["a+_a_b+_b"]
    ) if "a+_a_b+_b" in expr.expr_dict.keys() else print(
        "magnetron-cyclotron cross-Kerr: ", 0
    )

    print(
        "z mode self-Kerr: ", expr.expr_dict["z+_z+_z_z"]
    ) if "z+_z+_z_z" in expr.expr_dict.keys() else print(
        "z mode self-Kerr: ", 0
    )
    print(
        "z-a cross-Kerr: ", expr.expr_dict["a+_a_z+_z+"]
    ) if "a+_a_z+_z+" in expr.expr_dict.keys() else print("z-a cross-Kerr: ", 0)
    print(
        "z-b cross-Kerr: ", expr.expr_dict["b+_b_z+_z+"]
    ) if "b+_b_z+_z+" in expr.expr_dict.keys() else print("z-b cross-Kerr: ", 0)


def commutator(expr1, expr2):
    """
    Compute the commutator [expr1, expr2] = expr1*expr2 - expr2*expr1

    inputs:
      expr1: an Expression instance
      expr2: an Expression instance

    returns:
      A new Expression instance with the result
    """
    return expr1*expr2 - expr2*expr1