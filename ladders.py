"""
File: ladders.py
Author: Ja-Eun Kim
Date: 2025-07-30
Description: Automates calculations involving multimode noncommutative Bosonic ladder operators.
"""

import copy
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
        self.cache = None  # memoization for normal_ordering()
        self.LOGGING = False

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
            if first_alphabet_idx == 0:
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
        first_alphabet_idx = 0
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

                # organize the term in Hilbert space order
                hspace_organized_term = self.hspace_organize(
                    term1 + "_" + term2
                )
                print(
                    "hilbert_space_organized: ", hspace_organized_term
                ) if self.LOGGING else None

                if term1 == "" and term2 == "":
                    # if both terms are constant, the string manipulation doesn't execute
                    # so manually multiply the coefficients and add to the result
                    self.add_expr_dicts(result.expr_dict, {"": coeff1 * coeff2})
                    continue

                temp = {}
                for single_mode_term in self.split_modes(hspace_organized_term):
                    print(
                        "- single_mode_term: ", single_mode_term
                    ) if self.LOGGING else None
                    mode = single_mode_term[0]  # the first character is the mode
                    normal_ordered_single_mode_dict = self.normal_ordering(
                        mode, single_mode_term, 1
                    )
                    print(
                        "-- normal_ordered_single_mode_dict: ",
                        normal_ordered_single_mode_dict,
                    ) if self.LOGGING else None
                    if not temp:
                        temp = normal_ordered_single_mode_dict
                    else:
                        temp = self._multiply_normal_ordered_single_mode_dicts(
                            temp, normal_ordered_single_mode_dict
                        )

                overall_coeff = (
                    coeff1 * coeff2
                )  # multiply the initial coefficients
                for key, value in temp.items():
                    temp[key] = (
                        value * overall_coeff
                    )  # multiply the coefficient to each term

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

    def __multiply__(self, other_expr):
        return self.multiply(other_expr)

    def _multiply_normal_ordered_single_mode_dicts(self, dict1, dict2):
        """
        Multiply two dictionaries that are organized with commutators.
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

    def hspace_organize(self, term_string):
        """
        Given a string representation of a SINGLE term consisting of multiple modes,
        organize in the alphabetical order of operators.
        ("Hilbert space organize")

        input: 
          term_string: key of expr_dict. (No coefficients, no (+).)
        
        example:
          hspace_organize("a_b+_a+_b"): "a_a+_b+_b"
          (But this result is not in normal order, so normal_ordering() takes care of it.)
        """

        organized_term = ""

        # go over each mode in term_string
        # (find_modes gives a list of mode letters in alphabetical order)
        for mode in self.find_modes({term_string: 1}):
            operators = term_string.split("_")
            for operator in operators:  # operator could be either creation or annihilation
                if mode in operator:
                    # if this operator is in the mode that we're looking for,
                    # copy it to the organized term string
                    organized_term += operator + "_"

        return organized_term[:-1]  # remove the trailing "_"

    def normal_ordering(self, mode, single_mode_term, coeff):
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
                self.normal_ordering(mode, term_with_n, coeff),
                self.normal_ordering(mode, term_with_1, coeff),
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

    def split_modes(self, multi_mode_string):
        """
        Split a string with multiple modes into a list of terms w.r.t. each mode.

        inputs:
        multi_mode_string: a string containing multiple modes (e.g., "a_a+_b_b+").
             Should already be in alphabetical order (by running hspace_organize() beforehand)
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


def add(expr1, expr2):
    """
    Return a new Expression instance that's the sum of expr1 and expr2
    """
    result = Expression("")

    for term, coeff in expr1.expr_dict.items():
        result.expr_dict[term] = coeff

    for term, coeff in expr2.expr_dict.items():
        if term in result.expr_dict.keys():
            result.expr_dict[term] += coeff
        else:
            result.expr_dict[term] = coeff

    result.modes = result.find_modes(result.expr_dict)

    return result

def power(expr, exponent):
    """
    Power.

    inputs:
        expr: an Expression instance
        exponent: (int)
    
    returns a new Expression instance
    """
    result = copy.deepcopy(expr)
    for _ in range(exponent - 1):
        result = result.multiply(expr)

    # no need to find_modes() because no new mode is introduced

    return result


def scalar_multiply(expr, scalar):
    """
    Returns a fresh Expression instance after scalar multiplication.
    """
    result = Expression("")
    for term, coeff in expr.expr_dict.items():
        result.expr_dict[term] = coeff * scalar
        # print("term: ", term, "coeff: ", coeff, result.expr_dict[term])

    # no need to find_modes() because no new mode is introduced

    return result


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