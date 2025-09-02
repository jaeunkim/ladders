# Multi-mode, non-commutative Bosonic ladder operator calculator
Adds, multiplies, and expands quantum mechanical expressions 
    written in terms of Bosonic ladder operators.
    
Maintains normal ordering (Wick ordering) by 
    applying the Bosonic commutation relation [a, a+] = 1.

### Data structure: 
For example, by initializing an instance by Expression("2a+_a(+)b+_b(+)1"),
- Expression.expr_dict: 
    Dictionary that has strings which represent each term of an expression as keys, 
    and the (complex) coefficients as values. 
    - Example: {"a+_a":2, "b+_b":1, "":1}
    - Note: key "" is used for constants. 
        {"":1} means constant 1, where an empty dictionary, {}, means 0.
- Expression.modes: a list of characters that represents the modes involved in the Expression.
    - Example: ['a', 'b']

### Syntax rules for initializing an Expression instance:
An operator should be a single letter that's not j. (j is reserved for the imaginary number)
- Example: "a_a+(+)3+4.ja(+)b"
- Creation operators are in the form of "a+" (you can use any letter)
- Annihilation operators are in the form of "a" (you can use any letter)
- Multiplication of operators are expressed with "_"
- Additions are written as "(+)" (because "+" is for dagger)
- Coefficients are multiplied in front of the operators

### Algorithm: 
Exactly as how you would it with pen and paper. 
- Step 1. Expand products of Expressions by multiplying each term
- Step 2. Convert each term to normal order using the commutation relation
- Step 3. Collect like terms (same term string) and sum their coefficients
    
### TODO:
- Caching (memoization):
    - Since the return values (dictionaries) are mutable and unhashable, 
        memoization breaks the code. 
    - Implement efficient cache passing when creating a new Expression instance 
        by adding or multiplying Expression instances
- Pretty printing
- Implement getter and setter for 'expr_dict' and 'modes' to protect internal states
- Handle mode-dependent coefficient factors: ex. 1/sqrt(omega_1) and 1/sqrt(omega_z)
