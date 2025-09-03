# Multi-mode, non-commutative Bosonic ladder operator calculator
Adds, multiplies, and expands quantum mechanical expressions 
    written in terms of Bosonic ladder operators.
    
Maintains normal ordering (Wick ordering) by 
    applying the Bosonic commutation relation `[a, a+] = 1`.

## Data structure: 
Example: initialize an instance by `Expression("2a+_a(+)b+_b(+)1")`,
- `Expression.expr_dict` (type `dict`):
    - key: strings that represnt the operators of each term
    - value: (complex) coefficients of each term
    - Example: `{"a+_a":2, "b+_b":1, "":1}`
    - Note: an empty string key `""` is used for constants.
      
        `{"":1}` means constant 1, where an empty dictionary `{}` means 0.
- `Expression.modes`: a list of characters that represents the modes involved in the Expression.
    - Example: `['a', 'b']`

## Syntax rules for initializing an Expression instance:
An operator should be a single letter that's not `j`. (`j` is reserved for the imaginary number)
- Example: `"a_a+(+)3+4.ja(+)b"`
- Creation operators are in the form of `"a+"` (you can use any letter)
- Annihilation operators are in the form of `"a"` (you can use any letter)
- Multiplication of operators are expressed with `"_"`
- Additions are written as `"(+)"` (because `"+"` is for dagger)
- Coefficients are multiplied in front of the operators

## Algorithm: `Expression.multiply()`

Exactly as you would with pen and paper.

- **Step 1**. Expand products of Expressions by multiplying each term
  
  Example: Multiply $a b^\dagger$ and $a^\dagger b$, which are expressed by `{"a_b+":1}` and `{"a+_b":1}`
      
  → you get $a b^\dagger a^\dagger b$, which is expressed by `{"a_b+_a+_b":1}`
- **Step 2**. ```group_by_mode()```: Rearrange operators so that those of the same mode are grouped together
  
    From the previous result → `{"a_a+_b+_b":1}`
      
- **Step 3**. ```normal_order()```: Convert each term to normal order using the commutation relation
  
  Example: "a_a+"  → `{"a+_a":1, "":1}`
      
  > **Reminder:** the empty string key `""` denotes a constant
      
- **Step 4**. Collect like terms (same term string) and sum their coefficients
    
## TODO:
- Caching (memoization):
    - Since the return values (dictionaries) are mutable and unhashable, 
        memoization breaks the code. 
    - Implement efficient cache passing when creating a new Expression instance 
        by adding or multiplying Expression instances
- Pretty printing
- Implement getter and setter for 'expr_dict' and 'modes' to protect internal states
- Handle mode-dependent coefficient factors: ex. 1/sqrt(omega_1) and 1/sqrt(omega_z)
