"""
Tests for Expression.from_dict(), Expression.copy(), and the copy protocol
hooks (__copy__ / __deepcopy__).

Run directly (python test_copy.py) or with pytest.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import copy

import numpy as np
import sympy as sp

from ladders import Expression, power, scalar_multiply


def test_copy_is_independent():
    """Mutating a copy must not touch the original -- both containers."""
    original = Expression("2a+_a(+)b+_b")
    duplicate = original.copy()

    duplicate.expr_dict["a+_a"] = 99
    duplicate.modes.append("z")

    assert original.expr_dict["a+_a"] == 2
    assert original.modes == ["a", "b"]


def test_copy_preserves_contents():
    """A copy holds the same terms, coefficients and modes."""
    original = Expression("2a+_a(+)3.5b+_b(+)1")
    duplicate = original.copy()

    assert duplicate.expr_dict == original.expr_dict
    assert duplicate.modes == original.modes
    assert duplicate.expr_dict is not original.expr_dict
    assert duplicate.modes is not original.modes


def test_copy_carries_logging_flag():
    """LOGGING is user configuration, so it survives the copy."""
    original = Expression("a")
    original.LOGGING = True
    assert original.copy().LOGGING is True


def test_copy_preserves_subclass():
    """type(self) means a subclass copy stays a subclass."""

    class SymbolicExpression(Expression):
        pass

    duplicate = SymbolicExpression("a+_a").copy()
    assert type(duplicate) is SymbolicExpression


def test_scalar_multiply_preserves_subclass():
    """The module-level helpers build through type(expr), too."""

    class SymbolicExpression(Expression):
        pass

    out = scalar_multiply(SymbolicExpression("a+_a"), 2)
    assert type(out) is SymbolicExpression
    assert out.expr_dict["a+_a"] == 2


def test_from_dict_copies_the_input():
    """The caller keeps ownership of the dictionary they passed in."""
    source = {"a+_a": 1, "b": 2}
    expr = Expression.from_dict(source)

    source["a+_a"] = 99

    assert expr.expr_dict["a+_a"] == 1
    assert expr.modes == ["a", "b"]  # modes recomputed when not supplied


def test_from_dict_trusts_supplied_modes():
    """Passing 'modes' skips the scan (and is taken at face value)."""
    expr = Expression.from_dict({"a+_a": 1}, modes=["a"])
    assert expr.modes == ["a"]


def test_copy_module_hooks():
    """copy.copy() and copy.deepcopy() both route through copy()."""
    original = Expression("2a+_a")

    for duplicate in (copy.copy(original), copy.deepcopy(original)):
        assert type(duplicate) is Expression
        assert duplicate.expr_dict == original.expr_dict
        duplicate.expr_dict["a+_a"] = 99
        assert original.expr_dict["a+_a"] == 2


def test_deepcopy_shares_repeated_references():
    """__deepcopy__ honours the memo dict, so one object copies once."""
    expr = Expression("a")
    pair = copy.deepcopy([expr, expr])
    assert pair[0] is pair[1]
    assert pair[0] is not expr


def test_copy_with_symbolic_coefficients():
    """Sympy coefficients are shared, not rebuilt -- and stay usable."""
    r = sp.Symbol("r", real=True)
    original = Expression.from_dict({"a+_a": sp.cosh(r), "a": 1})
    duplicate = original.copy()

    assert duplicate.expr_dict["a+_a"] is original.expr_dict["a+_a"]
    assert sp.simplify(duplicate.expr_dict["a+_a"].subs(r, 0) - 1) == 0


def test_power_still_correct():
    """power() now copies via copy() instead of copy.deepcopy()."""
    expr = Expression("a+_a")
    squared = power(expr, 2)

    # (a+ a)^2 = a+ a+ a a + a+ a  in normal order
    assert np.isclose(complex(squared.expr_dict["a+_a+_a_a"]), 1)
    assert np.isclose(complex(squared.expr_dict["a+_a"]), 1)
    # and nothing else: the expansion produces exactly these two terms
    assert set(squared.expr_dict) == {"a+_a+_a_a", "a+_a"}
    # the input must be untouched
    assert expr.expr_dict == {"a+_a": 1}


if __name__ == "__main__":
    for name, fn in sorted(list(globals().items())):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"{name} passed")
    print("\nAll tests passed.")
