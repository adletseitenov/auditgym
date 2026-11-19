"""CASE §04: «Калькулятор допускает только арифметику и десятичные числа, не код и файлы»."""
import unittest

from auditgym.calc import MAX_LENGTH, CalcError, calculate


class CalculatorArithmeticTest(unittest.TestCase):
    def check(self, expression, expected):
        self.assertEqual(calculate(expression), expected, expression)

    def test_contract_example(self):
        self.check("125000 * 465.37", "58171250.00")

    def test_addition(self):
        self.check("1 + 2", "3")

    def test_subtraction_to_negative(self):
        self.check("2 - 5", "-3")

    def test_multiplication_keeps_decimal_places(self):
        self.check("1.50 * 2", "3.00")

    def test_division_exact(self):
        self.check("10 / 4", "2.5")

    def test_division_rounds_to_34_digits_half_even(self):
        self.check("1 / 3", "0." + "3" * 34)
        self.check("2 / 3", "0." + "6" * 33 + "7")

    def test_precedence(self):
        self.check("2 + 3 * 4", "14")

    def test_parentheses(self):
        self.check("(2 + 3) * 4", "20")

    def test_nested_parentheses(self):
        self.check("((((1.5))))", "1.5")

    def test_unary_minus(self):
        self.check("-5 + 2", "-3")

    def test_double_unary_minus(self):
        self.check("--5", "5")

    def test_unary_minus_of_group(self):
        self.check("-(2 - 3)", "1")

    def test_left_associative(self):
        self.check("10 - 3 - 2", "5")
        self.check("100 / 10 / 5", "2")

    def test_no_negative_zero(self):
        self.check("0 * -1", "0")
        self.check("-0.00", "0.00")

    def test_whitespace_and_tabs(self):
        self.check(" \t1+\t2 ", "3")

    def test_fx_conversion_example(self):
        self.check("125000 * 465.3700", "58171250.0000")

    def test_large_result_has_no_exponent(self):
        result = calculate("99999999999999999 * 99999999999999999")
        self.assertNotIn("E", result.upper())
        self.assertEqual(result, "9999999999999999800000000000000001")

    def test_rounding_of_long_result_is_fixed_point(self):
        self.assertEqual(calculate("1000000000000000000000000000000000 * 1000"),
                         "1000000000000000000000000000000000000")

    def test_deterministic(self):
        self.assertEqual({calculate("7 / 13 * 1.1") for _ in range(20)}, {calculate("7 / 13 * 1.1")})

    def test_result_is_str(self):
        self.assertIsInstance(calculate("1"), str)


class CalculatorRejectsTest(unittest.TestCase):
    def rejects(self, expression):
        with self.assertRaises(CalcError, msg=repr(expression)):
            calculate(expression)

    def test_import_call(self):
        self.rejects("__import__('os')")

    def test_os_system(self):
        self.rejects("__import__('os').system('ls')")

    def test_open_file(self):
        self.rejects("open('/etc/passwd').read()")

    def test_power_operator(self):
        self.rejects("2 ** 3")
        self.rejects("9**9**9")

    def test_exponent_notation(self):
        self.rejects("1e999")
        self.rejects("1E5")
        self.rejects("1.5e-3")

    def test_names(self):
        self.rejects("x")
        self.rejects("x + 1")
        self.rejects("True")
        self.rejects("nan")
        self.rejects("inf")
        self.rejects("Infinity")

    def test_lambda(self):
        self.rejects("(lambda: 1)()")

    def test_attribute_access(self):
        self.rejects("(1).__class__")
        self.rejects("1 .real")

    def test_subscript_and_comprehension(self):
        self.rejects("[1][0]")
        self.rejects("[x for x in ()]")

    def test_string_literals(self):
        self.rejects("'1'")
        self.rejects('"1" + "2"')

    def test_comparison_and_bitwise(self):
        for expression in ("1 < 2", "1 == 1", "1 & 1", "1 | 2", "1 ^ 2", "~1", "1 << 2", "1 % 2", "7 // 2"):
            self.rejects(expression)

    def test_unary_plus(self):
        self.rejects("+1")

    def test_division_by_zero(self):
        self.rejects("1 / 0")
        self.rejects("1 / (2 - 2)")
        self.rejects("0 / 0")

    def test_empty_and_blank(self):
        self.rejects("")
        self.rejects("   ")

    def test_too_long(self):
        self.rejects("1+" * 100 + "1")
        self.assertEqual(len("1+" * 99 + "1"), 199)
        self.assertEqual(calculate("1+" * 99 + "1"), "100")

    def test_nesting_of_exactly_100_levels_is_accepted(self):
        """Docstring: nesting deeper than 100 levels is refused; 100 itself is fine (attacker E8 #6)."""
        self.assertEqual(calculate("-" * 100 + "1"), "1")
        self.rejects("-" * 101 + "1")
        self.assertEqual(calculate("-(" * 50 + "1" + ")" * 50), "1")
        self.rejects("-(" * 50 + "-1" + ")" * 50)

    def test_limit_is_200(self):
        self.assertEqual(MAX_LENGTH, 200)
        self.assertEqual(calculate("1" * 34 + " " * 166), "1" * 34)
        self.rejects("1" * 34 + " " * 167)

    def test_unicode_digits(self):
        for expression in ("٣", "١+٢", "１", "𝟙", "²", "½", "1٫5"):
            self.rejects(expression)

    def test_unicode_minus_and_operators(self):
        self.rejects("5 − 2")
        self.rejects("2 × 3")

    def test_malformed_numbers(self):
        for expression in ("1.", ".5", "1..2", "1.2.3", "1_000", "0x10", "0b1", "1,5", "01j"):
            self.rejects(expression)

    def test_unbalanced(self):
        for expression in ("(1", "1)", "()", "(", ")", "1 +", "* 2", "1 2", "(1)(2)"):
            self.rejects(expression)

    def test_newline_and_semicolon(self):
        self.rejects("1\n+2")
        self.rejects("1; 2")
        self.rejects("1 # comment")

    def test_non_string_input(self):
        for value in (None, 1, 1.5, b"1+1", ["1"], {"a": 1}):
            self.rejects(value)

    def test_too_deep_nesting(self):
        self.rejects("(" * 101 + "1" + ")" * 101)
        self.rejects("-" * 150 + "1")

    def test_literal_beyond_precision(self):
        self.rejects("9" * 35)
        self.assertEqual(calculate("9" * 34), "9" * 34)

    def test_overflow_and_underflow(self):
        big = "1" + "0" * 33
        self.rejects(" * ".join([big] * 5) + " * " + " * ".join([big] * 25))
        tiny = "0." + "0" * 30 + "1"
        self.rejects(" * ".join([tiny] * 6))

    def test_error_message_is_fixed(self):
        for expression in ("__import__('os')", "1/0", "x"):
            try:
                calculate(expression)
            except CalcError as exc:
                self.assertEqual(str(exc), "Bad expression.")


if __name__ == "__main__":
    unittest.main()
