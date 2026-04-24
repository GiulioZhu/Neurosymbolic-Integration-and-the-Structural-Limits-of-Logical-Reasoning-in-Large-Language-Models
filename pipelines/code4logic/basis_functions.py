"""
basis_functions.py
==================
Defines the primitive basis functions used by the CODE4LOGIC LLM for generating FOL code.
- Derived from the original CODE4LOGIC repository (Liu, 2025).
"""

from typing import List

# ------------------------------------------------------------------
# BASIS FUNCTIONS (The "API" the LLM learns to use)
# ------------------------------------------------------------------

def Constant(constant_name: str):
    """return a constant symbol"""
    return constant_name.lower()

def Variable(variable_name: str):
    """return a variable symbol,
    which starts with $"""
    return variable_name

def Function(function_name: str, terms: List[str]):
    """return a function symbol, for example,
    father (x) means the father of x"""
    return '{}({})'.format(
        function_name.lower(), ','.join(terms))

def Predicate(predicate_name: str, terms: List[str]):
    """return an atomic formula with a predicate,
    whose name starts with uppercase"""
    return '{}({})'.format(
        predicate_name.capitalize(), ', '.join(terms))

def Equal(term_a: str, term_b: str):
    """return an atomic formula with equal operation"""
    return '{} = {}'.format(term_a, term_b)

def NonEqual(term_a: str, term_b: str):
    """return an atomic formula with non-equal operation"""
    # Using Unicode for '≠' as in the paper 
    return '{} ≠ {}'.format(term_a, term_b)

def Negation(formula: str):
    """return the negation of the input formula"""
    # Using Unicode for '¬' as in the paper
    return '¬({})'.format(formula)

def Conjunction(formula_a: str, formula_b: str):
    """return the conjunction of the input formulas"""
    # Using Unicode for '∧' as in the paper
    return '({}) ∧ ({})'.format(formula_a, formula_b)

def Disjunction(formula_a: str, formula_b: str):
    """return the disjunction of the input formulas"""
    # Using Unicode for '∨' as in the paper
    return '({}) ∨ ({})'.format(formula_a, formula_b)

def Implication(antecedent_formula: str, consequent_formula: str):
    """return the implication formula of the
    antecedent formula and consequent formula"""
    # Using Unicode for '→' as in the paper
    return '({}) → ({})'.format(
        antecedent_formula, consequent_formula)

def Equivalence(formula_a: str, formula_b: str):
    """return the logical equivalence formula of
    the input formulas"""
    # Using Unicode for '↔' as in the paper
    return '({}) ↔ ({})'.format(formula_a, formula_b)

def Nonequivalence(formula_a: str, formula_b: str):
    """return the logical non-equivalence formula of
    the input formulas"""
    # Using Unicode for '⨁' as in the paper
    return '({}) ⨁ ({})'.format(formula_a, formula_b)

def ExistentialQuantification(formula: str, variable_symbol: str):
    """return an existential quantification of the input formula
    and the input variable symbol"""
    # Using Unicode for '∃' as in the paper
    return '∃{}({})'.format(variable_symbol, formula)

def UniversalQuantification(formula: str, variable_symbol: str):
    """return an universal quantification of the input formula
    and the input variable symbol"""
    # Using Unicode for '∀' as in the paper
    return '∀{}({})'.format(variable_symbol, formula)

def End(formula: str):
    return formula
