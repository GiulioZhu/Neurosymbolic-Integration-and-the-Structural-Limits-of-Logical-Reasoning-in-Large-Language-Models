"""
fol_verifier.py
================
Provides check_equivalence(fol_a, fol_b) which checks whether two
First-Order Logic formula strings (using Unicode operators as produced by
basis_functions.py) are logically equivalent.

- Core concept derived from original CODE4LOGIC repo (Liu, 2025).
- Custom Environment Adaptations: Completely rewritten the Z3 parser to correctly 
  handle quantified variables and environments. Added robust fallback layers 
  using NLTK/Prover9 and exact string match.

Priority order:
  1. Z3  – a recursive-descent parser converts the Unicode FOL string into
            Z3 expressions; equivalence = unsat(¬(A ↔ B)).
  2. Prover9 via NLTK – used if Z3 fails to parse either formula.
  3. Exact string match – last-resort fallback.

Returns a dict:
  {
    "match":  bool,
    "method": "z3" | "prover9" | "string_match",
    "reason": str   # short description of what happened
  }
"""

from __future__ import annotations
import re
import unicodedata
import logging

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# SECTION 1 – Z3 PARSER
# --- CUSTOM ADAPTATION (Giulio Zhu) ---
# Entire recursive descent parser written to replace brittle exact-match logic.
# ---------------------------------------------------------------------------
# We translate the Unicode FOL string produced by basis_functions.py into
# Z3 Bool/ForAll/Exists expressions.
#
# Grammar (subset handled here, lowest-precedence first):
#   formula  ::= iff
#   iff      ::= xor  ( '↔' xor )*
#   xor      ::= impl ( '⊕' impl )*   -- treated as XOR (Xor)
#   impl     ::= disj ( '→' formula )  -- right-associative
#   disj     ::= conj ( '∨' conj )*
#   conj     ::= unary( '∧' unary )*
#   unary    ::= '¬' unary | quant | atom
#   quant    ::= ('∀'|'∃') VAR formula
#   atom     ::= predicate | '(' formula ')'
#   predicate::= NAME '(' term,* ')'  |  NAME       -- zero-arg predicate
#   term     ::= NAME '(' term,* ')' | NAME
#
# Predicate and function symbols become uninterpreted Z3 functions.
# Variables become z3.Const with a fresh BoolSort domain when quantified.
# ---------------------------------------------------------------------------

_UNICODE_NORM = {
    '⊕': ' XOR ',   # Nonequivalence / XOR
    '↔': ' IFF ',
    '→': ' IMP ',
    '∧': ' AND ',
    '∨': ' OR ',
    '¬': ' NOT ',
    '∀': ' FORALL ',
    '∃': ' EXISTS ',
    '≠': ' NEQ ',
    '⇒': ' IMP ',
}


class _FOLParseError(Exception):
    pass


class _Tokenizer:
    """Simple character-by-character tokenizer."""

    _TOKEN_RE = re.compile(
        r'\s*(XOR|IFF|IMP|AND|OR|NOT|FORALL|EXISTS|NEQ'
        r'|[A-Za-z_][A-Za-z0-9_]*'
        r'|[(),=]'
        r'|\S)\s*',
        re.UNICODE,
    )

    def __init__(self, text: str):
        self._tokens: list[str] = []
        self._pos = 0
        # Normalise Unicode operators to ASCII keywords first
        for ch, repl in _UNICODE_NORM.items():
            text = text.replace(ch, repl)
        # Also handle ∧ written as ^ (just in case)
        text = text.replace('^', ' AND ').replace('~', ' NOT ')
        _KEYWORDS = {'XOR', 'IFF', 'IMP', 'AND', 'OR', 'NOT', 'FORALL', 'EXISTS', 'NEQ'}
        for m in self._TOKEN_RE.finditer(text):
            tok = m.group(1)
            if tok:
                # Fix 1: Case-normalise predicate/constant tokens so that
                # Stock(kO) and Stock(ko) resolve to the same Z3 symbol.
                if tok not in _KEYWORDS:
                    tok = tok.lower()
                self._tokens.append(tok)

    def peek(self) -> str | None:
        return self._tokens[self._pos] if self._pos < len(self._tokens) else None

    def consume(self) -> str:
        tok = self._tokens[self._pos]
        self._pos += 1
        return tok

    def expect(self, val: str) -> str:
        tok = self.consume()
        if tok != val:
            raise _FOLParseError(f"Expected '{val}', got '{tok}'")
        return tok

    def at_end(self) -> bool:
        return self._pos >= len(self._tokens)


def _try_import_z3():
    try:
        import z3
        return z3
    except ImportError:
        return None


# Mutable environment for quantified variables
class _ParseEnv:
    def __init__(self, z3_mod):
        self.z3 = z3_mod
        # predicate_name -> z3 FuncDecl (uninterpreted)
        self.predicates: dict[str, object] = {}
        # function_name -> z3 FuncDecl
        self.functions: dict[str, object] = {}
        # variable_name -> z3 Const (Bool sort, domain object)
        self.variables: dict[str, object] = {}
        # A single uninterpreted sort for all individuals
        self.individual_sort = z3_mod.DeclareSort('Ind')
        # Bool sort
        self.bool_sort = z3_mod.BoolSort()

    def get_predicate(self, name: str, arity: int):
        key = f"{name}/{arity}"
        if key not in self.predicates:
            arg_sorts = [self.individual_sort] * arity
            if arity == 0:
                self.predicates[key] = self.z3.Const(name, self.bool_sort)
            else:
                self.predicates[key] = self.z3.Function(
                    name, *arg_sorts, self.bool_sort
                )
        return self.predicates[key]

    def get_function(self, name: str, arity: int):
        key = f"fn:{name}/{arity}"
        if key not in self.functions:
            arg_sorts = [self.individual_sort] * arity
            self.functions[key] = self.z3.Function(
                name, *arg_sorts, self.individual_sort
            )
        return self.functions[key]

    def get_or_create_const(self, name: str):
        """Return an individual constant (entity in domain)."""
        if name not in self.variables:
            self.variables[name] = self.z3.Const(name, self.individual_sort)
        return self.variables[name]

    def make_bound_variable(self, name: str):
        """Create a fresh Z3 variable for quantification."""
        var = self.z3.Const(name, self.individual_sort)
        self.variables[name] = var
        return var


# ---- Recursive descent parser ----

def _parse_formula(tok: _Tokenizer, env: _ParseEnv):
    return _parse_iff(tok, env)


def _parse_iff(tok: _Tokenizer, env: _ParseEnv):
    left = _parse_xor(tok, env)
    while tok.peek() == 'IFF':
        tok.consume()
        right = _parse_xor(tok, env)
        left = left == right   # Z3 Equivalence
    return left


def _parse_xor(tok: _Tokenizer, env: _ParseEnv):
    left = _parse_impl(tok, env)
    while tok.peek() == 'XOR':
        tok.consume()
        right = _parse_impl(tok, env)
        left = env.z3.Xor(left, right)
    return left


def _parse_impl(tok: _Tokenizer, env: _ParseEnv):
    left = _parse_disj(tok, env)
    if tok.peek() == 'IMP':
        tok.consume()
        right = _parse_formula(tok, env)   # right-associative
        return env.z3.Implies(left, right)
    return left


def _parse_disj(tok: _Tokenizer, env: _ParseEnv):
    left = _parse_conj(tok, env)
    while tok.peek() == 'OR':
        tok.consume()
        right = _parse_conj(tok, env)
        left = env.z3.Or(left, right)
    return left


def _parse_conj(tok: _Tokenizer, env: _ParseEnv):
    left = _parse_unary(tok, env)
    while tok.peek() == 'AND':
        tok.consume()
        right = _parse_unary(tok, env)
        left = env.z3.And(left, right)
    return left


def _parse_unary(tok: _Tokenizer, env: _ParseEnv):
    if tok.peek() == 'NOT':
        tok.consume()
        sub = _parse_unary(tok, env)
        return env.z3.Not(sub)
    if tok.peek() in ('FORALL', 'EXISTS'):
        return _parse_quantifier(tok, env)
    return _parse_atom(tok, env)


def _parse_quantifier(tok: _Tokenizer, env: _ParseEnv):
    q = tok.consume()  # FORALL or EXISTS
    # Variable name
    if tok.peek() is None:
        raise _FOLParseError("Expected variable after quantifier")
    var_name = tok.consume()
    if not re.match(r'^[A-Za-z_][A-Za-z0-9_]*$', var_name):
        raise _FOLParseError(f"Expected variable name, got '{var_name}'")

    old = env.variables.get(var_name)
    bound_var = env.make_bound_variable(var_name)

    sub = _parse_formula(tok, env)

    # Restore previous binding
    if old is None:
        env.variables.pop(var_name, None)
    else:
        env.variables[var_name] = old

    if q == 'FORALL':
        return env.z3.ForAll([bound_var], sub)
    else:
        return env.z3.Exists([bound_var], sub)


def _parse_atom(tok: _Tokenizer, env: _ParseEnv):
    if tok.peek() == '(':
        tok.consume()
        inner = _parse_formula(tok, env)
        tok.expect(')')
        return inner

    # Must be a name (predicate or propositional atom)
    name = tok.peek()
    if name is None or not re.match(r'^[A-Za-z_][A-Za-z0-9_]*$', name):
        raise _FOLParseError(f"Expected predicate or '(', got '{name}'")
    tok.consume()

    if tok.peek() == '(':
        # Predicate with arguments
        tok.consume()
        args = []
        if tok.peek() != ')':
            args.append(_parse_term(tok, env))
            while tok.peek() == ',':
                tok.consume()
                args.append(_parse_term(tok, env))
        tok.expect(')')
        pred = env.get_predicate(name, len(args))
        if len(args) == 0:
            return pred
        return pred(*args)
    else:
        # Zero-argument predicate (propositional variable / ground atom)
        pred = env.get_predicate(name, 0)
        return pred


def _parse_term(tok: _Tokenizer, env: _ParseEnv):
    name = tok.peek()
    if name is None or not re.match(r'^[A-Za-z_][A-Za-z0-9_]*$', name):
        raise _FOLParseError(f"Expected term, got '{name}'")
    tok.consume()

    if tok.peek() == '(':
        tok.consume()
        args = []
        if tok.peek() != ')':
            args.append(_parse_term(tok, env))
            while tok.peek() == ',':
                tok.consume()
                args.append(_parse_term(tok, env))
        tok.expect(')')
        fn = env.get_function(name, len(args))
        return fn(*args)

    # Check if it's a bound variable or a constant
    if name in env.variables:
        return env.variables[name]
    return env.get_or_create_const(name)


def _fol_to_z3(fol_str: str, z3_mod):
    """Parse a Unicode FOL string and return a Z3 Bool expression."""
    env = _ParseEnv(z3_mod)
    tok = _Tokenizer(fol_str)
    expr = _parse_formula(tok, env)
    if not tok.at_end():
        raise _FOLParseError(
            f"Unexpected tokens after formula: {tok.peek()!r}"
        )
    return expr, env


def _check_z3(fol_a: str, fol_b: str):
    """
    Return (bool equivalent, str reason) using Z3.
    Raises _FOLParseError or ImportError on failure.
    """
    z3 = _try_import_z3()
    if z3 is None:
        raise ImportError("z3-solver not installed")

    expr_a, env_a = _fol_to_z3(fol_a, z3)
    expr_b, _     = _fol_to_z3(fol_b, z3)

    # Merge the two environments so shared predicate symbols are the same
    # We simply check with two fresh calls and rely on the fact that
    # we use the same predicate names → z3 will see them as the same
    # uninterpreted symbols only if we share the env.  Re-parse sharing env:
    env = _ParseEnv(z3)
    tok_a = _Tokenizer(fol_a)
    tok_b = _Tokenizer(fol_b)
    ea = _parse_formula(tok_a, env)
    if not tok_a.at_end():
        raise _FOLParseError(f"Trailing tokens in A: {tok_a.peek()!r}")
    eb = _parse_formula(tok_b, env)
    if not tok_b.at_end():
        raise _FOLParseError(f"Trailing tokens in B: {tok_b.peek()!r}")

    # Check satisfiability of ¬(A ↔ B)
    s = z3.Solver()
    s.set("timeout", 5000)   # 5-second timeout
    s.add(z3.Not(ea == eb))
    result = s.check()

    if result == z3.unsat:
        return True, "z3:unsat(¬(A↔B)) → equivalent"
    elif result == z3.sat:
        return False, "z3:sat(¬(A↔B)) → not equivalent"
    else:
        # unknown → fall through to next method
        raise _FOLParseError("Z3 returned unknown (timeout or resource limit)")


# ---------------------------------------------------------------------------
# SECTION 2 – PROVER9 / NLTK FALLBACK
# --- CUSTOM ADAPTATION (Giulio Zhu) ---
# Added logical theorem proving fallback to handle Z3 timeouts and syntax drifts.
# ---------------------------------------------------------------------------

def _normalise_for_nltk(fol_str: str) -> str:
    """
    Convert Unicode FOL to NLTK logic string format.
    NLTK uses: -P (negation), P & Q, P | Q, P -> Q, P <-> Q,
               all x. P, exists x. P
    """
    s = fol_str
    s = s.replace('¬', '-')
    s = s.replace('∧', '&')
    s = s.replace('∨', '|')
    s = s.replace('→', '->')
    s = s.replace('⇒', '->')
    s = s.replace('↔', '<->')
    s = s.replace('⊕', '!=')   # no native XOR in NLTK; approximate
    s = s.replace('∀', 'all ')
    s = s.replace('∃', 'exists ')
    return s


def _check_prover9(fol_a: str, fol_b: str):
    """
    Return (bool equivalent, str reason) using Prover9 via NLTK.
    Attempts to prove A→B and B→A.
    Raises on any failure.
    """
    try:
        from nltk.sem.logic import Expression as LogicExpression
        from nltk.inference.prover9 import Prover9
    except ImportError as e:
        raise ImportError(f"NLTK or Prover9 not available: {e}")

    read = LogicExpression.fromstring
    a_norm = _normalise_for_nltk(fol_a)
    b_norm = _normalise_for_nltk(fol_b)

    try:
        expr_a = read(a_norm)
        expr_b = read(b_norm)
    except Exception as e:
        raise _FOLParseError(f"NLTK parse error: {e}")

    p = Prover9(timeout=10)
    try:
        # A→B
        ab = p.prove(expr_b, [expr_a])
        # B→A
        ba = p.prove(expr_a, [expr_b])
        equiv = ab and ba
        reason = (
            f"prover9: A→B={ab}, B→A={ba} → {'equivalent' if equiv else 'not equivalent'}"
        )
        return equiv, reason
    except Exception as e:
        raise RuntimeError(f"Prover9 proof failed: {e}")


# ---------------------------------------------------------------------------
# SECTION 3 – PUBLIC API
# --- ORIGINAL CONCEPT (Liu, 2025) / CUSTOM INTEGRATION ---
# ---------------------------------------------------------------------------

def _normalise_string(s: str) -> str:
    """Light normalisation for string-match fallback."""
    s = s.strip()
    # Unicode normalise (NFC)
    s = unicodedata.normalize('NFC', s)
    # Collapse multiple spaces
    s = re.sub(r'\s+', ' ', s)
    return s


def check_equivalence(fol_a: str, fol_b: str) -> dict:
    """
    Check logical equivalence of two FOL formula strings.

    Parameters
    ----------
    fol_a : str
        The generated FOL formula (Unicode operators from basis_functions.py).
    fol_b : str
        The ground-truth FOL formula.

    Returns
    -------
    dict with keys:
        match  : bool
        method : "z3" | "prover9" | "string_match"
        reason : str
    """
    # --- Tier 1: Z3 ---
    try:
        equiv, reason = _check_z3(fol_a, fol_b)
        return {"match": equiv, "method": "z3", "reason": reason}
    except ImportError as e:
        logger.debug("Z3 not available: %s", e)
    except _FOLParseError as e:
        logger.debug("Z3 parse error: %s", e)
    except Exception as e:
        logger.debug("Z3 error: %s", e)

    # --- Tier 2: Prover9 ---
    try:
        equiv, reason = _check_prover9(fol_a, fol_b)
        return {"match": equiv, "method": "prover9", "reason": reason}
    except ImportError as e:
        logger.debug("Prover9 not available: %s", e)
    except Exception as e:
        logger.debug("Prover9 error: %s", e)

    # --- Tier 3: String match ---
    na = _normalise_string(fol_a)
    nb = _normalise_string(fol_b)
    equiv = na == nb
    return {
        "match": equiv,
        "method": "string_match",
        "reason": "string_match: exact comparison after normalisation",
    }


# ---------------------------------------------------------------------------
# Quick self-test (run this file directly)
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    tests = [
        # (A,                               B,                             expected_match)
        ("∃x Animal(x)",                   "∃x (Animal(x))",              True),
        ("∀x (Square(x) → Shape(x))",      "∀x (Square(x) → Shape(x))",  True),
        ("Turtle(rockie)",                  "Turtle(rockie)",              True),
        ("¬Turtle(rockie)",                 "¬Turtle(rockie)",             True),
        ("Turtle(rockie) ∨ Cute(rockie)",   "Turtle(rockie) ∨ Cute(rockie)", True),
        # Different predicates → NOT equivalent
        ("OcellatedWildTurkey(tom)",        "WildTurkey(joey)",            False),
        ("James(LunchInCompany)",           "HasLunch(james, company)",    False),
        # Structural equivalence: ¬(A∧B) ↔ (¬A ∨ ¬B)  De Morgan's law
        ("¬(P(a) ∧ Q(a))",                 "¬P(a) ∨ ¬Q(a)",              True),
    ]
    print(f"{'A':<40} {'B':<40} {'Expected':>8} {'Got':>8} {'Method':<12}")
    print("-" * 115)
    for a, b, expected in tests:
        res = check_equivalence(a, b)
        status = "OK" if res["match"] == expected else "FAIL"
        print(
            f"{a:<40} {b:<40} {str(expected):>8} {str(res['match']):>8}"
            f"  {res['method']:<12}  {status}"
        )
