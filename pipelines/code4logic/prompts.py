"""
prompts.py
==========
In-context learning prompt builder for CODE4LOGIC.
- Base ICL approach derived from the original CODE4LOGIC repository (Liu, 2025).
- Custom Environment Adaptations: Added `get_global_signature_block` to prevent symbol drift, 
  paradox-aware prompt generation for LogicNLI, and chain translation for NSA-LR.
"""

import inspect
import re
from typing import List
import pandas as pd
from basis_functions import (
    Constant, Variable, Function, Predicate, Equal, NonEqual, 
    Negation, Conjunction, Disjunction, Implication, Equivalence, 
    Nonequivalence, ExistentialQuantification, UniversalQuantification, End
)
from fol_grammar import FolGrammarParser

# --- ORIGINAL IMPLEMENTATION (Liu, 2025) ---
# Base code generation and prompt rules

def get_basis_functions_source() -> str:
    funcs = [
        Constant, Variable, Function, Predicate, Equal, NonEqual, 
        Negation, Conjunction, Disjunction, Implication, Equivalence, 
        Nonequivalence, ExistentialQuantification, UniversalQuantification, End
    ]
    source_code = ""
    for func in funcs:
        source_code += inspect.getsource(func) + "\n"
    return source_code


# ── Shared Strict Rules ───────────────────────────────────────────────────────

_PROMPT_RULES = """\
STRICT RULES — follow these exactly:

1. ARGUMENT ORDER: In binary predicates, the SUBJECT (agent/actor) ALWAYS comes first.
   CORRECT: PublishedBy(harryPotter, newVesselPress)   WRONG: PublishedBy(newVesselPress, harryPotter)
   CORRECT: PlayFor(ailtonSilva, fluminense)           WRONG: PlayFor(fluminense, ailtonSilva)
   CORRECT: LocatedIn(beijing, southernChina)          WRONG: LocatedIn(southernChina, beijing)

2. EXISTENTIAL QUANTIFICATION: "some X", "a X", "there is a X", "at least one X", or any
   sentence introducing an implicit object MUST use ExistentialQuantification with a
   type-guard AND the property — never collapse to a flat predicate:
   CORRECT: ExistentialQuantification(
                Conjunction(Predicate("FootballClub",["x"]), Predicate("LoanedTo",["ailton","x"])), "x")
   WRONG:   Predicate("LoanedTo",["ailton","footballClub"])

3. UNIVERSAL QUANTIFICATION: "No X does Y" and "All X do Y" MUST use UniversalQuantification:
   "No pets are cats"  →  UniversalQuantification(
                             Implication(Predicate("Pet",["x"]), Negation(Predicate("Cat",["x"]))), "x")

4. CONJUNCTION DECOMPOSITION: "Both X and Y" for a single entity → TWO predicates joined
   with Conjunction. Never fuse into one long predicate name:
   CORRECT: Conjunction(Predicate("Hosted",["beijing","summerOlympics"]),
                        Predicate("Hosted",["beijing","winterOlympics"]))
   WRONG:   Predicate("HostedBothSummerAndWinterOlympics", ["beijing"])

5. PREDICATE DECOMPOSITION & NAMING: CamelCase, start uppercase. Constants are lowercase camelCase.
   - Keep predicate names SHORT and GENERAL primitive concepts (e.g., Book, English, PlayFor).
   - NEVER fuse adjectives and nouns into one predicate.
     CORRECT: Conjunction(Predicate("Book", ["x"]), Predicate("English", ["x"]))
     WRONG:   Predicate("Englishbook", ["x"])
   - NEVER fuse subjects or objects into the verb predicate.
     CORRECT: Predicate("HasLunch", ["james", "company"])
     WRONG:   Predicate("Haslunchincompany", ["james"])

6. SYNTAX: Ensure every parenthesis and bracket is closed correctly. Output ONLY
   valid Python code using the functions below. DO NOT output conversational text, 
   markdown explanations, or imports.
"""


# ── FOLIO: Global Signature Prompt ───────────────────────────────────────────

# --- CUSTOM ADAPTATION (Giulio Zhu) ---
def get_global_signature_block(k_shots_df: pd.DataFrame) -> str:
    """
    Build a concise predicate-signature block from the k-shot examples.
    This anchors the model's vocabulary and prevents symbol drift across
    multi-sentence FOLIO stories.
    """
    signatures = set()
    for _, row in k_shots_df.iterrows():
        for fol_field in ["premises-FOL", "conclusion-FOL"]:
            fol_val = row.get(fol_field, "")
            if isinstance(fol_val, list):
                for f in fol_val:
                    # Extract Predicate("Name", ...) patterns
                    for m in re.finditer(r'Predicate\("([A-Z][A-Za-z]+)"', str(f)):
                        signatures.add(m.group(1))
            elif fol_val:
                for m in re.finditer(r'Predicate\("([A-Z][A-Za-z]+)"', str(fol_val)):
                    signatures.add(m.group(1))

    if not signatures:
        return ""

    block = "\n# GLOBAL PREDICATE VOCABULARY — use ONLY these predicate names for consistency:\n"
    block += "# " + ", ".join(sorted(signatures)) + "\n\n"
    return block


def create_folio_prompt(query_nl: str, k_shots_df: pd.DataFrame,
                         num_examples: int = 10) -> str:
    """
    FOLIO-specific CODE4LOGIC prompt.
    Adds a Global Signature block before ICL examples to prevent symbol drift
    across FOLIO's complex multi-sentence discourse stories.
    """
    converter = FolGrammarParser()

    prompt  = "Please utilize the functions provided below to systematically generate the "
    prompt += "first-order logic formula that corresponds to the natural language statement.\n\n"
    prompt += _PROMPT_RULES
    prompt += get_global_signature_block(k_shots_df)
    prompt += get_basis_functions_source() + "\n"

    pairs = []
    for _, row in k_shots_df.iterrows():
        p_text = row.get("premises", [])
        p_fol  = row.get("premises-FOL", [])
        if isinstance(p_text, list) and isinstance(p_fol, list):
            for t, f in zip(p_text, p_fol):
                if t and f:
                    pairs.append((str(t), str(f)))

        c_text = row.get("conclusion")
        c_fol  = row.get("conclusion-FOL")
        if c_text and c_fol:
            pairs.append((str(c_text), str(c_fol)))

    for i, (nl, fol) in enumerate(pairs[:num_examples], 1):
        code_sequence = converter.parse_and_convert(fol)
        if "Error" in code_sequence:
            continue
        prompt += f"natural_language_statement = {repr(str(nl))}\n"
        prompt += code_sequence + "\n"

    prompt += f"natural_language_statement = {repr(str(query_nl))}\n"
    return prompt


# Backwards-compatibility alias
def create_improved_prompt(query_nl, k_shots_df, num_examples=10):
    return create_folio_prompt(query_nl, k_shots_df, num_examples)


# ── LogicNLI: Paradox-Aware Prompt ───────────────────────────────────────────

# --- CUSTOM ADAPTATION (Giulio Zhu) ---
_LOGICNLI_RULES_EXTRA = """\
7. PARADOX DETECTION: If the premises lead to BOTH hypothesis H and its negation ¬H,
   the correct label is "Paradox". Include Paradox examples in your reasoning.
   When generating FOL, ensure negation of hypothesis is represented as Negation(formula).
"""

# Curated 4-label ICL examples for LogicNLI (one per label)
_LOGICNLI_ICL_EXAMPLES = [
    # True (entailment)
    (
        "All mammals breathe air. Dogs are mammals. Therefore, dogs breathe air.",
        "Dogs breathe air.",
        "UniversalQuantification(Implication(Predicate(\"Mammal\",[\"x\"]), Predicate(\"BreathesAir\",[\"x\"])), \"x\")\n"
        "Predicate(\"Mammal\",[\"dog\"])\n"
        "formula = End(Predicate(\"BreathesAir\",[\"dog\"]))"
    ),
    # False (contradiction)
    (
        "No birds are reptiles. Sparrows are birds.",
        "Sparrows are reptiles.",
        "UniversalQuantification(Implication(Predicate(\"Bird\",[\"x\"]), Negation(Predicate(\"Reptile\",[\"x\"]))), \"x\")\n"
        "Predicate(\"Bird\",[\"sparrow\"])\n"
        "formula = End(Negation(Predicate(\"Reptile\",[\"sparrow\"])))"
    ),
    # Unknown (neutral)
    (
        "Some students study mathematics. All physics students study mathematics.",
        "Mary is a physics student.",
        "formula = End(Predicate(\"Unknown\",[\"mary\"]))"
    ),
    # Paradox
    (
        "Every even prime is greater than two. Every even prime is not greater than two.",
        "Two is an even prime greater than two.",
        "formula = End(Conjunction(Predicate(\"GreaterThanTwo\",[\"two\"]), Negation(Predicate(\"GreaterThanTwo\",[\"two\"]))))"
    ),
]


def create_logicnli_prompt(hypothesis: str, premise: str,
                            k_shots_df: pd.DataFrame,
                            num_examples: int = 10) -> str:
    """
    LogicNLI-specific CODE4LOGIC prompt.
    Injects a Paradox-aware rule and ensures at least one of each 4 label types
    appears in the ICL examples.
    """
    prompt  = "Please utilize the functions provided below to generate the "
    prompt += "first-order logic formula for the given hypothesis, given the premise context.\n\n"
    prompt += "PREMISE CONTEXT:\n" + premise + "\n\n"
    prompt += _PROMPT_RULES
    prompt += _LOGICNLI_RULES_EXTRA
    prompt += get_basis_functions_source() + "\n"

    # Inject the 4 curated Paradox-aware examples first
    for ctx, hyp, code in _LOGICNLI_ICL_EXAMPLES:
        prompt += f"# Context: {ctx}\n"
        prompt += f"natural_language_statement = {repr(hyp)}\n"
        prompt += code + "\n"

    # Then fill remaining slots from training data
    converter = FolGrammarParser()
    added = len(_LOGICNLI_ICL_EXAMPLES)
    for _, row in k_shots_df.iterrows():
        if added >= num_examples:
            break
        hyp_text = str(row.get("hypothesis", "") or "").strip()
        hyp_fol  = str(row.get("hypothesis-FOL", "") or "").strip()
        if not hyp_text or not hyp_fol:
            continue
        code_sequence = converter.parse_and_convert(hyp_fol)
        if "Error" in code_sequence:
            continue
        prompt += f"natural_language_statement = {repr(hyp_text)}\n"
        prompt += code_sequence + "\n"
        added += 1

    prompt += f"natural_language_statement = {repr(hypothesis)}\n"
    return prompt


# ── NSA-LR: Full Chain Translation Prompt ────────────────────────────────────

# --- CUSTOM ADAPTATION (Giulio Zhu) ---
_NSA_LR_CHAIN_INSTRUCTION = """\
CHAIN TRANSLATION MODE:
Unlike standard NLI tasks, this reasoning chain contains MULTIPLE intermediate
steps. Translate EACH step as a separate named formula variable, then combine:
  step1 = End(...)
  step2 = End(...)
  formula = End(Conjunction(step1, step2))  # or Implication, etc.
Preserve the logical structure of ALL intermediate inferences, not just the conclusion.
"""


def create_nsa_lr_prompt(context: str, k_shots_df: pd.DataFrame,
                          num_examples: int = 6) -> str:
    """
    NSA-LR-specific CODE4LOGIC prompt.
    Uses chain translation mode: the model translates every intermediate
    reasoning step, not just the final conclusion.
    """
    prompt  = "Please utilize the functions provided below to generate the "
    prompt += "first-order logic formula representing the ENTIRE reasoning chain.\n\n"
    prompt += "REASONING CONTEXT:\n" + context + "\n\n"
    prompt += _PROMPT_RULES
    prompt += _NSA_LR_CHAIN_INSTRUCTION
    prompt += get_basis_functions_source() + "\n"

    # Use a reduced number of ICL examples (chain examples are longer)
    converter = FolGrammarParser()
    added = 0
    for _, row in k_shots_df.iterrows():
        if added >= num_examples:
            break
        ctx   = str(row.get("context", "") or "").strip()
        c_fol = row.get("conclusion_fol", "")
        if isinstance(c_fol, list) and c_fol:
            fol_str = str(c_fol[-1])   # Use the final step as the target
        else:
            fol_str = str(c_fol or "").strip()
        if not ctx or not fol_str:
            continue
        code_sequence = converter.parse_and_convert(fol_str)
        if "Error" in code_sequence:
            continue
        prompt += f"# Context: {ctx[:200]}...\n"
        prompt += code_sequence + "\n"
        added += 1

    prompt += f"# Context: {context[:200]}...\n"
    prompt += "# Translate the FULL reasoning chain:\n"
    return prompt


# ── Shared Utility ───────────────────────────────────────────────────────────

def clean_code_sequence(raw_code: str) -> str:
    """Truncates at first End() call"""
    match = re.search(r"(formula\s*=\s*End\(.+?\))", raw_code)
    if match:
        return raw_code[:match.end()]
    return raw_code
