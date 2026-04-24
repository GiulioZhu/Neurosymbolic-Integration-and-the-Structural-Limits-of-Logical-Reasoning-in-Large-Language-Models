import math
import re

_UNICODE_NORM = {
    '⊕': ' XOR ',   
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

WEIGHTS = {
    'AND': 1.0,
    'OR': 1.0,
    'FORALL': 2.0,
    'EXISTS': 2.0,
    'NOT': 2.0,
    'IMP': 3.0,
    'IFF': 3.0,
    'XOR': 3.5
}

def compute_locm(fol_str: str, num_premises: int = 1) -> float:
    """
    Computes Logical Complexity Metric (LoCM) for a given FOL formula.
    As hops/reasoning depth is required, num_premises is used as a proxy.
    """
    text = fol_str
    # Normalise Unicode operators to ASCII keywords
    for ch, repl in _UNICODE_NORM.items():
        text = text.replace(ch, repl)
    text = text.replace('^', ' AND ').replace('~', ' NOT ')
    
    _TOKEN_RE = re.compile(
        r'\s*(XOR|IFF|IMP|AND|OR|NOT|FORALL|EXISTS|NEQ'
        r'|[A-Za-z_][A-Za-z0-9_]*'
        r'|[(),=]'
        r'|\S)\s*',
        re.UNICODE,
    )
    
    tokens = []
    for m in _TOKEN_RE.finditer(text):
        tok = m.group(1)
        if tok:
            tokens.append(tok)
            
    # Count frequencies of operators
    freq = {op: 0 for op in WEIGHTS.keys()}
    for tok in tokens:
        if tok in freq:
            freq[tok] += 1
            
    # Calculate score
    S_phi = 0.0
    for op, count in freq.items():
        S_phi += WEIGHTS[op] * count
        
    # Hop term uses weight 2.0 and number of premises as h
    S_phi += 2.0 * num_premises
    
    return math.sqrt(S_phi) if S_phi > 0 else 0.0

if __name__ == "__main__":
    test_cases = [
        ("∀x (ConvictedCriminal(x) ∧ FoundGuilty(x) → SentencedToPunishment(x))", 1),
        ("∃x (Animal(x) ∧ ¬Big(x))", 2),
        ("P(x) ∨ Q(x)", 0),
        ("P(x) ↔ Q(x) ⊕ R(x)", 3)
    ]
    print("Testing LoCM Metric Calculation:\n")
    for fol, h in test_cases:
        score = compute_locm(fol, h)
        print(f"FOL: {fol}\nHops: {h}\nLoCM Score: {score:.4f}\n{'-'*40}")
