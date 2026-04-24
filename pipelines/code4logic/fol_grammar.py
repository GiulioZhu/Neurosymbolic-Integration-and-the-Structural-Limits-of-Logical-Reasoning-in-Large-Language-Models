"""
fol_grammar.py
==============
BNF grammar for parsing string-based FOL expressions back into structured sequences.
- Derived from the original CODE4LOGIC repository (Liu, 2025).
"""

from pyleri import Grammar, Sequence, Choice, Keyword, Regex, Ref, List as PyleriList, Token, Repeat

# ------------------------------------------------------------------
# GRAMMAR DEFINITION
# ------------------------------------------------------------------

class FolGrammar(Grammar):
    # 1. Define Refs for recursive structures
    term = Ref()
    formula = Ref()
    unary_formula = Ref()

    # 2. Basic Tokens & Keywords
    t_open = Token('(')
    t_close = Token(')')
    t_comma = Token(',')
    t_equal = Token('=')
    t_neq = Token('≠') # ≠
    
    k_true = Keyword('True')
    k_false = Keyword('False')

    # 3. Identifiers
    # Regex for names (e.g., predicates, functions)
    r_name = Regex('^[a-zA-Z_][a-zA-Z0-9_]*')
    
    # Allow full words starting with lowercase (e.g., 'socrates')
    # and ensure numbers are handled correctly.
    r_var_or_const = Regex('^[a-z][a-zA-Z0-9_]*')
    r_upper_const = Regex('^[A-Z][a-zA-Z0-9_]*')
    r_quoted_const = Regex("^'[a-zA-Z0-9_ ]+'")

    # 4. Term Definitions
    # Function: name(term, term)
    func_term = Sequence(r_name, t_open, PyleriList(term, delimiter=t_comma), t_close)
    
    # We define the content of the 'term' Ref here
    # Assign directly to avoid "UnusedElementError"
    term = Choice(func_term, r_var_or_const, r_upper_const, r_quoted_const)
    
    # 5. Atomic Formulas
    # Predicate: name(term, term)
    predicate = Sequence(r_name, t_open, PyleriList(term, delimiter=t_comma), t_close)
    
    # Equations
    eq_formula = Sequence(term, t_equal, term)
    neq_formula = Sequence(term, t_neq, term)

    # Parenthesized atomic formulas
    parens_formula = Sequence(t_open, formula, t_close)
    
    atomic_formula = Choice(k_true, k_false, eq_formula, neq_formula, predicate, parens_formula)

    # 6. Complex Formulas
    # Negation symbol (Handle variations)
    t_neg = Choice(Token('¬'), Token('!'), Token('~'))
    negation = Sequence(t_neg, unary_formula)

    # Binary Operators
    t_and = Choice(Token('∧'), Token('&'), Token('^'))
    t_or = Choice(Token('∨'), Token('|'), Token('v'))
    t_impl = Choice(Token('→'), Token('->'), Token('⇒'))
    t_equiv = Choice(Token('↔'), Token('<->'))
    t_xor = Choice(Token('⊕'))
    
    binary_op = Choice(t_and, t_or, t_impl, t_equiv, t_xor)
    
    # Binary Formula: (A op B)
    binary_formula = Sequence(unary_formula, binary_op, formula)

    # Quantifiers
    t_forall = Choice(Token('∀'), Token('A'))
    t_exists = Choice(Token('∃'), Token('E'))
    
    quantifier_sym = Choice(t_forall, t_exists)
    quantified_formula = Sequence(quantifier_sym, r_var_or_const, formula)

    # Unary
    unary_formula = Choice(atomic_formula, negation, quantified_formula)

    # 7. Final Formula Structure
    # We define the content of the 'formula' Ref
    formula = Choice(binary_formula, unary_formula)
    
    # Wrap in Repeat(..., 1, 1) to ensure the root element has a unique name "START"
    START = Repeat(formula, 1, 1)

class FOLGrammarTreeNode:
    def __init__(self, start=0, end=0, string="", node_type=""):
        self.start = start
        self.end = end
        self.string = string
        self.type = node_type
        self.children = []

class FolGrammarParser:
    def __init__(self):
        self.grammar = FolGrammar()
        self.expression2idx = {}
        self.code_sequence = []

    def get_code(self, node_type, children_attributes, node_string):
        """Generates the code line based on node_type and children outputs"""
        # --- 1. QUANTIFIERS ---
        if node_type == "quantified_formula":
            # [Sym, Var, Formula]
            quant_sym = children_attributes[0]['string'].strip()
            var_name  = children_attributes[1]['string'].strip()
            sub_id    = children_attributes[2]['id']
            
            func = "UniversalQuantification" if quant_sym in ['∀', 'A'] else "ExistentialQuantification"
            new_id = f"formula{len(self.expression2idx) + 1}"
            return new_id, f"{new_id} = {func}({sub_id}, '{var_name}')"

        # --- 2. BINARY OPS ---
        elif node_type == "binary_formula":
            # [Left, Op, Right]
            left_id = children_attributes[0]['id']
            op_str  = children_attributes[1]['string'].strip()
            right_id = children_attributes[2]['id']
            
            func_map = {
                '∧': 'Conjunction', '&': 'Conjunction', '^': 'Conjunction',
                '∨': 'Disjunction', '|': 'Disjunction', 'v': 'Disjunction',
                '→': 'Implication', '->': 'Implication', '⇒': 'Implication',
                '↔': 'Equivalence', '<->': 'Equivalence',
                '⊕': 'Nonequivalence'
            }
            func = func_map.get(op_str, 'Conjunction')
            new_id = f"formula{len(self.expression2idx) + 1}"
            return new_id, f"{new_id} = {func}({left_id}, {right_id})"

        # --- 3. NEGATION ---
        elif node_type == "negation":
            # [Sym, SubFormula]
            sub_id = children_attributes[1]['id']
            new_id = f"formula{len(self.expression2idx) + 1}"
            return new_id, f"{new_id} = Negation({sub_id})"

        # --- 4. PREDICATES & FUNCTIONS ---
        elif node_type in ["predicate", "func_term"]:
            name = children_attributes[0]['string'].strip()
            args_list_node = children_attributes[2] 
            
            args = []
            if 'children_attrs' in args_list_node and len(args_list_node['children_attrs']) > 0:
                for child in args_list_node.get('children_attrs', []):
                    if child['string'].strip() == ',': continue
                    args.append(child['id'])
            else:
                # The PyleriList wrapper was flattened, so args_list_node IS the single argument
                args.append(args_list_node['id'])

            new_id = f"formula{len(self.expression2idx) + 1}"
            args_str = ", ".join(args)
            
            if node_type == "predicate":
                return new_id, f"{new_id} = Predicate('{name}', [{args_str}])"
            else:
                return f"Function('{name}', [{args_str}])", None

        # --- 5. EQUALITY ---
        elif node_type == "eq_formula":
            left_id = children_attributes[0]['id']
            right_id = children_attributes[2]['id']
            new_id = f"formula{len(self.expression2idx) + 1}"
            return new_id, f"{new_id} = Equal({left_id}, {right_id})"
            
        elif node_type == "neq_formula":
            left_id = children_attributes[0]['id']
            right_id = children_attributes[2]['id']
            new_id = f"formula{len(self.expression2idx) + 1}"
            return new_id, f"{new_id} = NonEqual({left_id}, {right_id})"

        # --- 6. ATOMIC / WRAPPERS ---
        elif node_type == "parens_formula":
            # [ '(', formula, ')' ] -> return middle
            return children_attributes[1]['id'], None

        # --- 7. LEAF NODES ---
        elif node_type == "r_var_or_const":
            val = node_string
            if val[0].islower() and len(val) == 1: 
                return f"Variable('{val}')", None
            return f"Constant('{val}')", None
            
        elif node_type in ["r_upper_const", "r_quoted_const"]:
            return f"Constant('{node_string}')", None
        elif node_type == "k_true":
            return "'True'", None
        elif node_type == "k_false":
            return "'False'", None

        return f"'{node_string}'", None

    def construct_code_sequence(self, node: FOLGrammarTreeNode):
        children_attributes = []
        for child in node.children:
            children_attributes.append(self.construct_code_sequence(child))
            
        return_id, code = self.get_code(node.type, children_attributes, node.string)
        
        if code:
            self.code_sequence.append(code)
            idx = len(self.expression2idx) + 1
            self.expression2idx[node.string] = idx
            
            if return_id.startswith("formula"):
               return_id = return_id  # Keep it as formulaX variable 

        return {
            "type": node.type,
            "string": node.string,
            "id": return_id,
            "children_attrs": children_attributes
        }

    def map_pyleri_to_fol_node(self, pyleri_node):
        """Recursively converts Pyleri AST to FOLGrammarTreeNode"""
        if not hasattr(pyleri_node, 'element'):
            return FOLGrammarTreeNode(string=getattr(pyleri_node, 'string', ''), node_type="leaf")

        # Handle flattening for simple wrapper nodes unless they are special
        current_pyleri = pyleri_node
        while hasattr(current_pyleri, 'children') and len(current_pyleri.children) == 1:
            if hasattr(current_pyleri, 'element') and current_pyleri.element in [self.grammar.negation, self.grammar.quantified_formula]:
                break
            current_pyleri = current_pyleri.children[0]

        element = getattr(current_pyleri, 'element', None)
        
        # Determine the string name of the pyleri element type
        node_type = "unknown"
        if element == self.grammar.quantified_formula: node_type = "quantified_formula"
        elif element == self.grammar.binary_formula: node_type = "binary_formula"
        elif element == self.grammar.negation: node_type = "negation"
        elif element == self.grammar.predicate: node_type = "predicate"
        elif element == self.grammar.func_term: node_type = "func_term"
        elif element == self.grammar.eq_formula: node_type = "eq_formula"
        elif element == self.grammar.neq_formula: node_type = "neq_formula"
        elif element == self.grammar.parens_formula: node_type = "parens_formula"
        elif element == self.grammar.r_var_or_const: node_type = "r_var_or_const"
        elif element == self.grammar.r_upper_const: node_type = "r_upper_const"
        elif element == self.grammar.r_quoted_const: node_type = "r_quoted_const"
        elif element == self.grammar.k_true: node_type = "k_true"
        elif element == self.grammar.k_false: node_type = "k_false"
        elif hasattr(element, "name") and element.name: node_type = element.name

        fol_node = FOLGrammarTreeNode(string=getattr(current_pyleri, "string", ""), node_type=node_type)
        
        if hasattr(current_pyleri, 'children'):
             for child in current_pyleri.children:
                  fol_node.children.append(self.map_pyleri_to_fol_node(child))

        return fol_node

    def parse_and_convert(self, fol_string):
        self.expression2idx = {}
        self.code_sequence = []
        
        try:
            result = self.grammar.parse(fol_string)
        except Exception as e:
            return f"# Parsing Exception: {e}\nformula = End('Error')"

        if not result.is_valid:
            return f"# Invalid Syntax: {fol_string}\nformula = End('Error')" 
        
        root_node = result.tree.children[0]
        fol_tree = self.map_pyleri_to_fol_node(root_node)
        
        final_info = self.construct_code_sequence(fol_tree)
        
        if self.code_sequence:
            last_expr = self.code_sequence[-1].split("=")[0].strip()
            self.code_sequence.append(f"formula = End({last_expr})")
        else:
            self.code_sequence.append(f"formula = End({final_info['id']})")
            
        return "\n".join(self.code_sequence)
