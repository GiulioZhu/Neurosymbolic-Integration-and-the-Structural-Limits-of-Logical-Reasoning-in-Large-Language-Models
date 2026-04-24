"""
ast_rl.py
=========
Typed Abstract Syntax Tree nodes for representing First-Order Logic formulas.
- Pydantic models and structural definitions derived from the original NL2Logic repository (Putra et al., 2026).
- Custom Environment Adaptations: Extended with Z3 generation passes (`z3declaration_pass`, 
  `z3expression_pass`, and `convert_to_z3`) to interface directly with our automated verifier.
"""

from typing import List, Union, Literal
from pydantic import BaseModel, Field, PrivateAttr
import re

def sanitize_id(name: str) -> str:
    s = re.sub(r'\W+', '_', name.strip())
    if not s:
        return "_empty"
    if s[0].isdigit():
        return "_" + s
    return s

def escape_str(s: str) -> str:
    return s.replace("'", "\\'")
# --- ORIGINAL IMPLEMENTATION (Putra et al., 2026) ---
# The base Pydantic AST models and structural methods (to_dict, getChild, etc.)

class Constant(BaseModel):
    name : str
    def __str__(self):
        return self.name
    
    def to_dict(self):
        return {
            "node_type": "Constant",
            "name": self.name
        }

    def getChild(self):
        return []
    
    # --- CUSTOM ADAPTATION ---
    # The Z3 generation passes (z3declaration_pass, z3expression_pass) 
    # were injected into ALL node classes to compile the AST into executable Z3 code.
    def z3declaration_pass(self):
        v = sanitize_id(self.name)
        return f"c_{v} = Const('{escape_str(self.name)}', Entity)\n"
    
    def z3expression_pass(self):
        return f"c_{sanitize_id(self.name)}"

class Variable(BaseModel):
    _id : int = PrivateAttr()
    name : str
    def __str__(self):
        return self.name
    
    def to_dict(self):
        return {
            "node_type": "Variable",
            "name": self.name
        }

    def getChild(self):
        return []
    
    def z3declaration_pass(self):
        v = sanitize_id(self.name)
        return f"c_{v} = Const('{escape_str(self.name)}', Entity)\n"
    
    def z3expression_pass(self):
        return f"c_{sanitize_id(self.name)}"

Term = Union[Constant, Variable]

class RelationAdjective(BaseModel):
    adjective : str
    obj : Term
    _normalized_name : str = PrivateAttr(default=None)

    def get_all_predicates(self):
        return [{"name": self.adjective, "arity": 1, "type": "Adjective"}]

    def apply_normalization(self, mapping):
        if self.adjective in mapping:
            self._normalized_name = mapping[self.adjective]

    def __str__(self):
        return f"{self.adjective}({self.obj})"
    
    def to_dict(self):
        return {
            "node_type": "RelationAdjective",
            "adjective": self.adjective,
            "obj": self.obj
        }

    def getChild(self):
        return [self.obj]
    
    def z3declaration_pass(self):
        result = self.obj.z3declaration_pass()
        v = sanitize_id(self._normalized_name or self.adjective)
        result += f"f1_{v} = Function('{escape_str(self._normalized_name or self.adjective)}', Entity, BoolSort())\n"
        return result
    
    def z3expression_pass(self):
        v = sanitize_id(self._normalized_name or self.adjective)
        return f"f1_{v}({self.obj.z3expression_pass()})"

class RelationIntransitiveVerb(BaseModel):
    verb : str
    subject : Term
    _normalized_name : str = PrivateAttr(default=None)

    def get_all_predicates(self):
        return [{"name": self.verb, "arity": 1, "type": "IntransitiveVerb"}]

    def apply_normalization(self, mapping):
        if self.verb in mapping:
            self._normalized_name = mapping[self.verb]

    def __str__(self):
        return f"{self.verb}({self.subject})"
    
    def to_dict(self):
        return {
            "node_type": "RelationIntransitiveVerb",
            "verb": self.verb,
            "subject": self.subject
        }

    def getChild(self):
        return [self.subject]
    
    def z3declaration_pass(self):
        result = self.subject.z3declaration_pass()
        v = sanitize_id(self._normalized_name or self.verb)
        result += f"f1_{v} = Function('{escape_str(self._normalized_name or self.verb)}', Entity, BoolSort())\n"
        return result
    
    def z3expression_pass(self):
        v = sanitize_id(self._normalized_name or self.verb)
        return f"f1_{v}({self.subject.z3expression_pass()})"

class RelationTransitiveVerb(BaseModel):
    verb : str
    subject : Term
    obj : Term
    _normalized_name : str = PrivateAttr(default=None)

    def get_all_predicates(self):
        return [{"name": self.verb, "arity": 2, "type": "TransitiveVerb"}]

    def apply_normalization(self, mapping):
        if self.verb in mapping:
            self._normalized_name = mapping[self.verb]

    def __str__(self):
        return f"{self.verb}({self.subject},{self.obj})"
    
    def to_dict(self):
        return {
            "node_type": "RelationTransitiveVerb",
            "verb": self.verb,
            "subject": self.subject,
            "obj" : self.obj
        }

    def getChild(self):
        return [self.subject, self.obj]
    
    def z3declaration_pass(self):
        result = self.subject.z3declaration_pass()
        result += self.obj.z3declaration_pass()
        v = sanitize_id(self._normalized_name or self.verb)
        result += f"f2_{v} = Function('{escape_str(self._normalized_name or self.verb)}', Entity, Entity, BoolSort())\n"
        return result
    
    def z3expression_pass(self):
        v = sanitize_id(self._normalized_name or self.verb)
        return f"f2_{v}({self.subject.z3expression_pass()},{self.obj.z3expression_pass()})"

class RelationDitransitiveVerb(BaseModel):  
    verb : str
    subject : Term
    direct_obj : Term
    indirect_obj : Term
    _normalized_name : str = PrivateAttr(default=None)

    def get_all_predicates(self):
        return [{"name": self.verb, "arity": 3, "type": "DitransitiveVerb"}]

    def apply_normalization(self, mapping):
        if self.verb in mapping:
            self._normalized_name = mapping[self.verb]

    def __str__(self):
        return f"{self.verb}({self.subject},{self.indirect_obj},{self.direct_obj})"
    
    def to_dict(self):
        return {
            "node_type": "RelationDitransitiveVerb",
            "verb": self.verb,
            "subject": self.subject,
            "direct_obj" : self.direct_obj,
            "indirect_obj" : self.indirect_obj
        }

    def getChild(self):
        return [self.subject, self.indirect_obj, self.direct_obj] 
    
    def z3declaration_pass(self):
        result = self.subject.z3declaration_pass()
        result += self.indirect_obj.z3declaration_pass()
        result += self.direct_obj.z3declaration_pass()
        v = sanitize_id(self._normalized_name or self.verb)
        result += f"f3_{v} = Function('{escape_str(self._normalized_name or self.verb)}', Entity, Entity, Entity, BoolSort())\n"
        return result
    
    def z3expression_pass(self):
        v = sanitize_id(self._normalized_name or self.verb)
        return f"f3_{v}({self.subject.z3expression_pass()},{self.indirect_obj.z3expression_pass()},{self.direct_obj.z3expression_pass()})"

RelationalSentence = Union[RelationIntransitiveVerb, RelationAdjective, RelationTransitiveVerb, RelationDitransitiveVerb]

class BinaryOperator(BaseModel):
    left : "Sentence"
    right : "Sentence"
    operator : Literal["And", "Or", "If", "OnlyIf", "IfAndOnlyIf"]
    def __str__(self):
        op_map = {
            "And": "∧",
            "Or": "∨",
            "If": "→",
            "OnlyIf": "←",
            "IfAndOnlyIf": "↔"
        }
        return f"({self.left}) {op_map[self.operator]} ({self.right})"
    
    def to_dict(self):
        return {
            "node_type": "BinaryOperator",
            "operator": self.operator,
            "left": self.left.to_dict(),
            "right": self.right.to_dict()
        }

    def getChild(self):
        return [self.left, self.right]
    
    def get_all_predicates(self):
        return self.left.get_all_predicates() + self.right.get_all_predicates()

    def apply_normalization(self, mapping):
        self.left.apply_normalization(mapping)
        self.right.apply_normalization(mapping)

    def z3declaration_pass(self):
        result = self.left.z3declaration_pass()
        result += self.right.z3declaration_pass()
        return result
    
    def z3expression_pass(self):
        if self.operator == "And":
            return f"And({self.left.z3expression_pass()}, {self.right.z3expression_pass()})"
        elif self.operator == "Or":
            return f"Or({self.left.z3expression_pass()}, {self.right.z3expression_pass()})"
        elif self.operator == "If":
            return f"Implies({self.left.z3expression_pass()}, {self.right.z3expression_pass()})"
        elif self.operator == "OnlyIf":
            return f"Implies({self.right.z3expression_pass()}, {self.left.z3expression_pass()})"
        elif self.operator == "IfAndOnlyIf":
            return f"{self.left.z3expression_pass()} == {self.right.z3expression_pass()}"
        else:
            raise ValueError(f"Unknown operator: {self.operator}")
        
class UnaryOperator(BaseModel):
    sentence : "Sentence"
    operator : Literal["Not"]

    def __str__(self):
        return f"¬({self.sentence})"
    
    def to_dict(self):
        return {
            "node_type": "UnaryOperator",
            "operator": self.operator,
            "sentence": self.sentence.to_dict()
        }

    def getChild(self):
        return [self.sentence]
    
    def get_all_predicates(self):
        return self.sentence.get_all_predicates()

    def apply_normalization(self, mapping):
        self.sentence.apply_normalization(mapping)

    def z3declaration_pass(self):
        return self.sentence.z3declaration_pass()
    
    def z3expression_pass(self):
        if self.operator == "Not":
            return f"Not({self.sentence.z3expression_pass()})"
        else:
            raise ValueError(f"Unknown operator: {self.operator}")

LogicalSentence = Union[BinaryOperator, UnaryOperator]

class QuantifiedSentence(BaseModel):
    quantifier : Literal["ForAll", "ThereExists"]
    variable : Variable
    sentence : "Sentence"
    
    def __str__(self):
        if self.quantifier == "ForAll":
            c = "∀"
        elif self.quantifier == "ThereExists":
            c = "∃"
        return f"{c}{self.variable}. ({self.sentence})"
    
    def to_dict(self):
        return {
            "node_type": "QuantifiedSentence",
            "quantifier": self.quantifier,
            "variable": self.variable.to_dict(),
            "sentence": self.sentence.to_dict()
        }

    def getChild(self):
        return [self.variable, self.sentence]
    
    def get_all_predicates(self):
        return self.sentence.get_all_predicates()

    def apply_normalization(self, mapping):
        self.sentence.apply_normalization(mapping)

    def z3declaration_pass(self):
        result = self.variable.z3declaration_pass()
        result += self.sentence.z3declaration_pass()
        return result
    
    def z3expression_pass(self):
        if self.quantifier == "ForAll":
            return f"ForAll([{self.variable.z3expression_pass()}], {self.sentence.z3expression_pass()})"
        elif self.quantifier == "ThereExists":
            return f"Exists([{self.variable.z3expression_pass()}], {self.sentence.z3expression_pass()})"
        else:
            raise ValueError(f"Unknown quantifier: {self.quantifier}")
    

Sentence = Union[RelationalSentence, LogicalSentence, QuantifiedSentence]

for m in (UnaryOperator, BinaryOperator, QuantifiedSentence):
    m.model_rebuild()

class RelationalLogic(BaseModel):
    original_sentence : str
    sentences : List[Sentence]
    def __str__(self):
        result = ""
        for s in self.sentences:
            result += str(s)
            result += "\n"
        return result
    
    def to_dict(self):
        return {
            "node_type": "RelationalLogic",
            "original_sentence" : self.original_sentence,
            "sentences": [s.to_dict() for s in self.sentences],
            "text" : str(self)
        }
    
    def getChild(self):
        return self.sentences
    
    def get_all_predicates(self):
        preds = []
        for s in self.sentences:
            preds.extend(s.get_all_predicates())
        return preds

    def apply_normalization(self, mapping):
        for s in self.sentences:
            s.apply_normalization(mapping)

    # --- CUSTOM ADAPTATION (Giulio Zhu) ---
    # Root Z3 compilation methods
    def z3declaration_pass(self):
        result = []
        for s in self.sentences:
            decl = s.z3declaration_pass()
            result.append(decl)
        result.sort()
        return "Entity = DeclareSort('Entity')\n\n" + "".join(result) + "\n\n"
    

    def z3expression_pass(self):
        result = []
        for s in self.sentences:
            expr = s.z3expression_pass()
            result.append(f"s.add({expr})\n")
        return "".join(result)
    
    def convert_to_z3(self):
        z3_code = "# Auto-generated Z3 code\n\n"
        z3_code += "from z3 import *\n\n"
        z3_code += "Entity = DeclareSort('Entity')\n\n"
        z3_code += "s = Solver()\n\n"
        z3_code += "# --- Declarations ---\n\n"
        z3_code += self.z3declaration_pass()
        z3_code += "# --- Expressions ---\n\n"       
        z3_code += self.z3expression_pass()
        z3_code += "print(f'Checking satisfiability...')\n"
        z3_code += "result = s.check()\n"
        z3_code += "print(f'Result: {result}')\n"
        return z3_code