"""
pipeline.py
===========
Core natural language to logical AST parsing pipeline.
- Base parsing algorithms and structure derived from the original NL2Logic repository (Putra et al., 2026).
- Custom Environment Adaptations: Added `VLLMWrapper` and `OpenAIWrapper` for local async inference, 
  and updated prompt handling for compatibility with our evaluation environment.
"""

from ast_rl import *
from structured_output import *

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

try:
    from openai import OpenAI, AsyncOpenAI
except ImportError:
    OpenAI = None
    AsyncOpenAI = None

# --- CUSTOM ADAPTATION ---
# Added wrappers for async local inference (vLLM/OpenAI) to support automated pipeline evaluation.

class OpenAIWrapper:
    def __init__(self, model):
        self.model = model
        self.client = AsyncOpenAI()
    
    async def generate(self, text, fmt):
        if "Now, it is your turn" in text:
            parts = text.split("Now, it is your turn", 1)
            sys_msg = parts[0].strip()
            user_msg = "Now, it is your turn" + parts[1]
        elif "Now, classify this" in text:
            parts = text.split("Now, classify this", 1)
            sys_msg = parts[0].strip()
            user_msg = "Now, classify this" + parts[1]
        else:
            sys_msg = "You are a helpful logical parsing assistant."
            user_msg = text

        print(f"[DEBUG-PROMPT] SYS: {sys_msg[:100].replace('\n', ' ')}...")
        response = await self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": sys_msg},
                {"role": "user", "content": user_msg}
            ],
            response_format={"type": "json_object"}
        )
        import json
        return fmt(**json.loads(response.choices[0].message.content))


class VLLMWrapper:
    def __init__(self, model, url="http://localhost:8000/v1"):
        self.model = model
        self.url = url
    
    async def generate(self, text, fmt):
        if "Now, it is your turn" in text:
            parts = text.split("Now, it is your turn", 1)
            sys_msg = parts[0].strip()
            user_msg = "Now, it is your turn" + parts[1]
        elif "Now, classify this" in text:
            parts = text.split("Now, classify this", 1)
            sys_msg = parts[0].strip()
            user_msg = "Now, classify this" + parts[1]
        else:
            sys_msg = "You are a helpful logical parsing assistant."
            user_msg = text

        print(f"[DEBUG-PROMPT] SYS: {sys_msg[:100].replace('\n', ' ')}...")
        client = AsyncOpenAI(base_url=self.url, api_key="EMPTY")
        response = await client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": sys_msg},
                {"role": "user", "content": user_msg}
            ],
            response_format={"type": "json_object"},
            temperature=0,
            timeout=60
        )
        await client.close()
        import json
        return fmt(**json.loads(response.choices[0].message.content))

# --- END CUSTOM ADAPTATION ---

class Pipeline:
    # --- CUSTOM ADAPTATION ---
    # Refactored __init__ to use custom async LLM wrappers
    def __init__(self, llm, model, logging=False, url="http://localhost:8000/v1"):
        self.llm_type = llm
        if llm == 'openai':
            self.llm = OpenAIWrapper(model)
        elif llm == 'vllm':
            self.llm = VLLMWrapper(model,url)
        else: 
            raise ValueError("LLM is not valid")
        self.logging=logging
    # --- END CUSTOM ADAPTATION ---

    # --- ORIGINAL IMPLEMENTATION (Putra et al., 2026) ---
    # Core recursive parsing logic
    def log(self, text):
        if self.logging:
            print(text)
        
    async def _rephrase(self, text):
        r = await self.llm.generate(
                REPHRASE_SYSTEM_PROMPT + 'Now, it is your turn\n\nInput: "' + text + '"\nRephrased: ', 
                Rephrased
            )
        self.log(f"Rephrased '{text}' to '{r.rephrased}'")
        return r.rephrased
    
    async def rephrase_and_parse(self, text, row=None):
        text = await self._rephrase(text)
        return await self.parse(text, True, "")
        
    async def _parse_relation(self, text, prefix):
        choose_relation = await self.llm.generate(
                    CHOOSE_RELATION_SYSTEM_PROMPT + 'Now, it is your turn\n\nInput: "' + text + '"\nOutput: ', 
                    ChooseRelation)
        answer = choose_relation.answer.strip().upper()
        if answer == 'A':
            # Adjective
            p = await self.llm.generate(
                ADJECTIVE_SYSTEM_PROMPT + 'Now, it is your turn\n\nInput: "' + text + '"\nOutput: ', 
                AdjectiveParser
            )
            self.log(prefix + f"Adjective parser. Adjective: {p.adjective}, Object: {p.obj}")
            return RelationAdjective(obj=Constant(name=p.obj), adjective=p.adjective)
        elif answer == 'B':
            # Intransitive
            p = await self.llm.generate(
                INTRANSITIVE_SYSTEM_PROMPT + 'Now, it is your turn\n\nInput: "' + text + '"\nOutput: ', 
                IntransitiveParser
            )
            self.log(prefix + f"Intransitive parser. Verb: {p.verb}, Subject: {p.subject}")
            return RelationIntransitiveVerb(verb=p.verb, subject=Constant(name=p.subject))
        elif answer == 'C':
            # Transitive
            p = await self.llm.generate(
                TRANSITIVE_SYSTEM_PROMPT + 'Now, it is your turn\n\nInput: "' + text + '"\nOutput: ', 
                TransitiveParser
            )
            self.log(prefix + f"Transitive parser. Verb: {p.verb}, Subject: {p.subject}, Object: {p.obj}")
            return RelationTransitiveVerb(verb=p.verb, subject=Constant(name=p.subject), obj=Constant(name=p.obj))
        elif answer == 'D':
            # Ditransitive
            p = await self.llm.generate(
                DITRANSITIVE_SYSTEM_PROMPT + 'Now, it is your turn\n\nInput: "' + text + '"\nOutput: ', 
                DitransitiveParser
            )
            self.log(prefix + f"Ditransitive parser. Verb: {p.verb}, Subject: {p.subject}, Indirect Object: {p.indirect_obj}, Direct Object: {p.direct_obj}")
            return RelationDitransitiveVerb(verb=p.verb, subject=Constant(name=p.subject), indirect_obj=Constant(name=p.indirect_obj), direct_obj=Constant(name=p.direct_obj))
        else:
            raise ValueError("Invalid relation option")
        
    async def _parse_quantified(self, text, prefix):
        p = await self.llm.generate(
                QUANTIFIED_SYSTEM_PROMPT + 'Now, it is your turn\n\nInput: "' + text + '"\nOutput: ', 
                QuantifiedParser
            )
        quant = p.quantifier.strip()
        # Fix casing gracefully
        if quant.lower() == 'forall': quant = 'ForAll'
        elif quant.lower() == 'thereexists': quant = 'ThereExists'
        self.log(prefix + f"Quantified parser. Quantifier: {quant}, Variable: {p.variable}")
        if p.sentence_without_quantifier.lower() == text.lower() or p.sentence_without_quantifier == "":
            return await self._parse_relation(text, prefix)
        else:
            s = await self.parse(p.sentence_without_quantifier, True, prefix)
            return QuantifiedSentence(quantifier=quant, variable=Variable(name=p.variable if p.variable != "" else "x"), sentence=s)

    async def _parse_binary(self, text, prefix):
        p = await self.llm.generate(
                BINARY_LOGICAL_SYSTEM_PROMPT + 'Now, it is your turn\n\nInput: "' + text + '"\nOutput: ', 
                BinaryLogicalParser
            )
        op = p.operator.strip()
        if op.lower() == 'and': op = 'And'
        elif op.lower() == 'or': op = 'Or'
        elif op.lower() == 'if': op = 'If'
        elif op.lower() == 'onlyif': op = 'OnlyIf'
        elif op.lower() == 'ifandonlyif': op = 'IfAndOnlyIf'
        self.log(prefix + f"Binary operator parser. Operator: {op}")
        if p.left_operand.lower() == text.lower() or p.right_operand.lower() == text.lower() or p.left_operand == "" or p.right_operand == "":
            return await self._parse_relation(text, prefix)
        else:
            left = await self.parse(p.left_operand, False, prefix)
            right = await self.parse(p.right_operand, True, prefix)
            return BinaryOperator(operator=op, left=left, right=right)
    
    async def _parse_unary(self, text, prefix):
        p = await self.llm.generate(
            UNARY_LOGICAL_SYSTEM_PROMPT + 'Now, it is your turn\n\nInput: "' + text + '"\nOutput: ', 
            UnaryLogicalParser
        )
        op = 'Not' if p.operator.strip().lower() == 'not' else p.operator.strip()
        self.log(prefix + f"Unary operator parser. Operator: {op}")
        for w in ["not", "do not", "dont", "don't", "does not", "doesn't"]:
            if w not in text.lower() and w in p.operand.lower():
                return await self._parse_relation(text, prefix)
        if p.operand.lower() == text.lower() or p.operand.lower() == "":
            return await self._parse_relation(text, prefix)
        else:
            s = await self.parse(p.operand, True, prefix)
            return UnaryOperator(operator=op, sentence=s)

    async def normalize_pipeline(self, relational_logic: RelationalLogic):
        self.log("\n[NORMALIZATION PHASE]")
        all_preds = relational_logic.get_all_predicates()
        # Deduplicate to avoid redundant LLM work
        unique_preds = []
        seen = set()
        for p in all_preds:
            key = (p['name'], p['arity'])
            if key not in seen:
                unique_preds.append(p)
                seen.add(key)
        
        if not unique_preds:
            self.log("No predicates found for normalization.")
            return

        import json
        prompt = NORMALIZATION_SYSTEM_PROMPT + "\nInput: " + json.dumps(unique_preds) + "\nOutput: "
        result = await self.llm.generate(prompt, NormalizationResult)
        
        self.log(f"Normalization mapping: {result.mapping}")
        relational_logic.apply_normalization(result.mapping)
        self.log("Normalization applied successfully.\n")

    async def parse(self, text, last, prefix):
        if last:
            p = "     "
        else:
            p = "│    "

        if last:
            q = "└────"
        else:
            q = "├────"
        self.log( prefix + q + f"Parsing '{text}'")
        choose_parser =  await self.llm.generate(
                    CHOOSE_PARSER_SYSTEM_PROMPT + "Now, classify this\n\nSentence: '" + text + "'\nAnswer: ", 
                    ChooseParser)
        ans = choose_parser.answer.strip().upper()
        prefix += p
        self.log( prefix + f"Answer: {ans}")
        if ans == 'A':
            # Relation
            return await self._parse_relation(text, prefix)
        elif ans == 'B':
            # Quantified
            return await self._parse_quantified(text, prefix)
        elif ans == 'C':
            # Binary operator:
            return await self._parse_binary(text, prefix)
        elif ans == 'D':
            # Unary operator
            return await self._parse_unary(text, prefix)
        else:
            raise ValueError("Invalid parser option")
        