from pprint import pformat

import networkx as nx

from ydnpd.generation.agent.utils import (clean_split,
                               build_graph,
                               is_valid_pyro_code,
                               retrieve_pyro_model)
from ydnpd.generation.agent.errors import AgentError


def raise_agenet_error(msg=None):
    raise AgentError(msg)


PYRO_FAILED_CODE_TEMPLATE = """
Pyro code validation failed. Please review and try again.

{last_check_info}

Always return Pyro code, ready to be executed without Markdown formatting, enclosed within the tags <Answer>...</Answer>.
"""

SPECIFICATION_V1 = {
    "__initial__": {
        "instruction_fn": lambda context: (
            "Consider the following data schema:\n"
            f"\n{pformat(context['metadata']['schema'])}.\n\n"
            "Please review it carefully to ensure you fully understand it. After your analysis, provide the full list of variables from the schema.\n"
            "Think step by step. Then, provide your final answer (variable names only, EXACTLY as they appear in the schema) enclosed within the tags <Answer>...</Answer>, separated by \", \"."
        )
    },
    "SCHEMA": {
        "processing_fn": lambda answer, context: {"reported_variables": set(clean_split(answer))},
        "check_fn": lambda answer, additional_context, context: (
            None if (additional_context["reported_variables"] == set(context["metadata"]["schema"].keys()))
            else raise_agenet_error()
        ),
    },
    "SCHEMA_failed": {
        "instruction_fn": lambda context: (
            "Please take another attempt to review the schema provided earlier until you fully understand it. Then, reply with the full list of variables from the schema.\n"
            "Think step by step. Then, provide your final answer (variable names only, EXACTLY as they appear in the schema) enclosed within the tags <Answer>...</Answer>, separated by \", \"."
        )
    },
    "SCHEMA_success": {
        "instruction_fn": lambda context: (
            "Based on the schema provided, you are going to create a comprehensive list of constraints that these variables must satisfy, using your domain knowledge.\n"
            "The constraints should not refer to the range of each variable separately, as defined by the schema. Instead, focus only on constraints involving two or more variables. For example, a person who is 14 years old cannot have 12 years of education.\n"
            "Provide your constraints as a list of equalities and inequalities that a record must satisfy, formatted as Python boolean expressions. These constraints may include constant numbers if necessary.\n"
            "Recall that 'X implies Y' can be expressed as 'not X or Y'.\n"
            "Think step by step. Then, provide your final answer (list of constraints) enclosed within the tags <Answer>...</Answer>, separated by new lines."
        )
    },
    "ELICIT_CONSTRAINTS": {
        "processing_fn": lambda answer, context: {"constraints": clean_split(answer, "\n")},
        "check_fn": lambda answer, additional_context, context: None,
    },
    "ELICIT_CONSTRAINTS_failed": {
        "instruction_fn": lambda context: (
            "There was an issue with the constraints you provided. Please review the instructions and generate a new set of constraints. Ensure that your constraints involve two or more variables and are formatted as Python boolean expressions.\n"
            "Think step by step. Then, provide your final answer (list of constraints) enclosed within the tags <Answer>...</Answer>, separated by new lines."
        )
    },
    "ELICIT_CONSTRAINTS_success": {
        "instruction_fn": lambda context: (
            "Now, you are going to construct a causal graph, relying on your expertise, given only the schema dictionary defining each variable's name and domain/range/categories. If you are unfamiliar with a variable name, infer its identity from the context.\n"
            "Begin by identifying which variable(s) should serve as the root nodes in a directed acyclic graph (DAG) that will represent a structural causal model between all variables. The best root variables are those that are unaffected by any other variables.\n"
            "Think step by step. Then, provide your final answer (variable names only, EXACTLY as they appear in the schema) enclosed within the tags <Answer>...</Answer>, separated by ', '."
        )
    },
    "ROOT_NODES": {
        "processing_fn": lambda answer, context: {"root_nodes": clean_split(answer)},
        "check_fn": lambda answer, additional_context, context: (
            None if (set(additional_context["root_nodes"]) <= set(context["metadata"]["schema"].keys()))
            else raise_agenet_error()
        ),
    },
    "ROOT_NODES_failed": {
        "instruction_fn": lambda context: (
            "The root node selection failed. The variables you suggested are not a subset of the variables in the schema. Please review the schema and try again.\n"
            "Think step by step. Then, provide your final answer (variable names only, EXACTLY as they appear in the schema) enclosed within the tags <Answer>...</Answer>, separated by ', '."
        )
    },
    "ROOT_NODES_success": {
        "instruction_fn": lambda context: (
            "Now, we will define relationships between the root variable(s) and the remaining variables in our causal graph, relying on your expertise. If you are unfamiliar with a variable name, infer its identity from the context.\n"
            "Identify relationships between root node variables and the remaining variables in the directed acyclic graph. Define a relationship between two variables as 'X -> Y', using the '->' operator to denote a directed edge.\n"
            "Think step by step. Then, provide your final answer (variable names only, using the '->' operator between each directed pair) enclosed within the tags <Answer>...</Answer>, separated by ', '."
        )
    },
    "ROOT_TO_NON_EDGES": {
        "processing_fn": lambda answer, context: {
            "relationships": set(clean_split(answer)),
        },
        "check_fn": lambda answer, additional_context, context: None,
    },
    "ROOT_TO_NON_EDGES_failed": {
        "instruction_fn": lambda context: (
            "There was an issue with connecting edges between root and non-root variables. Please review your relationships and ensure they are correctly specified using the '->' operator, and that variables are named exactly as in the schema.\n"
            "Think step by step. Then, provide your final answer (variable names only, using the '->' operator between each directed pair) enclosed within the tags <Answer>...</Answer>, separated by ', '."
        )
    },
    "ROOT_TO_NON_EDGES_success": {
        "instruction_fn": lambda context: (
            "Next, we will define any necessary relationships between variables that are NOT root variables in our causal graph, again relying on your expertise.\n"
            "Identify relationships between non-root variables. Remember, you can define a relationship between two variables as 'X -> Y', using the '->' operator to denote a directed edge.\n"
            "Be careful to ensure that the graph remains a DAG, so avoid introducing any cycles.\n"
            "Think step by step. Then, provide your final answer (variable names only, using the '->' operator between each directed pair) enclosed within the tags <Answer>...</Answer>, separated by ', '."
        )
    },
    "NON_TO_NON_EDGES": {
        "processing_fn": lambda answer, context:
        {
            "relationships": context["relationships"] | set(clean_split(answer)),
        },
        "check_fn": lambda answer, additional_context, context: None,
    },
    "NON_TO_NON_EDGES_failed": {
        "instruction_fn": lambda context: (
            "There was an issue with connecting edges between non-root variables. Please ensure that your relationships are correctly specified without introducing cycles, and that variables are named exactly as in the schema.\n"
            "Think step by step. Then, provide your final answer (variable names only, using the '->' operator between each directed pair) enclosed within the tags <Answer>...</Answer>, separated by ', '."
        )
    },
    "NON_TO_NON_EDGES_success": {
        "instruction_fn": lambda context: None,
    },
    "DAG": {
        "processing_fn": lambda answer, context: {
            "relationships": (relationships := clean_split(answer) if answer else context["relationships"]),
            "graph": build_graph(relationships),
        },
        "check_fn": lambda answer, additional_context, context: (
            None if (nx.is_directed_acyclic_graph(additional_context["graph"]))
            else raise_agenet_error()
        ),
    },
    "DAG_failed": {
        "instruction_fn": lambda context: (
            f"The list of relationships you defined for our causal graph introduces cycles, making it invalid as a DAG: {str(context['relationships'])}\n\n"
            "Please remove the minimum number of edges to eliminate the cycles while keeping the most important relationships according to your domain expertise. Ensure that EVERY variable is still included.\n"
            "Think step by step. Then, return the final list of relationships without any cycles.\n"
            "Provide your final answer (relationships only, using the '->' operator between each directed pair) enclosed within the tags <Answer>...</Answer>, separated by ', '."
        )
    },
    "DAG_success": {
        "instruction_fn": lambda context: (
            "In this step, you are going to specify a set of structural equations, which are functions that relate each node's value to the random variables of its parent nodes.\n"
            "For variables that are root nodes in the causal graph, parameterize a continuous or categorical distribution (depending on the variable type) using your domain expertise from which a random value can be drawn.\n"
            "For variables that have parent nodes, parameterize a conditional distribution, which is a function of the values of the parents of that variable, using your domain expertise, from which a random value can be drawn.\n"
            "Ensure that the values of the root variables and variables with parents stay within their schema-defined ranges (for continuous variables) or are valid codes (for categorical variables).\n"
            "It is very important that your model aligns completely with the variable relationships you identified. For your reference, here are those relationships:\n"
            f"\n{context['relationships']}.\n\n"
            "Think step by step. Then, provide your final answer as a set of Pyro-like formulas ('X ~ ...', where you insert the formula) enclosed within the tags <Answer>...</Answer>, separated by new lines."
        ),
    },
    "STRUCTURAL_EQUATIONS": {
        "processing_fn": lambda answer, context: {
            "pseudocode": answer
        },
        "check_fn": lambda answer, additional_context, context: None,
    },
    "STRUCTURAL_EQUATIONS_success": {
        "instruction_fn": lambda context: (
            "In this step, you will refine the model from the previous step and specify it completely.\n"
            "First, identify any free parameters that need to be set, and name them. Pay special attention to conditional probabilities in categorical distributions.\n"
            f"Second, set the values of all these parameters based on your expertise in {context['metadata']['domain']} (not just illustrative numbers), grounding them in the scientific knowledge that you possess. Provide reasoning for your choices.\n"
            "DO NOT USE PLACEHOLDERS. Set appropriate values for all parameters as elaborately as needed.\n"
            "Review your model and ensure that it matches the relationships (conditional dependencies) you have identified.\n"
            "Think step by step. Then, provide your final answer as a set of Pyro-like formulas ('X ~ ...') enclosed within the tags <Answer>...</Answer>, separated by new lines."
        )
    },
    "PARAMETERS": {
        "processing_fn": lambda answer, context: {
            "pseudocode": answer
        },
        "check_fn": lambda answer, additional_context, context: None,
    },
    "PARAMETERS_success": {
        "instruction_fn": lambda context: f"""
Finally, you are going to convert the following Pyro-like formulas into executable Pyro code to create a structural causal model (SCM) that can be sampled from.

Consider this example:
<Example>
X ~ Categorical([0.1, 0.5, 0.2, 0.2])
Y ~ Normal(2 * X, 1)
</Example>

You should convert this into Pyro code like:
<Example>
import pyro
import pyro.distributions as dist

def model():
    X = pyro.sample("X", dist.Categorical(probs=[0.1, 0.5, 0.2, 0.2]))
    Y = pyro.sample("Y", dist.Normal(2 * X, 1))
    return {{"X": X, "Y": Y}}
</Example>

Notes:
1. The module 'pyro.distributions' does not have an attribute 'TruncatedNormal'.
2. Consider using 'if' statements for conditional distributions, especially for categorical distributions.

Be sure to include all functionality INSIDE the 'model()' function—do not use helper functions. Now, please convert the following Pyro-like formulas into executable Pyro code:

{context["pseudocode"]}

Be sure to properly handle any distributions and functional relationships between variables.
Always return Pyro code, ready to be executed without Markdown formatting, enclosed within the tags <Answer>...</Answer>.
""",
    },
    "PYRO_CODE": {
        "processing_fn": lambda answer, context: {
            "code": answer,
        },
        "check_fn": lambda answer, additional_context, context: is_valid_pyro_code(additional_context["code"])
    },
    "PYRO_CODE_failed": {
        "instruction_fn": lambda context: PYRO_FAILED_CODE_TEMPLATE.format(**context),
    },
    "PYRO_CODE_success":  {
        "instruction_fn": lambda context: f"""
The model defined in your code can be compiled in Python—good job!
However, the model may produce samples that are outside of the ranges defined by the domain.
Include additional lines of code to enforce the type and range of the samples by casting, rounding, clipping, and affine transformations as necessary.
For your reference, here is the schema again:

{pformat(context['metadata']['schema'])}

Always return Pyro code, ready to be executed without Markdown formatting, enclosed within the tags <Answer>...</Answer>.
""",
    },
    "ENFORCE_RANGE": {
        "processing_fn": lambda answer, context: {
            "code": answer,
        },
        "check_fn": lambda answer, additional_context, context: is_valid_pyro_code(additional_context["code"], context["pandera_schema"])
    },
    "ENFORCE_RANGE_failed": {
        "instruction_fn": lambda context: PYRO_FAILED_CODE_TEMPLATE.format(**context),
    },
    "ENFORCE_RANGE_success":  {
        "instruction_fn": lambda context: f"""
The model defined in your code can be compiled in Python—good job!
Recall that you previously produced a list of constraints; now it is time to enforce them in your code.
Include additional lines of code at the end of your function (but before the return statement) to enforce the constraints using rejection sampling.
The function should always return a sample, and retry if it fails due to the constraints. Consider using a 'while True' loop with 'continue' to achieve this.
For your reference, here are the constraints again:

{chr(10).join(context['constraints'])}

Always return Pyro code, ready to be executed without Markdown formatting, enclosed within the tags <Answer>...</Answer>.
""",
    },
    "ENFORCE_CONSTRAINTS": {
        "processing_fn": lambda answer, context: {
            "code": answer,
        },
        "check_fn": lambda answer, additional_context, context: is_valid_pyro_code(additional_context["code"], context["pandera_schema"])
    },
    "ENFORCE_CONSTRAINTS_failed": {
        "instruction_fn": lambda context: PYRO_FAILED_CODE_TEMPLATE.format(**context),
    },
    "ENFORCE_CONSTRAINTS_success": {
        "instruction_fn": lambda context: None,
    },
    "FINITO": {
        "processing_fn": lambda answer, context: {
            "model": retrieve_pyro_model(context["code"]),
        },
        "check_fn": lambda answer, additional_context, context: None
    },
}