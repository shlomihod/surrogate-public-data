import networkx as nx

from ydnpd.generation.agent.utils import (clean_split,
                               build_graph,
                               is_valid_pyro_code,
                               retrieve_pyro_model)


PROMPT_SUFFIX = "Think step by step. Then, provide your final answer (variable names only, EXACTLY as they appear in the schema) within the tags <Answer>...</Answer>, separated by \", \". "

PYRO_FAILED_CODE_TEMPLATE = """
            Pyro code validation failed, please review and try again.
            Here is the Traceback:
            ```
            {last_check_info}
            ```
            ALWAYS return Pyro code, ready to be execute without Markdown formattig, within the tags <Answer>...</Answer>.
            """


SPECIFICATION_V0 = {
    "__initial__": {
        "instruction_fn": lambda context: (
            f"Consider the following data schema: {context['metadata']['schema']}.\n"
            " Review you and make sure you understand it. Reply with the full list of variables from the scheme."
            f" {PROMPT_SUFFIX}"
        )
    },
    "SCHEMA": {
        "processing_fn": lambda answer, context: {"reported_variables": set(clean_split(answer))},
        "check_fn": lambda answer, additional_context, context: ((additional_context["reported_variables"]
                                                                    == set(context["metadata"]["schema"])),
                                                                    None),
    },
    "SCHEMA_failed": {
            "instruction_fn": lambda context: (
                f"Take anoter attempt to review the scheme provided before until you understand. Reply with the full list of variables from the scheme.\n"
                f" {PROMPT_SUFFIX}"
            )
    },
    "SCHEMA_success": {
        "instruction_fn": lambda context: (
            f"Consider the scheme provided before.\n"
            " You are going to create an exhausting list of constraints that this variables must satisfy using your knowledge."
            " The constraints do not refer to the range of each variable seperatly, as defined by the scheme."
            " We care only about CONSTRAINTS involving two or more variables."
            " For example, a perosn 14 years old cannot have have 12 years of education."
            " You will provide your constraints as a list of equalites and inequalites that a record must sastisfy formatted as Python boolean expression. "
            " It might also include constant number, if necessary."
            " Recall that `X implies Y` can be expressed as `not X or Y`"
            " Think step by step. Then, provide your final answer (list of constraints) within the tags <Answer>...</Answer>, separated by a new line. "
        )
    },
    "ELICIT_CONSTRAINTS": {
        "processing_fn": lambda answer, context: {"constraints": clean_split(answer, "\n")},
        "check_fn": lambda answer, additional_context, context: (True, None),
    },
    "ELICIT_CONSTRAINTS_failed": {
        "instruction_fn": lambda context: f"Constraints generation failed. Try again.",
    },
    "ELICIT_CONSTRAINTS_success": {
        "instruction_fn": lambda context: (
            f"Consider the scheme provided before.\n"
            " You are going to construct a causal graph, relying on your expertise, given only the above schema dictionary defining each variable name and domain/range/categories. When you are unfamiliar with a variable name, infer its identity from the context.\n"
            " You will start by identifying which variable(s) should serve as the root nodes in a directed acyclic graph (DAG), which will represent a structural causal model between all variables (the best root variables are unaffected by any other variables)."
            f" {PROMPT_SUFFIX}"
        )
    },
    "ROOT_NODES": {
        "processing_fn": lambda answer, context: {"root_nodes": clean_split(answer)},
        "check_fn": lambda answer, additional_context, context: (set(additional_context["root_nodes"])
                                                                    <= set(context["metadata"]["schema"].keys()),
                                                                    None),
    },
    "ROOT_NODES_failed": {
        "instruction_fn": lambda context: (
            f"The root node check failed."
            " The suggested root nodes set is not a subset of the variables in the schema."
            " Please review and try again. "
            f" {PROMPT_SUFFIX}",
        )
    },
    "ROOT_NODES_success": {
        "instruction_fn": lambda context: (
            " Now, we are going to relate the root variables to other variables in our causal graph, relying on your expertise. Again, when you are unfamiliar with a variable name, infer its identity from the context.\n"
            " You will now identify relationships between root node variable(s) and remaining variables in the directed acyclic graph. Define a relationship between two variables as 'X -> Y', using the '->' operator to denote a directed edge.\n"
            " hink step by step. Then, provide your final answer (variable names only, with the '->' operator between each directed pair) within the tags <Answer>...</Answer>, separated by ', '."
        ),
    },
    "ROOT_TO_NON_EDGES": {
        "processing_fn": lambda answer, context: {
            "relationships": set(clean_split(answer)),
        },
        "check_fn": lambda answer, additional_context, context: (True, None),
    },
    "ROOT_TO_NON_EDGES_failed": {
        "instruction_fn": lambda context: f"Edge connection failed, please review and try again. {PROMPT_SUFFIX}",
    },
    "ROOT_TO_NON_EDGES_success": {
        "instruction_fn": lambda context: (
            "Now, we are going to define any necessary relationships between variables that are NOT root variables in our causal graph, again relying on your expertise.\n"
            "You will now identify relationships between non-root variable(s). Remember, you can define a relationship between two variables as 'X -> Y', using the '->' operator to denote a directed edge.\n"
            "Think step by step. Remember, the graph is a DAG, so be careful not to introduce any cycles! Provide your final answer (variable names only, with the '->' operator between each directed pair) within the tags <Answer>...</Answer>, separated by ', '."
        )
    },
    "NON_TO_NON_EDGES": {
        "processing_fn": lambda answer, context: 
        {
            "relationships": context["relationships"] | set(clean_split(answer)),
        },
        "check_fn": lambda answer, additional_context, context: (True, None),
    },
    "NON_TO_NON_EDGES_failed": {
        "instruction_fn": lambda context: f"Edge connection failed, please review and try again. {PROMPT_SUFFIX}",
    },
    "NON_TO_NON_EDGES_success": {
        "instruction_fn": lambda context: None,
    },
    "DAG": {
        "processing_fn": lambda answer, context: {
            "relationships": (relationships := clean_split(answer) if answer else context["relationships"]),
            "graph": build_graph(relationships),
        },
        "check_fn": lambda answer, additional_context, context: (nx.is_directed_acyclic_graph(additional_context["graph"]),
                                                                    None),
    },
    "DAG_failed": {
        "instruction_fn": lambda context: (
            f"The list of relationships you defined for our causal graph introduces cycles, making it invalid as a DAG: {str(context['relationships'])} \n\n"
            "Please remove the minimum number of edges to eliminate the cycles, while keeping the most important relationships according to your domain expertise. Make sure, however, that EVERY variable is still included."
            "Think step by step. Then, return the final list of relationships without any cycles. Provide your final answer (relationships only, with the '->' operator between each directed pair) within the tags <Answer>...</Answer>, separated by ', '."
)
    },
    "DAG_success": {
        "instruction_fn": lambda context: (
            "In the penultimate step, you are going to specify a set of structural equations, which are functions that relate each node's value to the random variables of its parents.\n"
            "For variables that are root nodes in the causal graph, parameterize a continuous or categorical distribution (depending on the variable type) using your domain expertise from which a random value can be drawn.\n"
            "For variables that have parent nodes, parameterize a conditional distribution, which is a function of the values of the parents of that variable, using your domain expertise, from which a random value can be drawn.\n"
            "Be careful to ensure that the values of the root variables and variables with parents stay within their schema-defined range (in the case of continuous variables), or are valid codes (in the case of categorical variables).\n"
            f"Also, it is very important that your model is in full alignment in terms of variable relationships you found. There are given here for references: {context['relationships']}."
            "Think step by step. Then, provide your final answer as a set of Pyro-like formulas ('X ~ ...', where you insert the formula) within the tags <Answer>...</Answer>, separated by newlines."
            ),
    },
    "STRUCTURAL_EQUATIONS": {
        "processing_fn": lambda answer, context: {
            "pseudocode": answer
        },
        "check_fn": lambda answer, additional_context, context: (True, None),
    },
    "STRUCTURAL_EQUATIONS_success": {
        "instruction_fn": lambda context: f"""In this step you will iterate on the model from the last step, and specificy it completely.
        In other words, it will contains all information needed in order to perform sampling.
        First, identify the free parameters needed to be set, and name them. Pay attention especially to conditional probabilites in categorical distributions, but not only.
        Second, set the values of all these parameters based on your expertise in {context['metadata']['domain']} (not just illustrative numbers) and ground them in the scientific knowledge that you possess. Provide it with reasoning.
        Refiew your model and make sure that it matches the relationships (conditional dependeinces) you have identifed.
        Think step by step. Then, provide your final answer as a set of Pyro-like formulas ('X ~ ...', where you insert the formula) within the tags <Answer>...</Answer>, separated by newlines."""
    },
    "PARAMETERS": {
        "processing_fn": lambda answer, context: {
            "pseudocode": answer
        },
        "check_fn": lambda answer, additional_context, context: (True, None),
    },
    "PARAMETERS_success": {
        "instruction_fn": lambda context: f'''
        Finally, you are going to convert the following Pyro-like formulas into executable Pyro code to create a structural causal model (SCM) that I can sample from.
        The formulas specify how each variable is generated based on its parents in a directed acyclic graph (DAG).

        Consider this example:
        <Example>
        X ~ Normal(0, 1)
        Y ~ Normal(2 * X, 1)
        </Example>

        You should convert this into Pyro code like:
        import pyro
        import pyro.distributions as dist

        def model():
            X = pyro.sample("X", dist.Normal(0, 1))
            Y = pyro.sample("Y", dist.Normal(2 * X, 1))
            return {{"X": X, "Y": Y}}
        </Example>
        Notes:
        1. The module 'pyro.distributions' has no attribute 'TruncatedNormal'.
        2. Consider the use of if statements.

        Be careful to include all functionality INSIDE of the model() function - no helpers! Now, please convert the following Pyro-like formulas into executable Pyro code:\n

        {context["pseudocode"]}

        Be sure to properly handle any distributions and functional relationships between variables.
        ALWAYS return Pyro code, ready to be execute without Markdown formattig, within the tags <Answer>...</Answer>.
        ''',
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
        "instruction_fn": lambda context: f'''
        The model defined in your code can be compiled in Pyhton - Good job!
        However, the model probably produces samples that are outside of the ranges define by the domain.
        Include additional line of code to force the type and range of the samples by casting, rounding, clipping and affine transformation.
        For your reference, here is the scheme again:
        {context['metadata']['schema']}.
        ALWAYS return Pyro code, ready to be execute without Markdown formattig, within the tags <Answer>...</Answer>.
        ''',
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
        "instruction_fn": lambda context: f'''
        The model defined in your code can be compiled in Pyhton - Good job!
        Recall that you produced a list constraints, now it is the time to enforce them in your code.
        Include additional lines of code at the end of your function (but before the return statement) to force the constraints with reject sampling.
        The function should always return a sample, and try again if it fails due to the constraints. Consider using `while True` with `continue` for achieving that.
        For your reference, here is the constraints again:
        {context['constraints']}.
        ALWAYS return Pyro code, ready to be execute without Markdown formattig, within the tags <Answer>...</Answer>.
        ''',
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
        "check_fn": lambda answer, additional_context, context: (True, None)
    },
}
