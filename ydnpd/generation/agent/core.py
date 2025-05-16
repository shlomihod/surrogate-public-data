import re
import sys
import traceback
from datetime import datetime
from collections import Counter
from pprint import pprint
from enum import Enum

from statemachine import StateMachine
from statemachine.states import States
import weave

from ydnpd.utils import metadata_to_pandera_schema
from ydnpd.generation.agent.errors import AgentError
from ydnpd.generation.llm import create_llm


MAX_ATTEMPTS = 8


class LLMSession:
    def __init__(self, specification, metadata,
                 llm_path="openai/gpt-4o-mini",
                 llm_temperature=0.7,
                 llm_max_tokens=8192,
                 llm_top_p=1,
                #  llm_frequency_penalty=0,
                #  llm_presence_penalty=0,
                 verbose=False):

        self.specification = specification

        if llm_path.startswith("openai") and "o1" in llm_path:
            initial_role = "user"
            # max_token_param_name = "max_completion_tokens"
            if llm_temperature != 1:
                raise ValueError("temperature must be 1 for this model")
        else:
            initial_role = "system"
            # max_token_param_name = "max_tokens"

        self.context = {
            "metadata": metadata,
            "last_check_info": None,
            "pandera_schema": metadata_to_pandera_schema(metadata["schema"]),
        }

        self.llm_params = {
            "model_path": llm_path,
            "temperature": llm_temperature,
            "max_tokens": llm_max_tokens,
            "top_p": llm_top_p,
            # "frequency_penalty": llm_frequency_penalty,
            # "presence_penalty": llm_presence_penalty,
        }

        self.chat_fn = create_llm(**self.llm_params)

        self.message_history = [
            {
                "role": initial_role,
                "content": f"You are an expert on {metadata['domain']}."
            }
        ]

        self.attempts = Counter()

        self.verbose = verbose
        self.last_transition_time = datetime.now()

    def chat_complete(self, user_message, with_answer=True):

        self.message_history.append({
            "role": "user",
            "content": user_message
            })

        # if self.verbose:
        #     pprint(f"USER: {user_message}")

        # try:
        assistant_message = self.chat_fn(self.message_history)

        self.message_history.append({
            "role": "assistant",
            "content": assistant_message
        })

        if with_answer and (match := re.search(r"<Answer>(.*?)</Answer>",
                                               assistant_message,
                                               re.DOTALL)):

            answer = match.group(1).strip()

        else:
            answer = None

        return assistant_message, answer

    def execute_step(self, instruction_fn, process_fn, check_fn):

        prompt = instruction_fn(self.context)

        if prompt:
            _, answer = self.chat_complete(prompt)
            if self.verbose:
                print(f"{answer=}")
        else:
            answer = None

        try:
            additional_context = process_fn(answer, self.context)

            if self.verbose:
                print(f"{additional_context=}")

            check_info = check_fn(answer,
                                  additional_context,
                                  self.context)

            check_result = True

            self.context |= additional_context

            if self.verbose:
                print(f"{self.context=}")

        except AgentError as err:
            check_result = False
            check_info = str(err)

        finally:
            self.context["last_check_info"] = check_info

            if self.verbose:
                print(f"{check_result=}")
                print(f"{check_info=}")
                print()

        return check_result


class StepMixIn:

    @weave.op
    def on_enter_state(self, event, state):

        if self.model.attempts[state.id] > MAX_ATTEMPTS:
            raise AgentError(f"Reached max attempts on state {state.id} ({self.model.attempts[state.id]} > {MAX_ATTEMPTS})")

        self.model.attempts[state.id] += 1

        check_result = self.model.execute_step(self.model.specification[event]["instruction_fn"],
                                               self.model.specification[state.id]["processing_fn"],
                                               self.model.specification[state.id]["check_fn"])

        if state.final:
            if check_result:
                weave.publish(self.model.context, "context")
                return
            else:
                raise ValueError("Final state should be always successful")

        follow_event_name = f"{state.id}_{'success' if check_result else 'failed'}"
        follow_event = getattr(self, follow_event_name)
        follow_event()


class CasualModelingAgentStage(Enum):
    SCHEMA = 1
    ELICIT_CONSTRAINTS = 2
    ROOT_NODES = 3
    ROOT_TO_NON_EDGES = 4
    NON_TO_NON_EDGES = 5
    DAG = 6
    STRUCTURAL_EQUATIONS = 7
    PARAMETERS = 8
    PYRO_CODE = 9
    ENFORCE_RANGE = 10
    ENFORCE_CONSTRAINTS = 11
    FINITO = 12


class CasualModelingAgentMachine(StateMachine, StepMixIn):

    states = States.from_enum(
        CasualModelingAgentStage,
        initial=CasualModelingAgentStage.SCHEMA,
        final=CasualModelingAgentStage.FINITO,
        use_enum_instance=True,
    )

    # TODO: Refactor with dynamic generation
    SCHEMA_failed = states.SCHEMA.to.itself()
    SCHEMA_success = states.SCHEMA.to(states.ELICIT_CONSTRAINTS)

    ELICIT_CONSTRAINTS_failed = states.ELICIT_CONSTRAINTS.to.itself()
    ELICIT_CONSTRAINTS_success = states.ELICIT_CONSTRAINTS.to(states.ROOT_NODES)

    ROOT_NODES_failed = states.ROOT_NODES.to.itself()
    ROOT_NODES_success = states.ROOT_NODES.to(states.ROOT_TO_NON_EDGES)

    ROOT_TO_NON_EDGES_failed = states.ROOT_TO_NON_EDGES.to.itself()
    ROOT_TO_NON_EDGES_success = states.ROOT_TO_NON_EDGES.to(states.NON_TO_NON_EDGES)

    NON_TO_NON_EDGES_failed = states.NON_TO_NON_EDGES.to.itself()
    NON_TO_NON_EDGES_success = states.NON_TO_NON_EDGES.to(states.DAG)

    DAG_failed = states.DAG.to.itself()
    DAG_success = states.DAG.to(states.STRUCTURAL_EQUATIONS)

    STRUCTURAL_EQUATIONS_failed = states.STRUCTURAL_EQUATIONS.to.itself()
    STRUCTURAL_EQUATIONS_success = states.STRUCTURAL_EQUATIONS.to(states.PARAMETERS)

    PARAMETERS_failed = states.PARAMETERS.to.itself()
    PARAMETERS_success = states.PARAMETERS.to(states.PYRO_CODE)

    PYRO_CODE_failed = states.PYRO_CODE.to.itself()
    PYRO_CODE_success = states.PYRO_CODE.to(states.ENFORCE_RANGE)

    ENFORCE_RANGE_failed = states.ENFORCE_RANGE.to.itself()
    ENFORCE_RANGE_success = states.ENFORCE_RANGE.to(states.ENFORCE_CONSTRAINTS)

    ENFORCE_CONSTRAINTS_failed = states.ENFORCE_CONSTRAINTS.to.itself()
    ENFORCE_CONSTRAINTS_success = states.ENFORCE_CONSTRAINTS.to(states.FINITO)
