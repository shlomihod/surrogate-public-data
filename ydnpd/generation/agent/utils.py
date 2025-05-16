import linecache
from io import StringIO
import sys
from multiprocessing import Process, Queue
from typing import Any, Optional
import time

import torch
import pyro
import pyro.distributions as dist
import networkx as nx
import pandas as pd
import pandera as pa

from ydnpd.datasets import load_dataset
from ydnpd.utils import metadata_to_pandera_schema
from ydnpd.generation.agent.errors import AgentError


MAX_TIMEOUT_SAMPLING = 10
MAX_SAMPLING_CHECKS = 10
PRODUCTION_RANDOM_STATE = 42


def extract_single(answer):
    if len(answer) != 1:
        raise ValueError()
    return answer[0]


def clean_split(s, sep=","):
    try:
        return [item.strip() for item in s.split(sep)]
    except Exception:
        raise AgentError("Splitting answer into componenets failed.")


def build_graph(relationships):
    G = nx.DiGraph()
    try:
        for relationship in relationships:
            source, target = map(str.strip, relationship.split("->"))
            G.add_edge(source, target)

    except Exception as err:
        raise AgentError(f"Graph building based on relationships failed: {str(err)}")
    return G


def sample_dataset(model, num_samples, pandera_schema, code=None):
    records = []
    for _ in range(num_samples):
        try:
            sample = model()
        except Exception as err:
            err_str = format_exec_error(code, err) if code is not None else str(err)
            raise AgentError(f"Model execution failed:\n{err_str}")
        if sample is None:
            raise AgentError("Model sampled a None value")
        elif not isinstance(sample, dict):
            raise AgentError("Model should return a dictionary")
        elif any(none_vars := [key for key, value in sample.items() if value is None]):
            raise AgentError(f"Model sampled a None value in var(s): {none_vars}")

        record = {}
        for key, value in sample.items():
            if isinstance(value, (int, float)):
                record[key] = value
            elif isinstance(value, torch.Tensor):
                if value.dim() == 0:  # Check if tensor is a scalar
                    record[key] = value.item()
                else:
                    raise AgentError(f"Model sampled a tensor for variable '{key}' which not a scalar (shape: {value.shape})")
            else:
                raise AgentError(f"Model sampled a value for variable '{key}' is neither numeric nor a PyTorch tensor")

        records.append(record)

    df = pd.DataFrame(records)

    if pandera_schema is not None:
        try:
            pandera_schema.validate(df)
        except pa.errors.SchemaError as err:
            raise AgentError(f"Schema validation of sampled dataset failed: {str(err)}")

    return df


def format_exec_error(code_str, error):
    """
    Format error information for code executed via exec(), showing exact line numbers and context.

    Args:
        code_str (str): The source code string that was executed
        error (Exception): The exception that was caught

    Returns:
        str: Formatted error message with line numbers and context
    """

    # Create a custom string source for linecache
    error_lines = code_str.splitlines()
    source_name = "<string>"  # This matches exec's internal name

    # Add the code to linecache
    linecache.cache[source_name] = (
        len(code_str),
        None,
        error_lines,
        source_name
    )

    # Get the full traceback
    exc_type, exc_value, exc_traceback = sys.exc_info()

    try:
        # Format the error message
        error_msg = StringIO()
        error_msg.write(f"Error Type: {exc_type.__name__}\n")
        error_msg.write(f"Error Message: {str(exc_value)}\n\n")

        # Extract traceback for the exec'd code
        tb = exc_traceback
        while tb and tb.tb_frame.f_code.co_filename != "<string>":
            tb = tb.tb_next

        if tb:
            error_line_no = tb.tb_lineno

            # Show the problematic line and its context
            context_range = 2
            start_line = max(error_line_no - context_range, 1)
            end_line = min(error_line_no + context_range, len(error_lines))

            error_msg.write("Code context:\n")
            for i in range(start_line - 1, end_line):
                line_marker = ">" if i + 1 == error_line_no else " "
                line_number = f"{i + 1:4d}"
                code_line = error_lines[i]
                error_msg.write(f"{line_marker} {line_number}| {code_line}\n")

            error_msg.write("\n")

        return error_msg.getvalue()

    finally:
        # Clean up linecache
        del linecache.cache[source_name]


def retrieve_pyro_model(pyro_code):

    if pyro_code is None:
        raise AgentError("Code could not be extracted. Make sure that the code is within the tags <Answer>...</Answer>")

    local_dict = {}

    local_dict.update({
        'pyro': pyro,
        'dist': dist,
        'torch': torch
    })

    try:
        exec(pyro_code, globals(), local_dict)
    except Exception as err:
        raise AgentError(f"Model compilation failed:\n{format_exec_error(pyro_code, err)}")

    try:
        model = local_dict['model']
    except KeyError:
        raise AgentError("`model` function was not found in the code")

    pyro.clear_param_store()
    # model_trace = pyro.poutine.trace(model).get_trace()

    return model


def _run_pyro_model_worker(queue: Queue, pyro_code: str, max_attempts: int, pandera_schema) -> None:
    """Worker function that creates and runs Pyro model in the subprocess."""
    model = retrieve_pyro_model(pyro_code)
    try:
        result = sample_dataset(model, max_attempts, pandera_schema, pyro_code)
        queue.put((True, result))
    except AgentError as err:
        queue.put((False, err))


def run_pyro_model_with_timeout(
    pyro_code: str,
    max_attempts: int,
    timeout: float,
    pandera_schema,
) -> tuple[bool, Any]:
    """
    Run Pyro model in a separate process with timeout.

    Args:
        pyro_code: The Pyro code to run
        timeout: Timeout in seconds

    Returns:
        Tuple of (success: bool, result_or_error: Any)
    """
    queue = Queue()
    process = Process(
        target=_run_pyro_model_worker,
        args=(queue, pyro_code, max_attempts, pandera_schema)
    )

    try:
        process.start()
        start_time = time.monotonic()

        while time.monotonic() - start_time < timeout:
            if not process.is_alive():
                break
            time.sleep(0.1)

        if not queue.empty():
            status, result = queue.get_nowait()
            if status:
                return result
            else:
                raise result

        raise AgentError(f"Model execution timed out after {timeout} seconds")

    finally:
        if process.is_alive():
            process.terminate()
            process.join(timeout=0.5)
            if process.is_alive():
                process.kill()
        process.close()


def is_valid_pyro_code(
    pyro_code: str, 
    pandera_schema: Optional[dict] = None,
    max_attempts: int = MAX_SAMPLING_CHECKS,
    sampling_timeout: int = MAX_TIMEOUT_SAMPLING,
) -> Optional[str]:
    """
    Validate Pyro code with process-based timeout handling.
    """

    retrieve_pyro_model(pyro_code)
    return run_pyro_model_with_timeout(pyro_code, max_attempts, sampling_timeout, pandera_schema)


def produce_dataset(metadata, specification, num_samples, **llm_kwargs):

    from ydnpd.generation.agent.core import CasualModelingAgentMachine, LLMSession

    try:

        llm_sess = LLMSession(
            specification=specification,
            metadata=metadata,
            **llm_kwargs)

        _ = CasualModelingAgentMachine(llm_sess)

        pandera_schema = metadata_to_pandera_schema(metadata["schema"])

        df = sample_dataset(llm_sess.context["model"], num_samples, pandera_schema)

        code = llm_sess.context["code"]

        error = None

    except AgentError as e:
        df, code, error = None, None, e

    return df, code, error


def produce_datasets(dataset_name,
                     specification,
                     num_datasets,
                     num_samples=None,
                     **llm_kwargs):
    dfs = []
    codes = []
    errors = []

    reference_df, schema, domain = load_dataset(dataset_name)

    medatadata = {"schema": schema, "domain": domain}

    if num_samples is None:
        num_samples = len(reference_df)

    while len(dfs) < num_datasets:
        print(len(dfs))

        df, code, error = produce_dataset(medatadata,
                                          specification,
                                          num_samples,
                                          **llm_kwargs)
        if error is None:
            dfs.append(df)
            codes.append(code)
        else:
            errors.append(error)
            print(error)

    return dfs, codes, errors


def mix_datasets(dfs, num_samples,
                 random_state=PRODUCTION_RANDOM_STATE):
    return (pd.concat(dfs)
            .sample(num_samples, replace=False, random_state=random_state)
            .reset_index(drop=True))


def produce_mixture_dataset(dataset_name, specification,
                            num_samples, num_datasets,
                            random_state=PRODUCTION_RANDOM_STATE,
                            **llm_kwargs):

    dfs, codes, errors = produce_datasets(dataset_name, specification,
                                          num_samples, num_datasets,
                                          **llm_kwargs)

    mixture_df = (dfs, num_samples, random_state)

    return mixture_df, (dfs, codes, errors)