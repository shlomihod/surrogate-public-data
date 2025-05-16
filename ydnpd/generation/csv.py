# https://cookbook.openai.com/examples/sdg1

from typing import Any, Dict
import pandas as pd
import pandera as pa
from io import StringIO
import math
from pprint import pformat

import weave

from ydnpd.datasets.loader import load_dataset
from ydnpd.generation.llm import create_llm
from ydnpd.generation.agent.errors import AgentError
from ydnpd.utils import metadata_to_pandera_schema

RANDOM_STATE = 42
DEFAULT_BATCH_SIZE = 200
DEFAULT_RETRY_FACTOR = 3


def _get_system_prompt(domain: str) -> str:
    """Get the system prompt for CSV generation."""
    return (
        f"You are an expert in {domain} who generates synthetic data that closely mirrors real-world {domain} data. "
        f"Your goal is to create data that would be indistinguishable from real {domain} records.\n\n"
        "Follow exactly these rules:\n"
        "1. Only output the CSV data with no additional text or explanations\n"
        "2. Always include a header row matching the schema exactly\n"
        "3. Strictly adhere to the provided schema's data types and possible values for all fields\n"
        "4. Use comma as the separator\n"
        "5. Ensure all values and relationships between fields are realistic and statistically plausible\n"
        "6. Generate diverse data while maintaining real-world patterns and constraints\n"
        "7. Include occasional edge cases at realistic frequencies\n"
    )


def _format_user_prompt(num_rows: int, schema: Dict[str, Any]) -> str:
    """Format the user prompt for CSV generation."""
    return (
        f"Generate {num_rows} rows of data with these fields:\n\n"
        f"{pformat(schema)}"
    )


def _parse_csv_to_dataframe(content: str) -> pd.DataFrame:
    """Parse CSV content into a DataFrame."""
    try:
        # Strip markdown if present
        if "```" in content:
            content = content.split("```")[1].strip()
            if content.startswith("csv"):
                content = content[3:].strip()

        return pd.read_csv(StringIO(content))
    except Exception as e:
        raise AgentError(f"Failed to parse CSV content: {str(e)}")


def _validate_single_record(row: pd.Series, schema: pa.DataFrameSchema) -> bool:
    """Validate a single record against the schema."""
    try:
        schema.validate(pd.DataFrame([row]))
        return True
    except pa.errors.SchemaError:
        return False


@weave.op
def generate_csv(
    dataset_name: str,
    llm_path: str,
    total_rows: None | int = None,
    batch_size: int = DEFAULT_BATCH_SIZE,
    temperature: float = 0.7,
    max_tokens: int = 8192,
    retry_factor: int = DEFAULT_RETRY_FACTOR,
    verbose: bool = False,
) -> pd.DataFrame:
    """
    Generate synthetic CSV data using structured prompts.

    Args:
        dataset_name: Name of the dataset to use for schema
        total_rows: Total number of valid rows to generate
        batch_size: Number of rows to request in each LLM call
        llm_path: Path to LLM model
        temperature: LLM temperature parameter
        max_tokens: Maximum tokens for LLM response
        retry_factor: Number of tries allowed per needed batch
        verbose: Whether to print progress information
    """
    family_name, _ = dataset_name.split("/")
    reference_df, raw_schema, domain = load_dataset(dataset_name)
    schema = metadata_to_pandera_schema(raw_schema, coerce=True)

    if total_rows is None:
        total_rows = len(reference_df)

    valid_records = []

    needed_batches = math.ceil(total_rows / batch_size)
    max_tries = needed_batches * retry_factor

    llm = create_llm(
        model_path=llm_path,
        temperature=temperature,
        max_tokens=max_tokens,
    )

    if verbose:
        print(f"\n### {family_name} @ {llm_path} ###")
        print(f"Starting generation of {total_rows} records")
        print(f"Strategy: {needed_batches} batches of {batch_size} rows")
        print(f"Maximum tries: {max_tries} ({retry_factor}x per batch)")

    for try_num in range(max_tries):
        if len(valid_records) >= total_rows:
            break

        if verbose:
            print(f"\n### {family_name} @ {llm_path} ###")
            print(f"Try {try_num + 1}/{max_tries}")
            print(f"Generating batch of {batch_size} rows ({len(valid_records)}/{total_rows} collected)")

        messages = [
            {"role": "system", "content": _get_system_prompt(domain)},
            {"role": "user", "content": _format_user_prompt(batch_size, raw_schema)}
        ]

        try:
            response = llm(messages)
            batch_df = _parse_csv_to_dataframe(response)

            if verbose:
                print(f"Received {len(batch_df)} rows, validating...")

            valid_in_batch = 0
            invalid_in_batch = 0

            # Validate records individually and keep the good ones
            for _, row in batch_df.iterrows():
                if _validate_single_record(row, schema):
                    valid_records.append(row)
                    valid_in_batch += 1
                else:
                    invalid_in_batch += 1

            if verbose:
                print(f"Batch results: {valid_in_batch} valid, {invalid_in_batch} invalid")
                print(f"Total valid records so far: {len(valid_records)}/{total_rows}")

        except AgentError as e:
            if verbose:
                print(f"Error in try {try_num + 1}: {str(e)}")
            if try_num == max_tries - 1 and not valid_records:
                raise AgentError(f"Failed to generate any valid records after {max_tries} tries: {str(e)}")
            continue

    # Convert valid records to DataFrame
    if not valid_records:
        raise AgentError(f"Could not generate any valid records after {max_tries} tries")

    result_df = pd.DataFrame(valid_records)
    result_df = schema.validate(result_df)

    if verbose:
        print(f"\nGeneration complete: got {len(result_df)} valid records")
        if len(result_df) < total_rows:
            print(f"Warning: Could only generate {len(result_df)} valid records out of {total_rows} requested")

    # Return exact number of rows if we have enough, otherwise return what we have
    if len(result_df) >= total_rows:
        return (result_df
                .sample(total_rows, random_state=RANDOM_STATE)
                .reset_index(drop=True))

    return result_df
