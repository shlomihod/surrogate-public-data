import pandas as pd
import time
from datetime import datetime
from pathlib import Path

from ydnpd.generation.agent.core import CasualModelingAgentMachine, LLMSession
from ydnpd.generation.agent.specifications import SPECIFICATION_V0
from ydnpd.generation.agent.data_config import SCHEMA

import wandb

wandb.init(project="llm_state_machine", name="LLM_Causal_Model_Agent")

NUM_AGENTS = 5
NUM_SAMPLES = 1000
OUTPUT_DIR = Path("output_data")
OUTPUT_DIR.mkdir(exist_ok=True)
TIMEOUT = 5 * 60  # seconds

def generate_samples(agent_id):
    start_time = time.time()
    llm_sess = LLMSession(
        specification=SPECIFICATION_V0,
        domain="demographic and census data",
        schema=SCHEMA,
        verbose=True
    )
    agent = CasualModelingAgentMachine(llm_sess)

    all_data = []
    for _ in range(NUM_SAMPLES):
        elapsed_time = time.time() - start_time
        if elapsed_time > TIMEOUT:
            print(f"Agent {agent_id} timed out after {TIMEOUT // 60} minutes")
            break

        sample = llm_sess.context["model"]()
        sample = {k: v.item() if hasattr(v, 'item') else v for k, v in sample.items()}
        all_data.append(sample)

    if all_data:
        df = pd.DataFrame(all_data)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = OUTPUT_DIR / f"agent_{agent_id}_{timestamp}.csv"
        df.to_csv(filename, index=False)
        print(f"Agent {agent_id} saved data to {filename}")

def main():
    for agent_id in range(NUM_AGENTS):
        generate_samples(agent_id)

if __name__ == "__main__":
    main()
