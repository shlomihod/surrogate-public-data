import asyncio
import pandas as pd
from datetime import datetime
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor

import wandb

from ydnpd.generation.agent.core import CasualModelingAgentMachine, LLMSession
from ydnpd.generation.agent.specifications import SPECIFICATION_V0
from ydnpd.generation.agent.data_config import SCHEMA


wandb.init(project="llm_state_machine", name="LLM_Causal_Model_Agent")

# TODO: this doesn't seem to work well for some reason. i think there
# are too many areas where the generation can blcok

NUM_AGENTS = 5 
NUM_SAMPLES = 1000 
OUTPUT_DIR = Path("output_data") 
OUTPUT_DIR.mkdir(exist_ok=True)
TIMEOUT = 5 * 60

async def generate_samples(agent_id):
    llm_sess = LLMSession(
        specification=SPECIFICATION_V0,
        domain="demographic and census data",
        schema=SCHEMA,
        verbose=True
    )
    agent = CasualModelingAgentMachine(llm_sess)

    all_data = []
    for _ in range(NUM_SAMPLES):
        sample = llm_sess.context["model"]() 
        sample = {k: v.item() if hasattr(v, 'item') else v for k, v in sample.items()}
        all_data.append(sample)

    df = pd.DataFrame(all_data)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = OUTPUT_DIR / f"agent_{agent_id}_{timestamp}.csv"
    df.to_csv(filename, index=False)
    print(f"Agent {agent_id} saved data to {filename}")

async def generate_samples_with_timeout(agent_id):
    try:
        await asyncio.wait_for(generate_samples(agent_id), TIMEOUT)
    except asyncio.TimeoutError:
        print(f"Agent {agent_id} timed out after {TIMEOUT // 60} minutes")

async def main():
    with ThreadPoolExecutor(max_workers=NUM_AGENTS) as executor:
        loop = asyncio.get_event_loop()
        tasks = [
            loop.run_in_executor(executor, asyncio.run, generate_samples_with_timeout(agent_id))
            for agent_id in range(NUM_AGENTS)
        ]
        await asyncio.gather(*tasks)

asyncio.run(main())