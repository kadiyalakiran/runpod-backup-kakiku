import json
import os
from archon.completions import Archon

os.environ.setdefault("OPENAI_BASE_URL", "http://localhost:8000/v1")
os.environ.setdefault("OPENAI_API_KEY", "dummy")

with open("configs/gen_rank_local.json") as f:
    config = json.load(f)

archon = Archon(config, api_key_data={"OPENAI_API_KEY": [os.environ["OPENAI_API_KEY"]]})
result = archon.generate("Explain KV caching in two sentences.")
print(result)
