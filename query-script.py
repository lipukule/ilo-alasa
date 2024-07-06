#!/usr/bin/env python3

import time

lap = time.perf_counter()

def mu(it):
	global lap
	old_lap = lap
	lap = time.perf_counter()
	print(f"{it}: {lap - old_lap:0.4f} seconds")

import numpy as np
import voyageai
import sys
import json
import glob
from pathlib import Path

mu("imports")

files = glob.glob("./content/*.md")
doc_embeddings = json.loads(Path("./embeddings.json").read_text())

mu("loaded")

vo = voyageai.Client()

total_tokens = vo.count_tokens([sys.argv[1]], model="voyage-multilingual-2")
print(f"total tokens: {total_tokens}")
query_embedding = vo.embed([sys.argv[1]], model="voyage-multilingual-2", input_type="query").embeddings[0]

mu("queried")

similarities = np.dot(doc_embeddings, query_embedding)
retrieved_id = np.argmax(similarities)

mu("similarities")

print(files[retrieved_id])
