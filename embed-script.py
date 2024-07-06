#!/usr/bin/env python3

import voyageai
import glob
import json
from pathlib import Path

files = glob.glob("./content/*.md")
contents = [Path(x).read_text() for x in files]

vo = voyageai.Client()

print("Embedding...")
result = vo.embed(contents, model="voyage-multilingual-2", input_type="document")

print("Embedded!")
Path("embeddings.json").write_text(json.dumps(result.embeddings))
