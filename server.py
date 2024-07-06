#!/usr/bin/env python3

from aiohttp import web
import voyageai
import numpy as np
import json
from pathlib import Path
import glob
import heapq
from aiohttplimiter import default_keyfunc, Limiter

vo = voyageai.AsyncClient()
limiter = Limiter(keyfunc=default_keyfunc)

doc_embeddings = json.loads(Path("./embeddings.json").read_text())
basenames = json.loads(Path("./basenames.json").read_text())
contents = [Path(f"./content/{x}.md").read_text() for x in basenames]

@limiter.limit("10/minute")
async def search(request):
    query = request.rel_url.query['i']
    total_tokens = vo.count_tokens([query], model="voyage-multilingual-2")
    if total_tokens > 25:
        return web.Response(status=413)

    result = await vo.embed([query], model="voyage-multilingual-2", input_type="query")
    query_embedding = result.embeddings[0]

    similarities = np.dot(doc_embeddings, query_embedding)
    most_similar = heapq.nlargest(10, range(len(similarities)), similarities.take)
    most_similar_basenames = [basenames[idx] for idx in most_similar]

    total_tokens = vo.count_tokens([contents[idx] for idx in most_similar], model="rerank-lite-1")

    reranking = await vo.rerank(query, [contents[idx] for idx in most_similar], model="rerank-lite-1", top_k=5)

    return web.Response(text=json.dumps([most_similar_basenames[item.index] for item in reranking.results]))

app = web.Application()
app.add_routes([
    web.get('/alasa', search),
])

web.run_app(app, port=8127)
