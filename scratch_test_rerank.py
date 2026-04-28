import sys; sys.path.insert(0,'src')
from dotenv import load_dotenv; load_dotenv()
from retriever import HybridRetriever
import os, json
from google import genai
from google.genai import types

r = HybridRetriever()
query = 'Whyschool'
fused = r.retrieve(query, top_k=20)

client = genai.Client()

# Build prompt for LLM reranker
chunks_text = ''
for i, chunk in enumerate(fused):
    chunks_text += f'--- Chunk ID: {chunk["id"]} ---\n{chunk["content"]}\n\n'

prompt = f'''You are a relevance ranking engine.
Query: "{query}"

Evaluate the following document chunks and rank them by how relevant they are to the query.
Return the result as a JSON array of chunk IDs, ordered from most relevant to least relevant.
Max 5 chunk IDs. Only include IDs of chunks that are actually relevant.

Chunks:
{chunks_text}
'''

response = client.models.generate_content(
    model='gemini-2.5-flash',
    contents=prompt,
    config=types.GenerateContentConfig(
        response_mime_type='application/json',
    )
)

print('LLM Reranked IDs:', response.text)
