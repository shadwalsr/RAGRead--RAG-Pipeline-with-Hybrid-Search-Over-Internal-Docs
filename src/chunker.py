import hashlib
import re
from google import genai
import os

def generate_id(text):
    # stable ID based on content so we don't index the same chunk twice
    return hashlib.md5(text.encode()).hexdigest()

def split_text(full_text, source_name, strategy="structure_aware"):
    # main entry point for splitting a doc into pieces
    # three modes: fixed (dumb), structural (regex-based), semantic (AI-based)
    
    if strategy == "fixed_size":
        return _fixed_split(full_text, source_name)
    elif strategy == "semantic":
        return _semantic_split(full_text, source_name)
    else:
        # default to structural since it's the most reliable for resumes/docs
        return _structural_split(full_text, source_name)

def _fixed_split(text, src, size=500, overlap=50):
    chunks = []
    # very basic sliding window
    for i in range(0, len(text), size - overlap):
        chunk_content = text[i:i + size]
        chunks.append({
            "id": f"{src}_fixed_{i}",
            "content": chunk_content,
            "metadata": {"source": src, "chunking_strategy": "fixed"}
        })
    return chunks

def _structural_split(text, src):
    # split by natural boundaries like newlines or double-newlines
    # good for keeping sections together
    sections = re.split(r'\n\n+', text)
    chunks = []
    for i, section in enumerate(sections):
        if not section.strip(): continue
        chunks.append({
            "id": f"{src}_struct_{i}",
            "content": section.strip(),
            "metadata": {"source": src, "chunking_strategy": "structural"}
        })
    return chunks

def _semantic_split(text, src):
    # using the LLM to find the "best" break points
    # warning: this is slow and uses tokens, but very high quality
    api_key = os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        print("no API key for semantic chunking - falling back to structural")
        return _structural_split(text, src)
        
    client = genai.Client(api_key=api_key)
    
    # we ask the AI to mark where the topics change
    prompt = f"""Break the following text into distinct topical sections.
Use the marker '---SECTION---' to separate them.
Do not change the text itself.

TEXT:
{text[:4000]} # capping to 4k chars to stay safe with context limits
"""
    try:
        resp = client.models.generate_content(model="gemini-flash-latest", contents=prompt)
        sections = resp.text.split("---SECTION---")
        
        chunks = []
        for i, s in enumerate(sections):
            content = s.strip()
            if not content: continue
            chunks.append({
                "id": f"{src}_semantic_{i}",
                "content": content,
                "metadata": {"source": src, "chunking_strategy": "semantic"}
            })
        return chunks
    except Exception as e:
        print(f"semantic split failed: {e}")
        return _structural_split(text, src)
