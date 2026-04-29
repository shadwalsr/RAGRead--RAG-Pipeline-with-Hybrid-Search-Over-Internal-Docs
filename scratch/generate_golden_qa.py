import pypdf
import json
import os
from google import genai
from dotenv import load_dotenv

load_dotenv()

def generate():
    # Step 1: Extract text from the PDF
    pdf_path = r'data\raw\shadwal singh (3).pdf'
    if not os.path.exists(pdf_path):
        print(f"Error: {pdf_path} not found.")
        return
        
    reader = pypdf.PdfReader(pdf_path)
    full_text = ''
    for page in reader.pages:
        full_text += page.extract_text() + '\n'

    # Step 2: Use LLM to generate 50 Q&A pairs
    client = genai.Client()
    prompt = f'''You are an expert at creating benchmarks for RAG systems.
Based on the following resume text, generate a "Golden Q&A Dataset" consisting of 50 question-answer pairs.

TYPES OF QUESTIONS TO INCLUDE:
1. Fact Retrieval (30): Direct questions about skills, dates, roles, and achievements.
2. Multi-Hop (10): Questions that require connecting two different parts of the resume (e.g., comparing two different jobs or connecting a skill to a specific project).
3. Negative Cases (10): Questions that the resume CANNOT answer or are ambiguous (e.g., favorite food, address, or non-existent degrees).

Respond strictly with a JSON array of objects:
[
  {{ "question": "...", "answer": "...", "type": "fact/multi-hop/negative" }}
]

RESUME TEXT:
{full_text}
'''

    print("Generating 50 Q&A pairs... this may take a moment.")
    try:
        response = client.models.generate_content(
            model='gemini-flash-latest',
            contents=prompt,
            config={'response_mime_type': 'application/json'}
        )

        # Save to file
        os.makedirs('data/eval', exist_ok=True)
        with open('data/eval/golden_qa.json', 'w') as f:
            f.write(response.text)

        print('SUCCESS: Created data/eval/golden_qa.json with 50 pairs.')
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    generate()
