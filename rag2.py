import os
import json
import time
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from sentence_transformers import SentenceTransformer
from sklearn.neighbors import NearestNeighbors
from transformers import GPTNeoXJapaneseForCausalLM, GPTNeoXJapaneseTokenizer
import faiss
import torch
from fastapi import FastAPI, Request 
import requests
import uvicorn
from fastapi.middleware.cors import CORSMiddleware
from starlette.middleware.sessions import SessionMiddleware as StarletteSessionMiddleware
#from starlette.requests import Request as StarletteRequest
import tiktoken



device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
remote_model_url = "http://219.117.47.40:8000/v1/chat/completions"

tokenizer = tiktoken.get_encoding("cl100k_base") 
tokenizer = tiktoken.encoding_for_model("gpt-3.5-turbo")


file_path = '/home/orgil/ai-lab/research/output.txt'
texts = []
with open(file_path, 'r', encoding='utf-8') as f:
    text = f.read().strip()
    if text:
        texts.append(text)

def chunk_document_japanese(text, chunk_size=256):
    input_ids = tokenizer.encode(text)
    chunks = []
    for i in range(0, len(input_ids), chunk_size):
        chunk_ids = input_ids[i:i + chunk_size]     
        chunk_text = tokenizer.decode(chunk_ids)         
        chunks.append(chunk_text)
    return chunks

document_chunks = []

for text in texts:
    document_chunks.extend(chunk_document_japanese(text))

#for i, chunk in enumerate(document_chunks):
    #print(f"Chunk {i}: {chunk}")

embedder = SentenceTransformer("BAAI/bge-m3").to(device)
#embedder = SentenceTransformer("pkshatech/GLuCoSE-base-ja").to(device)
#embedder = SentenceTransformer("colorfulscoop/sbert-base-ja").to(device)
#embedder = SentenceTransformer("distiluse-base-multilingual-cased-v2").to(device)
#embedder = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2").to(device)
#embedder = SentenceTransformer("paraphrase-multilingual-mpnet-base-v2").to(device)
#embedder = SentenceTransformer("sonoisa/sentence-bert-base-ja-mean-tokens-v2").to(device)
embeddings = embedder.encode(document_chunks, convert_to_tensor=True)
embeddings = embeddings.cpu().numpy()

print(f"Number of chunks: {len(document_chunks)}")
print(f"Number of embeddings: {embeddings.shape[0]}")


#neighbors = NearestNeighbors(n_neighbors=5, metric='cosine').fit(embeddings)

index = faiss.IndexFlatL2(embeddings.shape[1])
index.add(embeddings)

print(f"FAISS index size: {index.ntotal}")

system_message = (
    "あなたは「AIくん」というチャットボットアシスタントで、「神明工業"
    "という会社の従業員にサービスを提供しています"
    "あなたは以下の文書に基づいて正確な情報を提供します。"
    "従業員が文書に記載されていない情報を尋ねた場合は、"
    "正直に誠実に知らないと答えます。常に人々を尊重し、"
    "質問にはできるだけ詳しく答えてください。"
    "これがあなたのナレッジベースです。この「QUERY」に答えてください。"
)

#generator = pipeline("text-generation", model=model, tokenizer=tokenizer, device=0 if torch.cuda.is_available() else -1)

def retrieve_and_generate(query, conversation_history=None, past_contexts=None, max_history_length=8):
    start_time = time.time()
    
    if past_contexts is None:
        past_contexts = set()
        
    if conversation_history is None:
        conversation_history = []

    query_embedding = embedder.encode([query])
    #distances, indices = neighbors.kneighbors(query_embedding, n_neighbors=5)
    distances, indices = index.search(query_embedding, k=10)
    
    print(f"Distances: {distances}")
    print(f"Indices: {indices}")

    valid_results = [(idx, dist) for idx, dist in zip(indices[0], distances[0]) if idx != -1]

    print(f"Valid Distances: {[dist for _, dist in valid_results]}")
    print(f"Valid Indices: {[idx for idx, _ in valid_results]}")

    ranked_chunks = sorted(valid_results, key=lambda x: x[1])
    
    retrieved_chunks = []
    total_tokens = 0
    max_retrieved_chunks = 10
    for idx, distance in ranked_chunks[:max_retrieved_chunks]:
        if idx not in past_contexts:
            chunk = document_chunks[idx]
            
            chunk_tokens = len(tokenizer.encode(chunk))
            retrieved_chunks.append((chunk, distance))
            total_tokens += chunk_tokens
            past_contexts.add(idx)
        
            print(f"\n=== Retrieved Chunk {len(retrieved_chunks)} (Distance: {distance}) ===")
            print(chunk)
    
    context = system_message + '\n'.join(chunk for chunk, _ in retrieved_chunks)
    
    conversation_history.append(f"Query: {query}\n")
    context_with_history = context + '\n'.join(conversation_history)
    
    print("\n===== Model Input =====")
    print(context_with_history)
    print("=======================+++++++++++++===========================\n")

    payload = {
        "model": "Qwen/Qwen2-72B-Instruct", 
        "messages": [
            {"role": "system", "content": system_message},
            {"role": "user", "content": context_with_history}
        ],
        "max_tokens": 8192,
        "temperature": 0,
    }

    response = requests.post(remote_model_url, json=payload)
    response.raise_for_status()
    
    generated_text = response.json()['choices'][0]['message']['content']
    
    """generated_outputs = generator(context_with_history, max_new_tokens=1500, no_repeat_ngram_size=2, early_stopping=True)

    generated_text = generated_outputs[0]['generated_text']"""
    
    #input_text = context_with_history
    
    #input_ids = tokenizer.encode(context_with_history, return_tensors='pt').to(device)

    #input_tokens = tokenizer(context_with_history, truncation=True, max_length=4096, return_tensors='pt').to(device)
    
    #with torch.no_grad():
        #outputs = model.generate(input_tokens['input_ids'], max_new_tokens=1500, no_repeat_ngram_size=2, early_stopping=True)
        #outputs = model.generate(input_text, max_new_tokens=1500, no_repeat_ngram_size=2, early_stopping=True)
        
    #generated_text = outputs[0]
    #generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

    answer_start = generated_text.find(query) + len(query)
    answer = generated_text[answer_start:].strip()

    #answer = tokenizer.decode(outputs[0], skip_special_tokens=True)
    answer = answer.replace(system_message, "").strip()
    for chunk, _ in retrieved_chunks:
        answer = answer.replace(chunk, "").strip()
    
    conversation_history.append(f"AI: {answer}\n")

    conversation_tokens = tokenizer.encode(' '.join(conversation_history))
    if len(conversation_tokens) > 10000:
        while len(conversation_tokens) > 10000:
            conversation_history.pop(0)
            conversation_tokens = tokenizer.encode(' '.join(conversation_history))  

    torch.cuda.empty_cache()
    
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Time taken to get an answer: {elapsed_time:.2f} seconds")
    return context, query, answer, conversation_history

conversation_history = []
query = "長男の妻の妹のお父さんがなくなったんですが、弔休は何日ですか？"

context, query, answer, conversation_history = retrieve_and_generate(query, conversation_history)
print("=======================++++++ANSWER+++++++===========================\n")
print(answer)

"""app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.add_middleware(StarletteSessionMiddleware, secret_key = "supersecretkey")

@app.post("/query")
async def query(request: Request):
    try:
        data = await request.json()
        query_text = data.get("query", "")
        session = request.session
        conversation_history = session.get("conversation_history", [])
        context, query, answer, conversation_history = retrieve_and_generate(query_text, conversation_history)
        session["conversation_history"] = conversation_history
        return {"context": context, "query": query, "answer": answer}
    except Exception as e:
        return {"error": str(e), "message": "failed to process the query"}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)"""
