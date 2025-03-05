import os
from dotenv import load_dotenv
from openai import OpenAI
import chromadb
from chromadb.utils import embedding_functions

load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")

openai_ef = embedding_functions.OpenAIEmbeddingFunction(
    api_key=openai_api_key,
    model_name="text-embedding-3-small"
)

chroma_client = chromadb.PersistentClient(path="chroma_db")
collection_name = "document_qa_collection"
collection = chroma_client.get_or_create_collection(
    name=collection_name, 
    embedding_function=openai_ef
)

client = OpenAI(api_key=openai_api_key)

response = client.chat.completions.create(
    model="gpt-3.5-turbo",
    messages=[
        {"role": "system", "content": "You are a helpful assistant that can answer questions DevOps."},
        {"role": "user", "content": "What is the main workflow of DevOps?"} 
    ]
)

def load_documents_from_directory(directory_path):
    print("==== Loading documents from directory ====")
    documents = []
    for filename in os.listdir(directory_path):
        if filename.endswith(".txt"):
            with open(
                os.path.join(directory_path, filename), "r", encoding="utf-8"
            ) as file:
                documents.append({"id": filename, "text": file.read()})
    return documents

def split_text(text, chunk_size=1000, chunk_overlap=200):
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start = end - chunk_overlap
    return chunks

directory_path = "./news_articles"
documents = load_documents_from_directory(directory_path)

print(f"Loaded {len(documents)} documents")

chunked_documents = []
for doc in documents:
    chunks = split_text(doc["text"])
    print("*** Splitting docs into chunks ***")
    for i, chunk in enumerate(chunks):
        chunked_documents.append({"id": f"{doc['id']}_chunk{i+1}", "text": chunk})


# print(f"Chunked {len(chunked_documents)} chunks")

def get_openai_embeddings(text):
    response = client.embeddings.create(
        model="text-embedding-3-small",
        input=text
    )
    embedding = response.data[0].embedding
    print("*** Generating embeddings ***")
    return embedding

for doc in chunked_documents:
    print(f"Processing document: {doc['id']}")
    embedding = get_openai_embeddings(doc["text"])
    doc["embedding"] = embedding

#print(doc["embedding"])

for doc in chunked_documents:
    print("*** Inserting documents into Chroma DB ***")
    collection.upsert(
        ids=[doc["id"]],
        documents=[doc["text"]],
        embeddings=[doc["embedding"]]
    )

def query_documents(question, n_results=2):
    results = collection.query(
        query_texts=question,
        n_results=n_results
    )
    relevant_chunks = [doc for sublist in results["documents"] for doc in sublist] 
    print("*** Returning Relevant chunks ***")
    return relevant_chunks

def generate_answer(question, relevant_chunks):
    context = "\n\n".join(relevant_chunks)
    prompt = (
        "You are an assistant for question-answering tasks. Use the following pieces of "
        "retrieved context to answer the question. If you don't know the answer, say that you "
        "don't know. Use three sentences maximum and keep the answer concise."
        "\n\nContext:\n" + context + "\n\nQuestion:\n" + question
    )

    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {
                "role": "system",
                "content": prompt,
            },
            {
                "role": "user",
                "content": question,
            },
        ],
    )
    answer = response.choices[0].message.content
    return answer













