from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document
import os

def load_pokemon_documents(data_dir="data/"):
    documents = []
    for filename in os.listdir(data_dir):
        if filename.endswith(".txt"):
            filepath = os.path.join(data_dir, filename)
            with open(filepath, "r", encoding="utf-8") as f:
                content = f.read()
            
            # Dividi per separatore --- Nome ---
            chunks = content.split("---")
            current_name = None
            for chunk in chunks:
                chunk = chunk.strip()
                if not chunk:
                    continue
                # I blocchi dispari sono i nomi, i pari il contenuto
                if current_name is None:
                    current_name = chunk
                else:
                    doc = Document(
                        page_content=f"--- {current_name} ---\n{chunk}",
                        metadata={"source": filename, "pokemon": current_name}
                    )
                    documents.append(doc)
                    current_name = None

    return documents

print("Caricamento e parsing dei Pokemon...")
documents = load_pokemon_documents()
print(f"  → {len(documents)} Pokemon caricati")

print("Generazione embedding e salvataggio...")
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)
vectorstore = FAISS.from_documents(documents, embeddings)
vectorstore.save_local("faiss_db")

print("Indicizzazione completata! Database salvato in faiss_db/")