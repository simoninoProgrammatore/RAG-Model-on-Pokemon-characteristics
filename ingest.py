from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document
import os
import re

def load_pokemon_documents(data_dir="data/"):
    """
    Carica i chunk granulari dai file .txt
    Ogni chunk ha formato: --- NomePokemon - Aspetto ---
    """
    documents = []
    
    for filename in os.listdir(data_dir):
        if filename.endswith(".txt"):
            filepath = os.path.join(data_dir, filename)
            with open(filepath, "r", encoding="utf-8") as f:
                content = f.read()
            
            # Split per separatore ---
            chunks = content.split("---")
            current_header = None
            
            for chunk in chunks:
                chunk = chunk.strip()
                if not chunk:
                    continue
                
                # I blocchi dispari sono gli header (Nome - Aspetto)
                if current_header is None:
                    current_header = chunk
                else:
                    # Estrai nome Pokemon e aspetto dall'header
                    # Formato: "Pikachu - Statistiche"
                    match = re.match(r"^(.+?)\s*-\s*(.+)$", current_header)
                    if match:
                        pokemon_name = match.group(1).strip()
                        aspect = match.group(2).strip()
                    else:
                        pokemon_name = current_header
                        aspect = "generale"
                    
                    doc = Document(
                        page_content=f"--- {current_header} ---\n{chunk}",
                        metadata={
                            "source": filename,
                            "pokemon": pokemon_name,
                            "aspect": aspect.lower()
                        }
                    )
                    documents.append(doc)
                    current_header = None

    return documents

print("Caricamento e parsing dei Pokemon (chunking granulare)...")
documents = load_pokemon_documents()
print(f"  → {len(documents)} chunk caricati")

# Mostra distribuzione per aspetto
aspects = {}
for doc in documents:
    aspect = doc.metadata.get("aspect", "?")
    aspects[aspect] = aspects.get(aspect, 0) + 1

print("\nDistribuzione chunk per aspetto:")
for aspect, count in sorted(aspects.items()):
    print(f"  - {aspect}: {count}")

print("\nGenerazione embedding e salvataggio...")
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)
vectorstore = FAISS.from_documents(documents, embeddings)
vectorstore.save_local("faiss_db")

print("Indicizzazione completata! Database salvato in faiss_db/")