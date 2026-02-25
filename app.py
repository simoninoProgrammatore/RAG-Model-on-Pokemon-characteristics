import streamlit as st
from langchain_community.vectorstores import FAISS
from langchain_community.retrievers import BM25Retriever
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_anthropic import ChatAnthropic
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.documents import Document
from dotenv import load_dotenv
import requests
import os
import time
import re

load_dotenv()

# ─────────────────────────────────────────────────────────────
# Configurazione pagina
# ─────────────────────────────────────────────────────────────
st.set_page_config(page_title="Pokédex AI", page_icon="🔴", layout="wide")
st.title("🔴 Pokédex AI")
st.caption("Fai qualsiasi domanda sui Pokemon!")

# ─────────────────────────────────────────────────────────────
# Smart Retriever (Entity first → fallback semantico)
# ─────────────────────────────────────────────────────────────
class SmartRetriever:
    def __init__(self, bm25_retriever, faiss_retriever, all_documents):
        self.bm25 = bm25_retriever
        self.faiss = faiss_retriever
        self.all_docs = all_documents

        # Indice nome → lista chunk
        self.name_index = {}
        for doc in all_documents:
            pokemon = doc.metadata.get("pokemon", "").lower()
            if pokemon:
                if pokemon not in self.name_index:
                    self.name_index[pokemon] = []
                self.name_index[pokemon].append(doc)

    def extract_pokemon_names(self, query: str):
        query_lower = query.lower()
        found = []

        for name in self.name_index.keys():
            pattern = r"\b" + re.escape(name) + r"\b"
            if re.search(pattern, query_lower):
                found.append(name)

        return found

    def _detect_aspect(self, query: str):
        query_lower = query.lower()
        
        if any(k in query_lower for k in ["stat", "hp", "attacco", "difesa", "velocità"]):
            return "statistiche"
        elif any(k in query_lower for k in ["tipo", "type"]):
            return "tipi"
        elif any(k in query_lower for k in ["abilità", "ability", "abilita"]):
            return "abilità"
        elif any(k in query_lower for k in ["mossa", "mosse", "move"]):
            return "mosse"
        
        return None

    def metadata_match(self, query: str):
        names = self.extract_pokemon_names(query)
        matched_docs = []
        
        for name in names:
            if name in self.name_index:
                docs = self.name_index[name]
                
                aspect_filter = self._detect_aspect(query)
                if aspect_filter:
                    docs = [d for d in docs if d.metadata.get("aspect") == aspect_filter]
                
                matched_docs.extend(docs)
        
        return matched_docs

    def semantic_search(self, query: str):
        keywords_multi = {
            "tutti", "quali", "confronta", "migliore",
            "peggiore", "più", "meno", "lista",
            "quanti", "elenco"
        }

        is_multi = any(k in query.lower() for k in keywords_multi)
        k = 15 if is_multi else 5

        self.bm25.k = k
        self.faiss.search_kwargs["k"] = k

        bm25_docs = self.bm25.invoke(query)
        faiss_docs = self.faiss.invoke(query)

        seen = set()
        merged = []

        for doc in bm25_docs + faiss_docs:
            pokemon = doc.metadata.get("pokemon", "?")
            aspect = doc.metadata.get("aspect", "?")
            key = f"{pokemon}_{aspect}"
            
            if key not in seen:
                seen.add(key)
                merged.append(doc)

        return merged

    def retrieve(self, query: str):
        matched = self.metadata_match(query)
        if matched:
            return matched
        return self.semantic_search(query)


# ─────────────────────────────────────────────────────────────
# Caricamento retriever
# ─────────────────────────────────────────────────────────────
@st.cache_resource
def load_retrievers():
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
    vectorstore = FAISS.load_local(
        "faiss_db",
        embeddings,
        allow_dangerous_deserialization=True
    )

    documents = []
    for filename in os.listdir("data/"):
        if filename.endswith(".txt"):
            with open(os.path.join("data/", filename), "r", encoding="utf-8") as f:
                content = f.read()

            chunks = content.split("---")
            current_header = None

            for chunk in chunks:
                chunk = chunk.strip()
                if not chunk:
                    continue

                if current_header is None:
                    current_header = chunk
                else:
                    # Estrai nome e aspetto
                    match = re.match(r"^(.+?)\s*-\s*(.+)$", current_header)
                    if match:
                        pokemon_name = match.group(1).strip()
                        aspect = match.group(2).strip()
                    else:
                        pokemon_name = current_header
                        aspect = "generale"
                    
                    documents.append(
                        Document(
                            page_content=f"--- {current_header} ---\n{chunk}",
                            metadata={
                                "source": filename,
                                "pokemon": pokemon_name,
                                "aspect": aspect.lower()
                            }
                        )
                    )
                    current_header = None

    bm25 = BM25Retriever.from_documents(documents, k=5)
    faiss = vectorstore.as_retriever(search_kwargs={"k": 5})

    return bm25, faiss, documents


@st.cache_resource
def load_smart_retriever():
    bm25, faiss, docs = load_retrievers()
    return SmartRetriever(bm25, faiss, docs)


@st.cache_resource
def load_llm():
    return ChatAnthropic(
        model="claude-haiku-4-5-20251001",
        max_tokens=512,
        temperature=0
    )


smart_retriever = load_smart_retriever()
llm = load_llm()

# ─────────────────────────────────────────────────────────────
# Retry per Overload 529
# ─────────────────────────────────────────────────────────────
def safe_invoke(chain, payload, retries=5):
    for attempt in range(retries):
        try:
            return chain.invoke(payload)
        except Exception as e:
            if "overloaded" in str(e).lower():
                time.sleep(2 ** attempt)
            else:
                raise e
    raise Exception("Anthropic overloaded dopo vari tentativi")


# ─────────────────────────────────────────────────────────────
# Fetch dati PokeAPI
# ─────────────────────────────────────────────────────────────
@st.cache_data
def fetch_pokemon_data(name: str):
    try:
        r = requests.get(
            f"https://pokeapi.co/api/v2/pokemon/{name.lower()}",
            timeout=5
        )
        if r.status_code == 200:
            return r.json()
    except:
        pass
    return None


# ─────────────────────────────────────────────────────────────
# Prompt
# ─────────────────────────────────────────────────────────────
prompt_template = PromptTemplate.from_template("""
Sei un esperto di Pokemon.

Rispondi basandoti SOLO sul contesto fornito.
Se la domanda richiede un confronto, organizza la risposta in modo chiaro.
Se non trovi la risposta nel contesto, dillo chiaramente.

Contesto:
{context}

Domanda: {question}

Risposta:
""")


def format_docs(docs):
    text = "\n\n".join(doc.page_content for doc in docs)
    return text[:8000]


# ─────────────────────────────────────────────────────────────
# Session state
# ─────────────────────────────────────────────────────────────
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "last_docs" not in st.session_state:
    st.session_state.last_docs = []
if "last_question" not in st.session_state:
    st.session_state.last_question = ""


# ─────────────────────────────────────────────────────────────
# Layout
# ─────────────────────────────────────────────────────────────
col_chat, col_sidebar = st.columns([2, 1])

with col_chat:
    for msg in st.session_state.chat_history:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    if question := st.chat_input("Chiedi qualcosa sui Pokemon..."):
        with st.chat_message("user"):
            st.markdown(question)

        st.session_state.chat_history.append({
            "role": "user",
            "content": question
        })

        docs = smart_retriever.retrieve(question)
        context = format_docs(docs)

        with st.chat_message("assistant"):
            with st.spinner("Sto cercando..."):
                chain = prompt_template | llm | StrOutputParser()
                response = safe_invoke(
                    chain,
                    {"context": context, "question": question}
                )
            st.markdown(response)

        st.session_state.chat_history.append({
            "role": "assistant",
            "content": response
        })

        st.session_state.last_docs = docs
        st.session_state.last_question = question
        st.rerun()


# ─────────────────────────────────────────────────────────────
# Sidebar
# ─────────────────────────────────────────────────────────────
with col_sidebar:
    docs = st.session_state.last_docs
    question = st.session_state.last_question

    if not docs:
        st.info("👈 Fai una domanda per vedere le info!")
    else:
        main_doc = docs[0]
        main_name = main_doc.metadata.get("pokemon", "")

        if main_name:
            data = fetch_pokemon_data(main_name)
            if data:
                st.subheader(f"#{data['id']} {data['name'].capitalize()}")

                img_url = data["sprites"]["other"]["official-artwork"]["front_default"]
                if img_url:
                    st.image(img_url, width=200)

                st.subheader("Statistiche")
                for stat in data["stats"]:
                    st.write(f"{stat['stat']['name']}: {stat['base_stat']}")
                    st.progress(min(stat["base_stat"] / 255, 1.0))

        st.subheader("📄 Fonti recuperate")
        for doc in docs:
            pokemon = doc.metadata.get("pokemon", "?")
            aspect = doc.metadata.get("aspect", "?")
            source = doc.metadata.get("source", "?")
            
            with st.expander(f"🔹 {pokemon} - {aspect.capitalize()} ({source})"):
                st.text(doc.page_content[:300] + "...")


    if st.session_state.chat_history:
        st.divider()
        if st.button("🗑️ Nuova chat", use_container_width=True):
            st.session_state.chat_history = []
            st.session_state.last_docs = []
            st.session_state.last_question = ""
            st.rerun()