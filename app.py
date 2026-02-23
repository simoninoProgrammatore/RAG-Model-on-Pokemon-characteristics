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

load_dotenv()

# ── Configurazione pagina ──────────────────────────────────────────────────────
st.set_page_config(page_title="Pokédex AI", page_icon="🔴", layout="wide")
st.title("🔴 Pokédex AI")
st.caption("Fai qualsiasi domanda sui Pokemon!")

# ── Caricamento modelli (cached) ───────────────────────────────────────────────
@st.cache_resource
def load_retrievers():
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vectorstore = FAISS.load_local("faiss_db", embeddings, allow_dangerous_deserialization=True)

    documents = []
    for filename in os.listdir("data/"):
        if filename.endswith(".txt"):
            with open(os.path.join("data/", filename), "r", encoding="utf-8") as f:
                content = f.read()
            chunks = content.split("---")
            current_name = None
            for chunk in chunks:
                chunk = chunk.strip()
                if not chunk:
                    continue
                if current_name is None:
                    current_name = chunk
                else:
                    documents.append(Document(
                        page_content=f"--- {current_name} ---\n{chunk}",
                        metadata={"source": filename, "pokemon": current_name}
                    ))
                    current_name = None

    bm25 = BM25Retriever.from_documents(documents, k=5)
    faiss = vectorstore.as_retriever(search_kwargs={"k": 5})
    return bm25, faiss

@st.cache_resource
def load_llm():
    return ChatAnthropic(model="claude-haiku-4-5-20251001", max_tokens=2048)

bm25_retriever, faiss_retriever = load_retrievers()
llm = load_llm()

# ── Funzioni retrieval ─────────────────────────────────────────────────────────
STOPWORDS = {
    "di", "che", "tipo", "è", "qual", "quale", "mi", "parlami", "dimmi",
    "cosa", "come", "ha", "le", "la", "lo", "il", "i", "gli", "un", "uno",
    "una", "del", "della", "dello", "allora", "quali", "sono", "mosse",
    "abilità", "statistiche", "stats", "vorrei", "sapere", "puoi", "dirmi",
    "quante", "quanto", "suoi"
}

def enhance_query(query: str) -> str:
    tokens = query.lower().split()
    keywords = [t for t in tokens if t not in STOPWORDS]
    return query + " " + " ".join(keywords)

def hybrid_retrieve(query: str) -> list[Document]:
    keywords_multi = {"tutti", "quali", "confronta", "migliore", "peggiore",
                      "più", "meno", "lista", "quanti", "elenco"}
    is_multi = any(k in query.lower() for k in keywords_multi)
    k = 20 if is_multi else 3

    bm25_retriever.k = k
    faiss_retriever.search_kwargs["k"] = k

    bm25_docs = bm25_retriever.invoke(enhance_query(query))
    faiss_docs = faiss_retriever.invoke(query)

    seen, merged = set(), []
    for doc in bm25_docs + faiss_docs:
        key = doc.metadata.get("pokemon", doc.page_content[:50])
        if key not in seen:
            seen.add(key)
            merged.append(doc)
    return merged

def extract_main_pokemon(query: str, docs: list[Document]) -> str:
    query_lower = query.lower()
    for doc in docs:
        name = doc.metadata.get("pokemon", "").strip().lower()
        if name and name in query_lower:
            return name
    if docs:
        return docs[0].metadata.get("pokemon", "").strip()
    return ""

# ── Fetch dati da PokeAPI ──────────────────────────────────────────────────────
@st.cache_data
def fetch_pokemon_data(name: str):
    try:
        r = requests.get(f"https://pokeapi.co/api/v2/pokemon/{name.lower()}", timeout=5)
        if r.status_code == 200:
            return r.json()
    except:
        pass
    return None

# ── Prompt ─────────────────────────────────────────────────────────────────────
prompt_template = PromptTemplate.from_template("""Sei un esperto di Pokemon. Rispondi alla domanda
basandoti SOLO sul contesto fornito. Se non trovi la risposta nel contesto, dillo chiaramente.

Contesto:
{context}

Domanda: {question}

Risposta:""")

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

# ── Session state ──────────────────────────────────────────────────────────────
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "last_docs" not in st.session_state:
    st.session_state.last_docs = []
if "last_question" not in st.session_state:
    st.session_state.last_question = ""

# ── Layout ─────────────────────────────────────────────────────────────────────
col_chat, col_sidebar = st.columns([2, 1])

with col_chat:
    for msg in st.session_state.chat_history:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    if question := st.chat_input("Chiedi qualcosa sui Pokemon..."):
        with st.chat_message("user"):
            st.markdown(question)
        st.session_state.chat_history.append({"role": "user", "content": question})

        docs = hybrid_retrieve(question)
        context = format_docs(docs)

        with st.chat_message("assistant"):
            with st.spinner("Sto cercando..."):
                chain = prompt_template | llm | StrOutputParser()
                response = chain.invoke({"context": context, "question": question})
            st.markdown(response)

        st.session_state.chat_history.append({"role": "assistant", "content": response})
        st.session_state.last_docs = docs
        st.session_state.last_question = question
        st.rerun()

with col_sidebar:
    docs = st.session_state.last_docs
    question = st.session_state.last_question

    if not docs:
        st.info("👈 Fai una domanda per vedere le informazioni sul Pokemon!")
    else:
        # Immagine e stats del Pokemon principale
        main_name = extract_main_pokemon(question, docs)
        if main_name:
            data = fetch_pokemon_data(main_name)
            if data:
                st.subheader(f"#{data['id']} {data['name'].capitalize()}")

                img_url = data["sprites"]["other"]["official-artwork"]["front_default"]
                if img_url:
                    st.image(img_url, width=200)

                types = [t["type"]["name"].capitalize() for t in data["types"]]
                type_colors = {
                    "Fire": "🔴", "Water": "🔵", "Grass": "🟢", "Electric": "🟡",
                    "Psychic": "🟣", "Ice": "🩵", "Dragon": "🟠", "Dark": "⚫",
                    "Fairy": "🩷", "Normal": "⚪", "Fighting": "🟤", "Flying": "🌀",
                    "Poison": "💜", "Ground": "🟫", "Rock": "🪨", "Bug": "🐛",
                    "Ghost": "👻", "Steel": "⚙️"
                }
                st.write(" ".join([f"{type_colors.get(t, '❓')} {t}" for t in types]))

                st.subheader("Statistiche")
                stat_names = {
                    "hp": "HP", "attack": "Attacco", "defense": "Difesa",
                    "special-attack": "Att. Sp.", "special-defense": "Dif. Sp.", "speed": "Velocità"
                }
                for stat in data["stats"]:
                    name = stat_names.get(stat["stat"]["name"], stat["stat"]["name"])
                    value = stat["base_stat"]
                    st.write(f"**{name}**: {value}")
                    st.progress(min(value / 255, 1.0))

        # Fonti recuperate
        st.subheader("📄 Fonti recuperate")
        for doc in docs:
            pokemon = doc.metadata.get("pokemon", "?")
            source = doc.metadata.get("source", "?")
            with st.expander(f"🔹 {pokemon} ({source})"):
                st.text(doc.page_content[:300] + "...")

    if st.session_state.chat_history:
        st.divider()
        if st.button("🗑️ Nuova chat", use_container_width=True):
            st.session_state.chat_history = []
            st.session_state.last_docs = []
            st.session_state.last_question = ""
            st.rerun()