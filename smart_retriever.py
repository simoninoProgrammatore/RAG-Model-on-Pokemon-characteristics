from langchain_core.documents import Document
from typing import List
import re


class SmartRetriever:
    def __init__(self, bm25_retriever, faiss_retriever, all_documents: List[Document]):
        self.bm25 = bm25_retriever
        self.faiss = faiss_retriever
        self.all_docs = all_documents

        # Costruisco indice nome -> lista di documenti (ora ci sono più chunk per Pokemon)
        self.name_index = {}
        for doc in all_documents:
            pokemon = doc.metadata.get("pokemon", "").lower()
            if pokemon:
                if pokemon not in self.name_index:
                    self.name_index[pokemon] = []
                self.name_index[pokemon].append(doc)

    # ─────────────────────────────────────────────
    # 1️⃣ Entity matching diretto su metadata
    # ─────────────────────────────────────────────
    def extract_pokemon_names(self, query: str) -> List[str]:
        query_lower = query.lower()
        found = []

        for name in self.name_index.keys():
            # match parola intera (evita match parziali)
            pattern = r"\b" + re.escape(name) + r"\b"
            if re.search(pattern, query_lower):
                found.append(name)

        return found

    def metadata_match(self, query: str) -> List[Document]:
        """
        Trova tutti i chunk dei Pokemon menzionati.
        Opzionalmente filtra per aspetto se la query lo richiede.
        """
        names = self.extract_pokemon_names(query)
        matched_docs = []
        
        for name in names:
            if name in self.name_index:
                docs = self.name_index[name]
                
                # Filtra per aspetto se rilevante
                aspect_filter = self._detect_aspect(query)
                if aspect_filter:
                    docs = [d for d in docs if d.metadata.get("aspect") == aspect_filter]
                
                matched_docs.extend(docs)
        
        return matched_docs

    def _detect_aspect(self, query: str) -> str:
        """
        Rileva quale aspetto cercare in base alla query.
        """
        query_lower = query.lower()
        
        if any(k in query_lower for k in ["stat", "hp", "attacco", "difesa", "velocità"]):
            return "statistiche"
        elif any(k in query_lower for k in ["tipo", "type"]):
            return "tipi"
        elif any(k in query_lower for k in ["abilità", "ability", "abilita"]):
            return "abilità"
        elif any(k in query_lower for k in ["mossa", "mosse", "move"]):
            return "mosse"
        
        return None  # Nessun filtro specifico

    # ─────────────────────────────────────────────
    # 2️⃣ Hybrid semantic fallback
    # ─────────────────────────────────────────────
    def semantic_search(self, query: str) -> List[Document]:
        keywords_multi = {
            "tutti", "quali", "confronta", "migliore",
            "peggiore", "più", "meno", "lista",
            "quanti", "elenco"
        }

        is_multi = any(k in query.lower() for k in keywords_multi)
        k = 15 if is_multi else 5  # Aumentato perché ora ci sono più chunk

        self.bm25.k = k
        self.faiss.search_kwargs["k"] = k

        bm25_docs = self.bm25.invoke(query)
        faiss_docs = self.faiss.invoke(query)

        # Merge con deduplica
        seen = set()
        merged = []

        for doc in bm25_docs + faiss_docs:
            # Deduplica per combinazione pokemon + aspect
            pokemon = doc.metadata.get("pokemon", "?")
            aspect = doc.metadata.get("aspect", "?")
            key = f"{pokemon}_{aspect}"
            
            if key not in seen:
                seen.add(key)
                merged.append(doc)

        return merged

    # ─────────────────────────────────────────────
    # 3️⃣ Entry point principale
    # ─────────────────────────────────────────────
    def retrieve(self, query: str) -> List[Document]:
        # Step 1: entity matching
        matched_docs = self.metadata_match(query)

        if matched_docs:
            return matched_docs

        # Step 2: fallback semantico
        return self.semantic_search(query)