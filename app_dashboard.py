"""
Dashboard Interativo Streamlit
Interface de exploração visual para Movies Knowledge Base
"""

import streamlit as st
import sys
from pathlib import Path


base_dir = Path(__file__).parent / 'movies_knowledge_base'
sys.path.insert(0, str(base_dir.parent / 'movies_knowledge_base'))

import chromadb
from src.services.embedder import DocumentEmbedder
from src.application.search_cloud import search_movies_cloud
from src.config.chroma_config import CHROMA_API_KEY, CHROMA_TENANT, CHROMA_DATABASE

st.set_page_config(
    page_title="Movies Knowledge Base",
    page_icon="🎬",
    layout="wide"
)

@st.cache_resource
def load_cloud_collection():
    """Conecta ao Chroma Cloud"""
    client = chromadb.CloudClient(
        api_key=CHROMA_API_KEY,
        tenant=CHROMA_TENANT,
        database=CHROMA_DATABASE
    )
    collection = client.get_collection("movies_docs")
    return collection

@st.cache_resource
def load_embedder():
    """Carrega modelo de embeddings"""
    return DocumentEmbedder(model_name='all-MiniLM-L6-v2')

def main():
    st.title("🎬 Movies Knowledge Base")
    
    st.sidebar.title("Navegação")
    page = st.sidebar.radio(
        "Escolha uma página",
        ["Visão Geral", "Busca Semântica"]
    )
    
    collection = load_cloud_collection()
    embedder = load_embedder()
    
    if page == "Visão Geral":
        st.header("Visão Geral do Sistema")
        
        total = collection.count()
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Total de Filmes", f"{total:,}")
        
        with col2:
            st.metric("Embedding Dimension", "384")
        
        with col3:
            st.metric("Vector DB", "Chroma Cloud")
        
        st.markdown("---")
        st.info("""
        **Projeto Final - DSML**
        
        Sistema de busca semântica de filmes usando embeddings e ChromaDB.
        """)

    elif page == "Busca Semântica":
        st.header("Busca Semântica")
        
        st.markdown("Busque filmes usando linguagem natural.")
        
        query = st.text_input("Digite sua busca:", placeholder="ex: filme de ação com perseguição")
        n_results = st.slider("Número de resultados:", 1, 10, 5)
        
        if st.button("Buscar", type="primary"):
            if query:
                with st.spinner("Buscando..."):
                    results = search_movies_cloud(query, n_results=n_results)
                
                if results['documents'] and len(results['documents'][0]) > 0:
                    st.success(f"Encontrados {len(results['documents'][0])} resultados")
                    
                    for i, (doc, dist, meta) in enumerate(zip(
                        results['documents'][0],
                        results['distances'][0],
                        results['metadatas'][0]
                    ), 1):
                        similarity = (1 - dist) * 100
                        title = doc.split('\n')[0]
                        with st.expander(f"{i}. {title} - Similaridade: {similarity:.1f}%"):
                            st.text(doc[:500])
                else:
                    st.warning("Nenhum resultado encontrado.")
            else:
                st.warning("Digite uma busca.")
    

    
    st.sidebar.markdown("---")
    st.sidebar.caption("☁️ Powered by Chroma Cloud")

if __name__ == "__main__":
    main()
