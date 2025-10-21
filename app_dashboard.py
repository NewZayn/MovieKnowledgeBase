import streamlit as st
import sys
from pathlib import Path

base_dir = Path(__file__).parent / 'movies_knowledge_base'
sys.path.insert(0, str(base_dir.parent / 'movies_knowledge_base'))

import chromadb
from src.services.embedder import DocumentEmbedder
from src.application.search_cloud import search_movies_cloud
from src.config.chroma_config import CHROMA_API_KEY, CHROMA_TENANT, CHROMA_DATABASE
from src.services.clustering import DocumentClusterer
from src.services.anomaly_detection import AnomalyDetector

st.set_page_config(
    page_title="Movies Knowledge Base",
    page_icon="🎬",
    layout="wide"
)

@st.cache_resource
def load_cloud_collection():
    client = chromadb.CloudClient(
        api_key=CHROMA_API_KEY,
        tenant=CHROMA_TENANT,
        database=CHROMA_DATABASE
    )
    collection = client.get_collection("movies_docs")
    return collection

@st.cache_resource
def load_embedder():
    """Load embeddings model"""
    return DocumentEmbedder(model_name='all-MiniLM-L6-v2')

def main():
    st.title("🎬 Movies Knowledge Base")
    
    collection = load_cloud_collection()
    embedder = load_embedder()
    
    st.sidebar.title("Navigation")
    page = st.sidebar.radio(
        "Choose a page",
        ["Overview", "Semantic Search", "Clustering", "Anomaly Detection"]
    )
    
    if page == "Overview":
        st.header("System Overview")
        
        total = collection.count()
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Total Movies", f"{total:,}")
        
        with col2:
            st.metric("Embedding Dimension", "384")
        
        with col3:
            st.metric("Vector DB", "Chroma Cloud")
        
        st.markdown("---")
        st.info("""
        **Final Project - DSML**
        
        Semantic movie search system using embeddings and ChromaDB.
        """)

    elif page == "Semantic Search":
        st.header("Semantic Search")
        
        st.markdown("Search movies using natural language.")
        
        query = st.text_input("Enter your search:", placeholder="e.g., action movie with car chase")
        n_results = st.slider("Number of results:", 1, 10, 5)
        
        if st.button("Search", type="primary"):
            if query:
                with st.spinner("Searching..."):
                    results = search_movies_cloud(query, n_results=n_results)
                
                if results['documents'] and len(results['documents'][0]) > 0:
                    st.success(f"Found {len(results['documents'][0])} results")
                    
                    for i, (doc, dist) in enumerate(zip(
                        results['documents'][0],
                        results['distances'][0]
                    ), 1):
                        similarity = (2 - dist) / 2 * 100
                        title = doc.split('\n')[0]
                        with st.expander(f"{i}. {title} - Similarity: {similarity:.1f}%"):
                            st.text(doc[:500])
                else:
                    st.warning("No results found.")
            else:
                st.warning("Please enter a search query.")
    
    elif page == "Clustering":
        st.header("Document Clustering")
        
        st.markdown("Automatically group similar movies.")
        
        algorithm = st.selectbox("Algorithm", ["K-Means", "DBSCAN"])
        
        if algorithm == "K-Means":
            n_clusters = st.slider("Number of Clusters", 3, 15, 8)
        else:
            eps = st.slider("Epsilon", 0.3, 1.5, 0.5, 0.1)
            min_samples = st.slider("Min samples", 3, 10, 5)
        
        if st.button("Run Clustering", type="primary"):
            with st.spinner(f"Running {algorithm}..."):
                base_path = Path(__file__).parent / 'movies_knowledge_base'
                embeddings_dir = base_path / 'data/processed/embeddings'
                
                try:
                    clusterer = DocumentClusterer(str(embeddings_dir))
                    
                    if algorithm == "K-Means":
                        labels = clusterer.cluster_kmeans(n_clusters=n_clusters)
                    else:
                        labels = clusterer.cluster_dbscan(eps=eps, min_samples=min_samples)
                    
                    analysis = clusterer.analyze_clusters()
                    
                    st.success("Clustering completed!")
                    
                    st.subheader("Cluster Analysis")
                    st.dataframe(analysis[['cluster_name', 'n_documents', 'percentage']], use_container_width=True)
                    
                    st.subheader("Examples per Cluster")
                    for _, row in analysis.head(5).iterrows():
                        with st.expander(f"{row['cluster_name']} - {row['n_documents']} movies"):
                            for title in row['sample_titles'][:3]:
                                st.markdown(f"- {title}")
                except Exception as e:
                    st.error(f"Error: {str(e)}")
                    st.info("Clustering requires local embeddings. Run the pipeline first.")
    
    elif page == "Anomaly Detection":
        st.header("Anomaly Detection")
        
        st.markdown("Identify unusual or problematic documents.")
        
        method = st.selectbox("Method", ["Isolation Forest", "Local Outlier Factor"])
        contamination = st.slider("Expected % of anomalies", 0.01, 0.15, 0.05, 0.01)
        
        if st.button("Detect Anomalies", type="primary"):
            with st.spinner(f"Running {method}..."):
                base_path = Path(__file__).parent / 'movies_knowledge_base'
                embeddings_dir = base_path / 'data/processed/embeddings'
                
                try:
                    detector = AnomalyDetector(str(embeddings_dir))
                    
                    if method == "Isolation Forest":
                        scores, is_anomaly = detector.detect_isolation_forest(contamination=contamination)
                    else:
                        scores, is_anomaly = detector.detect_lof(contamination=contamination)
                    
                    st.success("Detection completed!")
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.metric("Anomalies Detected", f"{is_anomaly.sum():,}")
                    
                    with col2:
                        st.metric("Percentage", f"{(is_anomaly.sum()/len(is_anomaly)*100):.1f}%")

                    st.subheader("Top 5 Anomalies")

                    top_anomalies = detector.get_top_anomalies(n=5)

                    for i, anom in enumerate(top_anomalies, 1):
                        title = anom['text'].split('\n')[0] if '\n' in anom['text'] else anom['text'][:80]
                        with st.expander(f"{i}. {title}"):
                            st.text(anom['text'][:400])
                except Exception as e:
                    st.error(f"Error: {str(e)}")
                    st.info("Anomaly detection requires local embeddings. Run the pipeline first.")
    
    st.sidebar.markdown("---")
    st.sidebar.caption("☁️ Powered by Chroma Cloud")

if __name__ == "__main__":
    main()

"""
Interactive Streamlit Dashboard
Visual exploration interface for Movies Knowledge Base
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
from src.services.clustering import DocumentClusterer
from src.services.anomaly_detection import AnomalyDetector

st.set_page_config(
    page_title="Movies Knowledge Base",
    page_icon="🎬",
    layout="wide"
)

@st.cache_resource
def load_cloud_collection():
    client = chromadb.CloudClient(
        api_key=CHROMA_API_KEY,
        tenant=CHROMA_TENANT,
        database=CHROMA_DATABASE
    )
    collection = client.get_collection("movies_docs")
    return collection

@st.cache_resource
def load_embedder():
    """Load embeddings model"""
    return DocumentEmbedder(model_name='all-MiniLM-L6-v2')

def main():
    st.title("🎬 Movies Knowledge Base")
    
    collection = load_cloud_collection()
    embedder = load_embedder()
    
    st.sidebar.title("Navigation")
    page = st.sidebar.radio(
        "Choose a page",
        ["Overview", "Semantic Search", "Clustering", "Anomaly Detection"]
    )
    
    if page == "Overview":
        st.header("System Overview")
        
        total = collection.count()
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Total Movies", f"{total:,}")
        
        with col2:
            st.metric("Embedding Dimension", "384")
        
        with col3:
            st.metric("Vector DB", "Chroma Cloud")
        
        st.markdown("---")
        st.info("""
        **Final Project - DSML**
        
        Semantic movie search system using embeddings and ChromaDB.
        """)

    elif page == "Semantic Search":
        st.header("Semantic Search")
        
        st.markdown("Search movies using natural language.")
        
        query = st.text_input("Enter your search:", placeholder="e.g., action movie with car chase")
        n_results = st.slider("Number of results:", 1, 10, 5)
        
        if st.button("Search", type="primary"):
            if query:
                with st.spinner("Searching..."):
                    results = search_movies_cloud(query, n_results=n_results)
                
                if results['documents'] and len(results['documents'][0]) > 0:
                    st.success(f"Found {len(results['documents'][0])} results")
                    
                    for i, (doc, dist) in enumerate(zip(
                        results['documents'][0],
                        results['distances'][0]
                    ), 1):
                        similarity = (2 - dist) / 2 * 100
                        title = doc.split('\n')[0]
                        with st.expander(f"{i}. {title} - Similarity: {similarity:.1f}%"):
                            st.text(doc[:500])
                else:
                    st.warning("No results found.")
            else:
                st.warning("Please enter a search query.")
    
    elif page == "Clustering":
        st.header("Document Clustering")
        
        st.markdown("Automatically group similar movies.")
        
        algorithm = st.selectbox("Algorithm", ["K-Means", "DBSCAN"])
        
        if algorithm == "K-Means":
            n_clusters = st.slider("Number of Clusters", 3, 15, 8)
        else:
            eps = st.slider("Epsilon", 0.3, 1.5, 0.5, 0.1)
            min_samples = st.slider("Min samples", 3, 10, 5)
        
        if st.button("Run Clustering", type="primary"):
            with st.spinner(f"Running {algorithm}..."):
                base_path = Path(__file__).parent / 'movies_knowledge_base'
                embeddings_dir = base_path / 'data/processed/embeddings'
                
                try:
                    clusterer = DocumentClusterer(str(embeddings_dir))
                    
                    if algorithm == "K-Means":
                        labels = clusterer.cluster_kmeans(n_clusters=n_clusters)
                    else:
                        labels = clusterer.cluster_dbscan(eps=eps, min_samples=min_samples)
                    
                    analysis = clusterer.analyze_clusters()
                    
                    st.success("Clustering completed!")
                    
                    st.subheader("Cluster Analysis")
                    st.dataframe(analysis[['cluster_name', 'n_documents', 'percentage']], use_container_width=True)
                    
                    st.subheader("Examples per Cluster")
                    for _, row in analysis.head(5).iterrows():
                        with st.expander(f"{row['cluster_name']} - {row['n_documents']} movies"):
                            for title in row['sample_titles'][:3]:
                                st.markdown(f"- {title}")
                except Exception as e:
                    st.error(f"Error: {str(e)}")
                    st.info("Clustering requires local embeddings. Run the pipeline first.")
    
    elif page == "Anomaly Detection":
        st.header("Anomaly Detection")
        
        st.markdown("Identify unusual or problematic documents.")
        
        method = st.selectbox("Method", ["Isolation Forest", "Local Outlier Factor"])
        contamination = st.slider("Expected % of anomalies", 0.01, 0.15, 0.05, 0.01)
        
        if st.button("Detect Anomalies", type="primary"):
            with st.spinner(f"Running {method}..."):
                base_path = Path(__file__).parent / 'movies_knowledge_base'
                embeddings_dir = base_path / 'data/processed/embeddings'
                
                try:
                    detector = AnomalyDetector(str(embeddings_dir))
                    
                    if method == "Isolation Forest":
                        scores, is_anomaly = detector.detect_isolation_forest(contamination=contamination)
                    else:
                        scores, is_anomaly = detector.detect_lof(contamination=contamination)
                    
                    st.success("Detection completed!")
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.metric("Anomalies Detected", f"{is_anomaly.sum():,}")
                    
                    with col2:
                        st.metric("Percentage", f"{(is_anomaly.sum()/len(is_anomaly)*100):.1f}%")

                    st.subheader("Top 5 Anomalies")

                    top_anomalies = detector.get_top_anomalies(n=5)

                    for i, anom in enumerate(top_anomalies, 1):
                        title = anom['text'].split('\n')[0] if '\n' in anom['text'] else anom['text'][:80]
                        with st.expander(f"{i}. {title}"):
                            st.text(anom['text'][:400])
                except Exception as e:
                    st.error(f"Error: {str(e)}")
                    st.info("Anomaly detection requires local embeddings. Run the pipeline first.")
    
    st.sidebar.markdown("---")
    st.sidebar.caption("☁️ Powered by Chroma Cloud")

if __name__ == "__main__":
    main()

