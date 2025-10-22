## DOC - MOVIEKNOWLEDGEBASE

This repository contains the code for a Movie Knowledge Base chatbot application. The application leverages Gradio for the user interface.

# Movies Knowledge Base

Semantic search system for movies using embeddings and Chroma Cloud.


## Execution 

**Requiriments** IMPORTANT

 "python3 -m venv venv"

 "source venv/bin/activate"

 "pip install requirements.txt"

**ChatBot**

 "python3 app.py"
 
**Dashboard**
 
 "streamlit run app_dashboard.py"

 **Pipiline**

 "python3 movies_knowledge_base/pipeline.py"

 

## Introduction

A system that allows searching for movies using natural language. Type "action movie with a chase scene" and find similar movies by meaning, not just exact words.

## How It Works

1. 43,970 movies from Kaggle transformed into embeddings (384-dimensional vectors)
2. Stored in Chroma Cloud (vector database)
3. Semantic similarity search using sentence-transformers

## Technologies

- Python 3.12
- Sentence-Transformers (all-MiniLM-L6-v2)
- Chroma Cloud
- Gradio + Streamlit
- Scikit-learn
