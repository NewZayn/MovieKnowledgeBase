---
title: MovieKnowledgeBase
emoji: 💬
colorFrom: yellow
colorTo: purple
sdk: gradio
sdk_version: 5.42.0
app_file: app.py
pinned: false
hf_oauth: true
hf_oauth_scopes:
- inference-api
license: mit
---

An example chatbot using [Gradio](https://gradio.app), [`huggingface_hub`](https://huggingface.co/docs/huggingface_hub/v0.22.2/en/index), and the [Hugging Face Inference API](https://huggingface.co/docs/api-inference/index).


---
title: MovieKnowledgeBase
emoji: 💬
colorFrom: yellow
colorTo: purple
sdk: gradio
sdk_version: 5.42.0
app_file: app.py
pinned: false
hf_oauth: true
hf_oauth_scopes:
- inference-api
license: mit
---

## DOC - MOVIEKNOWLEDGEBASE

This repository contains the code for a Movie Knowledge Base chatbot application. The application leverages Gradio for the user interface, Hugging Face's Inference API for natural language processing, and a vector database for storing and retrieving movie-related information.


# Movies Knowledge Base

Semantic search system for movies using embeddings and Chroma Cloud.

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

