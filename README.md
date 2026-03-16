---
title: MovieKnowledgeBase
emoji: 💬
colorFrom: yellow
colorTo: purple
sdk: gradio
sdk_version: 6.9.0
app_file: app.py
pinned: false
hf_oauth: true
hf_oauth_scopes:
- inference-api
license: mit
---

### Job Title: Movie Knowledge Base – Semantic Movie Search System

###  Profile: AI / Machine Learning Developer

### Employer: Academic Project – Jala University

### Start date: 09/2025

### End date: 12/2025

### Skills applied :

  Natural Language Processing (NLP)
  
  Semantic Search
  
  Vector Embeddings
  
  Information Retrieval

## Tools : 

- Python 3.12

## Frameworks : 
- Sentence-Transformers (all-MiniLM-L6-v2)
- Chroma Cloud
- Gradio + Streamlit
- Scikit-learn



# Movies Knowledge Base

## Descripition 

### DOC - MOVIEKNOWLEDGEBASE

This repository contains the code for a Movie Knowledge Base chatbot application. The application leverages Gradio for the user interface, Hugging Face's Inference API for natural language processing, and a vector database for storing and retrieving movie-related information.


## Introduction

A system that allows searching for movies using natural language. Type "action movie with a chase scene" and find similar movies by meaning, not just exact words.

## How It Works

1. 43,970 movies from Kaggle transformed into embeddings (384-dimensional vectors)
2. Stored in Chroma Cloud (vector database)
3. Semantic similarity search using sentence-transformers