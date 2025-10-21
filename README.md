# Movies Knowledge Base

Sistema de busca semântica de filmes usando embeddings e Chroma Cloud.

## Intrudução 

Sistema que permite buscar filmes usando linguagem natural. Digite "filme de ação com perseguição" e encontre filmes similares por significado, não apenas palavras exatas.

## Funcionamento

1. 43.970 filmes do Kaggle transformados em embeddings (vetores de 384 dimensões)
2. Armazenados no Chroma Cloud (banco vetorial)
3. Busca por similaridade semântica usando sentence-transformers

## Tecnologias

- Python 3.12
- Sentence-Transformers (all-MiniLM-L6-v2)
- Chroma Cloud
- Gradio + Streamlit
- Scikit-learn
