# Movies Knowledge Base

Sistema de busca semântica de filmes usando embeddings e ChromaDB.

**Projeto Final - Machine Learning & Data Science**

## O que é isso?

Um sistema que permite buscar filmes usando linguagem natural. Por exemplo, você pode digitar "filme de ação com perseguição de carro" e ele encontra filmes similares, mesmo que essas palavras exatas não estejam na descrição.

## Como funciona?

1. Pega descrições de ~44 mil filmes do Kaggle
2. Transforma cada descrição em um vetor de 384 números (embedding)
3. Guarda tudo no ChromaDB (banco de dados vetorial)
4. Quando você busca algo, transforma sua busca em vetor e acha os filmes mais parecidos

## Estrutura

```
movies_knowledge_base/
├── data/
│   ├── archive/              # CSVs do Kaggle
│   └── raw/documents/        # Documentos de filmes gerados
├── src/
│   ├── embedder.py           # Gera os embeddings
│   ├── vector_db.py          # Gerencia o ChromaDB
│   ├── search.py             # Busca semântica
│   ├── clustering.py         # Agrupa filmes similares
│   └── anomaly_detection.py # Detecta documentos estranhos
├── chroma_db/                # Banco vetorial
├── app.py                    # Dashboard web
└── pipeline.py               # Roda tudo
```

## Como usar

### 1. Instalar

```bash
# Criar ambiente virtual
python3 -m venv .venv
source .venv/bin/activate

# Instalar dependências
pip install -r requirements.txt
```

### 2. Baixar dados

Baixe os datasets do Kaggle:
- [The Movies Dataset](https://www.kaggle.com/datasets/rounakbanik/the-movies-dataset)

Coloque os CSVs em `data/archive/`

### 3. Gerar documentos e embeddings

```bash
# Gera os documentos de filmes
python src/document_generator.py

# Roda o pipeline completo
python pipeline.py
```

Isso vai demorar um pouco (são 44 mil filmes).

### 4. Usar o sistema

**Dashboard web:**
```bash
streamlit run app.py
```

**Busca no terminal:**
```bash
python src/search.py
```

## Funcionalidades

- **Busca semântica**: Busca por significado, não só palavras exatas
- **Clustering**: Agrupa filmes parecidos automaticamente
- **Detecção de anomalias**: Encontra documentos com problemas
- **Dashboard**: Interface visual para explorar os dados

## Tecnologias usadas

- Python 3.12
- Sentence-Transformers (all-MiniLM-L6-v2)
- ChromaDB
- Streamlit
- Scikit-learn

## Datasets

- movies_metadata.csv: 45,466 filmes
- credits.csv: elenco e equipe
- keywords.csv: palavras-chave

## Exemplos de busca

- "filme de romance trágico"
- "ação com explosões"
- "comédia romântica em Nova York"
- "ficção científica com viagem no tempo"

## Notas

- O modelo roda localmente (não precisa de API)
- Os dados ficam salvos no ChromaDB
- Primeira execução demora, mas depois é rápido

---

**Autor**: Gabriel  
**Curso**: Data Science & Machine Learning  
**Ano**: 2024
