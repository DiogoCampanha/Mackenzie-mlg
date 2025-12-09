# Projeto: Classificação de Gênero Literário com LLM + RAG

Este projeto implementa um sistema de classificação de gênero literário
utilizando **LLMs (Large Language Models)** em formato **GGUF**,
combinadas com **RAG (Retrieval-Augmented Generation)** para aumento de
precisão, eficiência e interpretabilidade.

## 🚀 Visão Geral

O sistema utiliza:

-   **Qwen2.5-1.5B-Instruct-GGUF** para classificação textual;
-   **Sentence-BERT (all-MiniLM-L6-v2)** para geração de embeddings;
-   **FAISS** para busca vetorial eficiente;
-   **Pipeline RAG** para recuperação de exemplos relevantes antes da
    inferência;
-   **Dataset próprio** construído a partir de textos do *Project
    Gutenberg* (Domínio Público).

## 📂 Estrutura Geral

    ├── gutenberg_texts/              # Textos brutos
    ├── gutenberg_dataset.csv         # Dataset final de parágrafos + rótulos
    ├── qwen2.5-1.5b-instruct.gguf    # Modelo LLM quantizado
    ├── notebook.ipynb                # Código exploratório
    └── README.md                     # Este arquivo

## 🧠 Modelos Utilizados

### 🔹 Embeddings

-   **Sentence-BERT -- all-MiniLM-L6-v2**
-   Origem: https://www.sbert.net
-   Referência acadêmica:\
    **Reimers, N.; Gurevych, I. Sentence-BERT: Sentence Embeddings using
    Siamese BERT-Networks (2019).**

### 🔹 LLM (Inferência)

-   **Qwen2.5 1.5B Instruct -- GGUF**
-   Inferência via **llama-cpp-python**
-   Roda totalmente offline

### 🔹 RAG

-   Recuperação baseada em embeddings (k=3 exemplos)
-   Melhora significativa na precisão em comparação com zero-shot LLM

## 📊 Resultados e Conclusões

A solução RAG apresentou desempenho superior ao uso do modelo LLM
isolado (zero-shot).\
Isso confirma achados da literatura, especialmente:

-   **Lewis et al. (2020)** --- *Retrieval-Augmented Generation*
-   **Reimers & Gurevych (2019)** --- eficácia de SBERT para busca
    semântica
-   **Dettmers et al. (2023)** --- eficiência de modelos quantizados

Links recomendados: - https://arxiv.org/abs/2005.11401\
- https://arxiv.org/abs/1908.10084\
- https://arxiv.org/abs/2305.14314

## ▶️ Execução

### 1. Instale dependências

``` bash
pip install sentence-transformers faiss-cpu llama-cpp-python pandas numpy
```

### 2. Baixe o modelo GGUF

``` bash
wget -O qwen2.5-1.5b-instruct.gguf https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct-GGUF/resolve/main/qwen2.5-1.5b-instruct-q4_k_m.gguf
```

### 3. Gere o dataset

``` python
python build_dataset.py
```

### 4. Rode o classificador

``` python
python classify.py
```

## 📘 Licença

Este projeto utiliza textos em domínio público e é disponibilizado sob
licença MIT.

## ✨ Autor

Projeto desenvolvido por Diogo Campanha (2025).
