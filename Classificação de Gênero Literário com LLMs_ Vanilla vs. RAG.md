# Classificação de Gênero Literário com LLMs: Vanilla vs. RAG

Este projeto explora e compara duas abordagens de classificação de texto *zero-shot* para determinar o gênero literário de trechos de livros do Project Gutenberg. A primeira solução utiliza uma pipeline padrão de classificação *zero-shot* da Hugging Face, enquanto a segunda implementa uma arquitetura de Geração Aumentada por Recuperação (RAG) para melhorar a precisão.

## Visão Geral do Projeto

O objetivo principal é demonstrar como a técnica RAG pode aprimorar a capacidade de um Modelo de Linguagem Grande (LLM) em tarefas de classificação, fornecendo contexto relevante em tempo de inferência. O notebook compara uma abordagem *vanilla* (sem contexto adicional) com uma solução RAG que recupera exemplos relevantes de uma base de dados vetorial.

O conjunto de dados foi construído a partir de obras clássicas do [Project Gutenberg](https://www.gutenberg.org/), com trechos de texto rotulados em quatro categorias de gênero:

- Aventura
- Romance
- Ficção Científica (Sci-Fi)
- Terror

## Estrutura do Projeto

O fluxo de trabalho é implementado no notebook `projeto_final_classificacao.ipynb` e segue as seguintes etapas:

1.  **Instalação de Dependências**: Configuração do ambiente com as bibliotecas necessárias, como `transformers`, `sentence-transformers`, `faiss-cpu`, e `llama-cpp-python`.
2.  **Coleta e Preparação de Dados**: Download de textos do Project Gutenberg e processamento para extrair trechos de tamanho uniforme.
3.  **Criação do Dataset**: Geração de um arquivo CSV (`gutenberg_dataset.csv`) contendo os trechos de texto e seus respectivos rótulos de gênero.
4.  **Abordagem 1: Classificação Zero-Shot (Baseline)**: Utilização da pipeline `zero-shot-classification` da Hugging Face com um modelo pré-treinado em Inferência de Linguagem Natural (NLI), como o `facebook/bart-large-mnli`.
5.  **Abordagem 2: Classificação com RAG**: Implementação de um fluxo RAG completo:
    - **Indexação**: Geração de *embeddings* dos textos de treinamento usando um modelo da `sentence-transformers`.
    - **Armazenamento**: Criação de um índice vetorial com `FAISS` para busca rápida de similaridade.
    - **Recuperação e Classificação**: Para cada novo texto, os trechos mais similares são recuperados do índice e fornecidos como contexto para um LLM local (Qwen 1.5B GGUF), que realiza a classificação final.
6.  **Avaliação**: Comparação das métricas de desempenho (Acurácia, Precisão, Recall e F1-Score) de ambas as abordagens.

## Resultados

A implementação da arquitetura RAG resultou em uma melhoria substancial na precisão da classificação em comparação com a abordagem *zero-shot* vanilla. Os resultados demonstram a eficácia de fornecer exemplos relevantes para o LLM, permitindo que ele tome decisões mais informadas e contextuais.

| Abordagem | Acurácia | F1-Score (Ponderado) |
| :--- | :---: | :---: |
| **Zero-Shot (Vanilla)** | 35.00% | 35.72% |
| **RAG + Qwen 1.5B** | **55.65%** | **55.27%** |

O aumento de mais de 20 pontos percentuais na acurácia destaca o valor da recuperação de informações para tarefas de classificação em domínios específicos.

## Como Executar o Projeto

1.  **Clone o Repositório**:

    ```bash
    git clone https://github.com/seu-usuario/seu-repositorio.git
    cd seu-repositorio
    ```

2.  **Instale as Dependências**:

    Recomenda-se o uso de um ambiente virtual. As dependências estão listadas no início do notebook e podem ser instaladas com `pip`.

    ```bash
    pip install -r requirements.txt
    ```

    *(Nota: Um arquivo `requirements.txt` pode ser gerado a partir das células de instalação do notebook)*

3.  **Execute o Notebook**:

    Abra e execute o `projeto_final_classificacao.ipynb` em um ambiente como Jupyter Lab ou Google Colab. Para a abordagem RAG, o notebook fará o download do modelo GGUF `Qwen/Qwen2.5-1.5B-Instruct-GGUF`.

    *É altamente recomendável executar o notebook em um ambiente com GPU para acelerar a geração de embeddings e a inferência do LLM.*

## Tecnologias Utilizadas

- **Modelos de Linguagem**: `facebook/bart-large-mnli`, `Qwen/Qwen2.5-1.5B-Instruct-GGUF`
- **Embeddings**: `sentence-transformers`
- **Índice Vetorial**: `faiss-cpu`
- **Orquestração**: `llama-cpp-python`
- **Bibliotecas de ML/DL**: `transformers`, `scikit-learn`
- **Manipulação de Dados**: `pandas`, `numpy`

## Referências

Os resultados obtidos estão alinhados com pesquisas contemporâneas sobre RAG e LLMs, incluindo estudos de Lewis et al. (2020) sobre Retrieval-Augmented Generation, Reimers & Gurevych (2019) sobre Sentence-BERT e investigações recentes sobre modelos compactos e quantizados, como Dettmers et al. (2023). Assim, o trabalho reforça a relevância do uso de arquiteturas híbridas para melhorar a confiabilidade e a precisão de modelos de linguagem em problemas aplicados de classificação.
