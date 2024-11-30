**RAG-based Customer Query Answering System**

This repository contains a Retrieval-Augmented Generation (RAG) system designed to answer customer queries by retrieving relevant information from a vector database and generating accurate responses using large language models.

**Features**

Natural Language Understanding: Powered by OpenAI's language model.
Fast Information Retrieval: Utilizes FAISS (Facebook AI Similarity Search) as a vector database for quick and efficient similarity searches.
High-Quality Embeddings: Sentence Transformer models are used to convert user queries and documents into dense embeddings for better semantic matching.
Modular Design: Built using the LangChain framework to easily handle document loading, vector stores, and language models.
Architecture Overview
Query Embedding: The user's query is transformed into a vector using a Sentence Transformer model.
Vector Search: The query vector is compared against a FAISS vector database containing indexed knowledge.
Contextual Retrieval: The most relevant documents are retrieved based on similarity to the query.
Answer Generation: OpenAI's language model generates a context-aware response using the retrieved documents.

**Technologies Used**

LangChain: Framework for building language model-powered applications.
OpenAI API: Language model for generating responses.
FAISS: Library for efficient similarity search and clustering of dense vectors.
Sentence Transformers: Pre-trained models to generate sentence embeddings.
