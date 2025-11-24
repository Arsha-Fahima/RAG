# 🔍 Tech Specification Comparator (RAG + ReAct AI Agent)

## 🧠 Project Overview
The **Tech Specification Comparator** is an AI-powered technical comparison system that uses **Retrieval Augmented Generation (RAG)** to compare GPU hardware specifications. It automatically extracts relevant information from technical text files and generates side-by-side comparisons using the **Gemini-2.0-Flash** model.

The project combines:
- 📂 Document Loading
- ✂️ Text Splitting into chunks
- 🧠 Embeddings using HuggingFace
- 💾 Chroma Vector Database
- 🤖 ReAct-Based Agent for tool execution
- 🔎 RAG Retrieval for spec extraction
- 🗣️ Natural Language comparison generation

---

## ✨ Features
| Feature | Description |
|--------|------------|
| GPU Spec Retrieval | Searches vector DB for relevant specification lines |
| AI-Powered Comparison | Uses Gemini LLM to analyze retrieved chunks |
| ReAct Agent | Decides when to call retrieval tools automatically |
| Custom Prompt | Controls structured reasoning |
| Chunk Visibility | Prints chunk details for debugging |
| Expandable | Add any `.txt` spec files to extend database |

---

## 🏗️ Tech Stack
| Component | Technology |
|-----------|-----------|
| LLM | Gemini-2.0-Flash |
| Embeddings | all-MiniLM-L6-v2 |
| Vector DB | ChromaDB |
| Agent Framework | LangChain |
| Loader | DirectoryLoader / TextLoader |
| Programming Language | Python 3 |

---

## 🚀 How it Works
1. Load `.txt` GPU spec files from folder
2. Split documents into 500-char chunks
3. Convert to vector embeddings & store in ChromaDB
4. AI agent uses retriever tool to fetch relevant chunks
5. Gemini model compares details based on user query

