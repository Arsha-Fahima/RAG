import os
from dotenv import load_dotenv
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.agents import create_react_agent, AgentExecutor, Tool
from langchain.prompts import PromptTemplate
from langchain_community.document_loaders import DirectoryLoader

load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
CHROMA_DIR = "./chroma_tech_specs"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
LLM_MODEL = "models/gemini-2.0-flash"

last_query = None


def setup_rag_retriever():
    """
    Sets up the RAG pipeline: Load -> Split -> Embed -> Store -> Retrieve.
    """
    print("Starting RAG setup...")

    loader = DirectoryLoader(
        ".",
        glob="*.txt",
        loader_cls=lambda path: TextLoader(path, encoding="utf-8")
    )
    documents = loader.load()
    print(f"Loaded {len(documents)} documents from current folder.")

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50,
        length_function=len,
    )
    splits = text_splitter.split_documents(documents)
    print(f"Split into {len(splits)} chunks.")

    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    print(f"Using embedding model: {EMBEDDING_MODEL}")

    vectorstore = Chroma.from_documents(
        documents=splits,
        embedding=embeddings,
        persist_directory=CHROMA_DIR
    )
    print(f"ChromaDB created/loaded at {CHROMA_DIR}")

    retriever = vectorstore.as_retriever(search_kwargs={"k": 2})
    print("Retriever configured (k=2). RAG setup complete.")

    return retriever


def create_tech_spec_comparator_agent(retriever):
    llm = ChatGoogleGenerativeAI(
        model=LLM_MODEL,
        temperature=0.1,
        google_api_key=GOOGLE_API_KEY
    )

    def safe_rag_tool(query: str):
        global last_query
        if last_query == query:
            return "No new information available."
        last_query = query

        docs = retriever.invoke(query)
        if not docs:
            return "No relevant information found."
        return "\n\n".join([d.page_content for d in docs])

    rag_tool = Tool(
        name="Tech_Specs_Retriever",
        description="Retrieve GPU technical specification text chunks.",
        func=safe_rag_tool
    )

    custom_prompt = PromptTemplate(
        input_variables=["input", "agent_scratchpad", "tools", "tool_names"],
        template=(
            "You are the 'Tech Spec Comparator', a GPU specification expert.\n"
            "Your goal is to compare requested specification fields clearly and accurately.\n\n"
            "RULES:\n"
            "- Use the tool ONLY when you need data retrieval.\n"
            "- If data for a spec cannot be found, say 'Not available in dataset'.\n"
            "- If you already retrieved data once, do not call the tool again.\n"
            "- After collecting VRAM, Memory Bus, and TDP (or stating unavailable), "
            "STOP using tools and produce the final answer.\n\n"
            "Final answer must follow this format:\n"
            "Final Answer:\n"
            "| GPU | VRAM | Memory Bus | TDP |\n"
            "|------|-------|-------------|-------|\n"
            "| <GPU1> | <value> | <value> | <value> |\n"
            "| <GPU2> | <value> | <value> | <value> |\n"
            "Summary: <short performance comparison>\n\n"
            "--- TOOL USAGE FORMAT ---\n"
            "Thought: what you plan to do\n"
            "Action: tool name\n"
            "Action Input: data to search\n\n"
            "---\n"
            "TOOLS:\n{tools}\n\n"
            "Tool Names: {tool_names}\n"
            "User question: {input}\n\n"
            "{agent_scratchpad}"
        )
    )

    react_agent = create_react_agent(
        llm=llm,
        tools=[rag_tool],
        prompt=custom_prompt
    )

    agent_executor = AgentExecutor(
        agent=react_agent,
        tools=[rag_tool],
        verbose=True,
        max_iterations=10
    )

    print("Tech Spec Comparator Agent initialized.")
    return agent_executor


if __name__ == "__main__":
    tech_specs_retriever = setup_rag_retriever()
    comparator_agent = create_tech_spec_comparator_agent(tech_specs_retriever)

    print("\n--- Agent Ready. Ask a question. ---")

    query_1 = "Compare the VRAM, Memory Bus,and TDP of the RTX 4090 and the RX 7900 XTX."
    print(f"\nUser Query 1: {query_1}")
    result_1 = comparator_agent.invoke({"input": query_1})

    print("\n\n--- Agent Response 1 ---")
    print(result_1["output"])
    print("\n" + "=" * 50 + "\n")
