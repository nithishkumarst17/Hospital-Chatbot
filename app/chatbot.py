from dotenv import load_dotenv

from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain_classic.chains import RetrievalQA


load_dotenv()


# Load embedding model
embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)


# Load vector database
db = Chroma(
    persist_directory="./vector_db",
    embedding_function=embedding_model
)


# Retriever
retriever = db.as_retriever(
    search_kwargs={
        "k": 3
    }
)


# LLM
llm = ChatGroq(
    model="llama-3.1-8b-instant",
    temperature=0
)


# RAG chain
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=retriever
)


print(
"""
Hospital AI Assistant
Type exit to stop
"""
)


while True:

    question = input("\nPatient: ")

    if question.lower() == "exit":
        break

    answer = qa_chain.invoke(
        {
            "query": question
        }
    )

    print(
        "\nBot:",
        answer["result"]
    )