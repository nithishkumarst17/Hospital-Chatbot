from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_chroma import Chroma

PDF_FILES=[
    "data/hospital_detail.pdf",
    "data/hospital_departments.pdf"
]
documents = []
for file in PDF_FILES:
    print(f"Loading {file}")
    loader = PyPDFLoader(file)
    docs = loader.load()
    documents.extend(docs)

print("PDF loading complete")
splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=100
)

chunks = splitter.split_documents(documents)
print("Chunks created:", len(chunks))

print("Loading embedding model...")
embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

db = Chroma.from_documents(
    documents=chunks,
    embedding=embedding_model,
    persist_directory="./vector_db"
)

print("Database created successfully!")