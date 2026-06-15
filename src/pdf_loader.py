from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma


PDF_PATH = "data/sample 1.pdf"

loader = PyPDFLoader(PDF_PATH)

documents = loader.load()
print(f"Pages loaded: {len(documents)}")
print(documents[0].page_content[:500])

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50
)
chunks = text_splitter.split_documents(documents)
print(f"Total chunks created: {len(chunks)}")

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

vector_store = Chroma.from_documents(
    documents=chunks,
    embedding=embeddings,
    persist_directory="./chroma_db"
)
print(vector_store._collection.count())

results = vector_store.similarity_search(
    "What is Auto Scaling?",
    k=3
)
for i, doc in enumerate(results, 1):
    print(f"\nResult {i}")
    print("-" * 50)
    print(doc.page_content[:500])