from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

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

for i, chunk in enumerate(chunks[:3]):
    print(f"\nChunk {i+1}")
    print("-" * 50)
    print(chunk.page_content)

for i in range(2):
    print(f"\nChunk {i+1}")
    print(chunks[i].page_content[-100:])

    print(f"\nChunk {i+2}")
    print(chunks[i+1].page_content[:100])