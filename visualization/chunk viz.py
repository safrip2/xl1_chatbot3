from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from transformers import AutoTokenizer

DATA_PATH = 'chunk test/'
DB_FAISS_PATH = 'visualize'

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained("intfloat/multilingual-e5-large")

# Create vector database
def create_vector_db():
    loader = DirectoryLoader(
        DATA_PATH, glob="*.pdf", loader_cls=PyPDFLoader
    )
    documents = loader.load()

    
    def tiktoken_len(text):
        tokens = tokenizer.encode(text, add_special_tokens=False)
        return len(tokens)

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=300,
        chunk_overlap=50,
        length_function=tiktoken_len,  # use the custom length function
        separators=["\n\n", "\n", " ", ".", ",", "\u200b", "\uff0c", "\u3001", "\uff0e", "\u3002", ""],
    )
    texts = text_splitter.split_documents(documents)

    embed_model = HuggingFaceEmbeddings(
        model_name="intfloat/multilingual-e5-large", model_kwargs={"device": "cuda"}
    )

    # Visualize chunk
    for i, chunk in enumerate(texts):
        print(f"Chunk {i+1}:")
        print(f"Metadata: {chunk.metadata}")
        print(f"Content: {chunk.page_content}")
        print(f"Length (characters): {len(chunk.page_content)}")
        print(f"Length (tokens): {tiktoken_len(chunk.page_content)}")

        # Get embedding for the chunk
        embedding = embed_model.embed_query(chunk.page_content)
        print(f"total dimensi vektor: {len(embedding)}")  # Output: 1024
        print(f"Embedding (first 10 dimensions): {embedding[:10]}")
        print("----")

if __name__ == "__main__":
    create_vector_db()
