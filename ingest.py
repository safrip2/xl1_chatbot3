#from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from transformers import AutoTokenizer

DATA_PATH = "data/clean"
DB_FAISS_PATH = "db_clean" 


tokenizer = AutoTokenizer.from_pretrained("intfloat/multilingual-e5-large", force_download=True)


def create_vector_db():
    loader = DirectoryLoader(
        DATA_PATH, glob="*.pdf", loader_cls=PyPDFLoader
    )
    documents = loader.load()

    # custom len jika sebelumnya karakter sekarang by token
    def tiktoken_len(text):
        tokens = tokenizer.encode(text, add_special_tokens=False)
        return len(tokens)
    
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=300,
        chunk_overlap=50,
        length_function=tiktoken_len,  # use the custom length function
        #separators=["\n\n", "\n", " ", ".", ",", "\u200b", "\uff0c", "\u3001", "\uff0e", "\u3002", ""]
    )
    texts = text_splitter.split_documents(documents)

    embed_model = HuggingFaceEmbeddings(
        model_name="intfloat/multilingual-e5-large", model_kwargs={"device": "cuda"} # -> bisa ganti model lain
    )

    db = FAISS.from_documents(texts, embed_model)
    db.save_local(DB_FAISS_PATH)

if __name__ == "__main__":
    create_vector_db()