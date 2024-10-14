import os
from langchain.chains import ConversationalRetrievalChain
from langchain_community.document_loaders import DirectoryLoader
from langchain_community.document_loaders import PyPDFLoader
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.memory import ConversationBufferMemory
from langchain.prompts import PromptTemplate
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEndpoint
from langchain.prompts.chat import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)
from langchain.schema import (
    AIMessage,
    HumanMessage,
    SystemMessage
)
from transformers import AutoTokenizer
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import MessagesPlaceholder
import time


tokenizer = AutoTokenizer.from_pretrained("intfloat/multilingual-e5-large", force_download=True)

# Konfigurasi
DATA_PATH = "Data Tujuan 1"
DB_FAISS_PATH = "vector raw"
HF_API_TOKEN = os.environ.get("HUGGINGFACEHUB_API_TOKEN")
ENDPOINT_URL = "https://chboebbxb5ccit06.us-east-1.aws.endpoints.huggingface.cloud"  # SeaLLM 7B V2.5

# Muat Dokumen
loader = DirectoryLoader(DATA_PATH, glob="*.pdf", loader_cls=PyPDFLoader)
documents = loader.load()

# Memisahkan Teks
def tiktoken_len(text):
    tokens = tokenizer.encode(text, add_special_tokens=False)
    return len(tokens)

text_splitter = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=50, length_function=tiktoken_len)
texts = text_splitter.split_documents(documents)

# Inisialisasi Model Embedding
embeddings = HuggingFaceEmbeddings(model_name="intfloat/multilingual-e5-large")

# Membuat atau Memuat Database FAISS
if os.path.exists(DB_FAISS_PATH):
    db = FAISS.load_local(DB_FAISS_PATH, embeddings, allow_dangerous_deserialization=True)
else:
    db = FAISS.from_documents(texts, embeddings)
    db.save_local(DB_FAISS_PATH)

# Inisialisasi HuggingFaceEndpoint
llm = HuggingFaceEndpoint(
    endpoint_url=ENDPOINT_URL,
    huggingfacehub_api_token=HF_API_TOKEN,
    timeout=600,
    temperature=0.09,
    max_new_tokens=256,
)

# Inisialisasi Memory
memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

system_template = """You are a friendly and informative virtual sales assistant for XL Satu. 
Your role is to provide all information about XL Satu and help prospective customers choose products. 
Use a casualand informal tone, with a few emojis 😊 for a friendly touch. 
Always ensure customers understand by asking, “Is there anything else I can help you with?” 
If a question is beyond your scope, direct customers to XL Home customer service on Instagram (@xlhomeid), Facebook (xlhome id), 
Twitter (@xlhomeid), Email (XLHomeCS@xl.co.id), or WhatsApp (+62817-0010-820). 
Use proper and correct Indonesian, and avoid guessing or providing inaccurate information"""

# Question chain untuk merumuskan ulang pertanyaan (menggunakan RunnableSequence)
instruction_to_system = """
Given a chat history and the latest user question which might reference context in the chat history, 
formulate a standalone question which can be understood without the chat history. 
Do NOT answer the question, just reformulate it if needed and otherwise return it as is.
"""
question_maker_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", instruction_to_system),
        ("human", "{question}"),
    ]
)
question_chain = question_maker_prompt | llm | StrOutputParser()  # Menggunakan RunnableSequence

# QA system prompt
qa_system_prompt = """You are an assistant for question-answering tasks.\
Use the following pieces of retrieved context to answer the question.\
If you don't know the answer, provide a summary of the context. Do not generate your answer.\
{context}"""
qa_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", qa_system_prompt),
        ("human", "{question}")
    ]
)

# Buat ConversationalRetrievalChain
chain = ConversationalRetrievalChain.from_llm(
    llm,
    retriever=db.as_retriever(search_kwargs={"k": 5}),
    memory=memory,
    chain_type="stuff",
    combine_docs_chain_kwargs={"prompt": qa_prompt},
    
)

# Contoh Interaksi (Few-Shot Prompting)
chat_history = []
examples = [
    HumanMessage(content="Apa itu XL Satu?"),
    AIMessage(content="XL Satu adalah paket layanan konvergensi dari XL Axiata yang menggabungkan layanan seluler, internet rumah, dan konten hiburan dalam satu paket. Apakah ada lagi yang bisa saya bantu? 😊"),
    HumanMessage(content="Apa saja keuntungan menggunakan XL Satu?"),
    AIMessage(content="Keuntungan menggunakan XL Satu antara lain: kuota internet yang lebih besar, kecepatan internet yang stabil di rumah, akses ke berbagai konten hiburan, serta kemudahan dalam pembayaran dan pengelolaan tagihan. Apakah ada lagi yang bisa saya bantu? 😊"),
]
memory.chat_memory.add_ai_message(system_template)  # Menambahkan system message ke memory
for message in examples:
    memory.chat_memory.add_message(message)        # Menambahkan contoh interaksi ke memory

"""# Mulai Percakapan
while True:
    query = input("Anda: ")
    if query.lower() == "exit":
        break
    result = chain({"question": query, "chat_history": chat_history})
    chat_history = result["chat_history"]  # Update chat history
    print("Chatbot: ", result["answer"])"""

# Mulai Percakapan
while True:
    query = input("Pertanyaan anda: ")
    if query.lower() == "exit":
        break

    start_time = time.time()  # Tandai waktu mulai
    result = chain.invoke({"question": query, "chat_history": chat_history})
    end_time = time.time()    # Tandai waktu selesai

    response_time = end_time - start_time  # Hitung waktu respons

    chat_history = result["chat_history"]
    print("Chatbot: ", result["answer"])
    print(f"Waktu respons: {response_time:.2f} detik") 

