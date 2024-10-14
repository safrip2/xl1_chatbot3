from fastapi import FastAPI, Depends, HTTPException, status, Request, Form, Response
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from fastapi.encoders import jsonable_encoder
from starlette.responses import RedirectResponse
from starlette.middleware.sessions import SessionMiddleware
from passlib.context import CryptContext
import uvicorn
import json
from langchain.chains import ConversationalRetrievalChain
from langchain.prompts import PromptTemplate
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.chains import RetrievalQA
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from config import run_chain, chain  # Import the chain from langchain_config.py
from fastapi.responses import JSONResponse
import os
from langchain_huggingface import HuggingFaceEndpoint
from langchain.memory import ConversationBufferMemory
from langchain.prompts.chat import (ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate)
from langchain.schema import (AIMessage, HumanMessage, SystemMessage)
from transformers import AutoTokenizer
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import MessagesPlaceholder


# Constants
DB_FAISS_PATH = "db_clean" 
HF_API_TOKEN = os.environ.get("HUGGINGFACEHUB_API_TOKEN")  # Fetch the token from your environment
ENDPOINT_URL = "https://chboebbxb5ccit06.us-east-1.aws.endpoints.huggingface.cloud" #SeaLLM 7B V2.5
#ENDPOINT_URL = "https://cc5n7u5k4n8jgpqu.us-east-1.aws.endpoints.huggingface.cloud" #sailor-14b-chat (https://python.langchain.com/v0.2/docs/integrations/platforms/huggingface/) ternyata beda model beda cara memanggil

app = FastAPI()

# Configure templates
templates = Jinja2Templates(directory="templates")

app.mount("/static", StaticFiles(directory="static"), name="static")

# Enable session management
app.add_middleware(SessionMiddleware, secret_key="your_secret_key")

# Simulated database for users
fake_users_db = []

# User model
class User:
    def __init__(self, username: str, hashed_password: str):
        self.username = username
        self.hashed_password = hashed_password

# Password hashing configuration
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

# Hashed passwords for the sample users (You should hash real passwords before storing them)
hashed_password_user1 = pwd_context.hash("admin@123")  # Replace with the actual hashed password
hashed_password_user2 = pwd_context.hash("user@123")   # Replace with the actual hashed password

# Create sample users
fake_users_db.append(User(username="admin", hashed_password=hashed_password_user1))
fake_users_db.append(User(username="user", hashed_password=hashed_password_user2))

# Function to verify password
def verify_password(plain_password, hashed_password):
    return pwd_context.verify(plain_password, hashed_password)

# Function to get a user by username
def get_user(username: str):
    for user in fake_users_db:
        if user.username == username:
            return user


@app.post("/chat_response")
async def chat_response(request: Request, prompt: str = Form(...)):
    try:
        # Dapatkan riwayat percakapan dari sesi
        chat_history = request.session.get("chat_history", [])

        # Jalankan rantai LangChain dengan riwayat percakapan
        result = run_chain(chain=chain, prompt=prompt, history=chat_history)
        answer = result["answer"]
        chat_history = result["chat_history"]  # Perbarui riwayat percakapan

        # Simpan riwayat percakapan terbaru ke dalam sesi
        request.session["chat_history"] = chat_history

        # Kembalikan respons dalam format JSON
        response_data = {"answer": answer, "chat_history": chat_history}
        return JSONResponse(content=response_data)
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)

# Other routes and main function remain unchanged
@app.post("/login")
async def login(request: Request, username: str = Form(...), password: str = Form(...)):
    user = get_user(username)
    if user is None or not verify_password(password, user.hashed_password):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Bearer"},
        )

    # For simplicity, store user information in session and redirect to a protected page
    request.session['user'] = user.username
    response_data = jsonable_encoder(json.dumps({"msg": "Success",}))
    res = Response(response_data)
    return res

@app.get("/logout")
async def logout(request: Request):
    # Remove user information from session
    request.session.pop('user', None)
    return RedirectResponse(url="/")


@app.get("/chat")
async def chat(request: Request):
    user = request.session.get('user')
    if user is None:
        return RedirectResponse(url="/")

    return templates.TemplateResponse("chat.html", {"request": request, "user": user})

@app.get("/")
async def read_root(request: Request):
    user = request.session.get('user')
    if user is None:
        return templates.TemplateResponse("login.html", {"request": request})
    else:
        return RedirectResponse(url="/chat")

if __name__ == "__main__":
    uvicorn.run("app:app", host='0.0.0.0', port=8000, reload=True)