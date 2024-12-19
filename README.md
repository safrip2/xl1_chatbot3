# xl1_chatbot2

# Chatbot LLM-RAG with SeaLLM 7B V2.5

This repository contains the code and assets for a chatbot project leveraging Large Language Model (LLM) and Retrieval-Augmented Generation (RAG) techniques. The chatbot is designed to assist users by providing accurate and contextually relevant responses based on indexed PDF documents. 

## Key Features
- **LLM Model**: Utilizes the SeaLLM 7B V2.5 model hosted on Huggingface Inference Endpoint.
- **RAG Implementation**: Enhances the chatbot's response generation by integrating a vector database for efficient document retrieval.
- **Performance Evaluation**: Implements RAGAS metrics, including context recall, to measure and improve chatbot performance.
- **Custom Front End**: Built with HTML and JavaScript, providing a user-friendly interface.

## Repository Structure
```
.
├── data
│   └── *.pdf                # Indexed PDF documents used for RAG
├── db_clean
│   └── *.faiss              # FAISS vector database storing model embeddings (clean data approach)
│   └── *.pkl                # Metadata of documents (clean data approach)
├── db_raw
│   └── *.faiss              # FAISS vector database storing model embeddings (raw data approach)
│   └── *.pkl                # Metadata of documents (raw data approach)
├── static
│   └── bot-avatar.png       # Chatbot avatar image
│   └── profile-avatar.png   # User profile avatar image
├── templates
│   └── chat.html            # Front end for chatbot interaction
│   └── login.html           # User login interface
├── visualization
│   └── chunk_viz.py         # Visualizing chunk generated for chatbot
├── app.py                   # Main application script
├── langchain_config.py      # Configuration for LangChain framework
├── ingest.py                # Script for ingesting and indexing data
└── README.md                # Project documentation (this file)
```

## How It Works
1. **Data Preparation**:
   - PDF documents are ingested and indexed using FAISS, creating vector embeddings for efficient similarity search.
2. **Model Integration**:
   - The chatbot is powered by SeaLLM 7B V2.5, accessed via API from the Huggingface Inference Endpoint.
3. **Query Handling**:
   - User queries are processed through the LangChain framework, retrieving the most relevant context from the FAISS database.
4. **Response Generation**:
   - The LLM generates a response based on the retrieved context, improving relevance and accuracy.

## Dependencies
- Python 3.9+
- LangChain
- FAISS
- Huggingface Hub

Additional dependencies are listed in `requirements.txt`.

## Installation
1. Clone this repository:
   ```bash
   git clone https://github.com/safrip2/xl1_chatbot2.git
   cd xl1_chatbot2.git
   ```
2. Install the required Python packages:
   ```bash
   pip install -r requirements.txt
   ```
3. Configure the Huggingface API key in `app.py`.
4. Run the ingestion script to index PDF documents:
   ```bash
   python ingest.py
   ```

## Usage
1. Start the application:
   ```bash
   uvicorn app:app
   ```
2. Access the chatbot interface at `http://localhost:5000/chat` or `http://127.0.0.1:8000`.

## Evaluation and Optimization
- The chatbot's performance is evaluated using RAGAS metrics, focusing on context recall and other relevant measures.
- Observations indicate the chatbot's preference for responding in English despite optimizations for Indonesian. Investigations into embedding mismatches, tokenization, or model behavior are ongoing.

## Known Issues
- The chatbot may default to English responses for certain queries. Future iterations aim to resolve this issue by adjusting embeddings or fine-tuning model settings.

## Portfolio Highlights
- Demonstrates practical application of LLM and RAG techniques.
- Showcases integration with FAISS and LangChain for advanced retrieval and generation capabilities.
- Provides insights into chatbot evaluation using industry-standard metrics like RAGAS.

## Future Work
- Expand language-specific optimizations for Indonesian.
- Integrate additional evaluation metrics for enhanced performance tracking.
- Improve front-end design for better user experience.

## License
This project is licensed under the MIT License. See the `LICENSE` file for details.
