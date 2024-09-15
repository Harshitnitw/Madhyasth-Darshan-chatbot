# Steps to run this chatbot:
1. Open the given directory in github codespace
2. In terminal write the following to install the necessary packages:
 
        pip install -r requirements.txt
3. In the root directory create a folder named vectorstore_cache
4. In .env file (create it in the root directory if not present), save your Groq API key like this:
   
        GROQ_API_KEY=”groqApiKeyValueCopiedFromGroqWebsiteAfterCreatingAccount”
5. Download the vectorstore with the following command in terminal:
 
        gdown https://drive.google.com/uc?id=1cwzPmElORHMN7wHdFvPTD_l33drB-ckI -O vectorstore_cache/vectorstore.pkl
6. Write the following command in terminal to start the chatbot:
 
        streamlit run main.py


# Report on creation of Madhyasth Darshan chatbot

1. Purpose: A chatbot for Madhyasth Darshan philosophy using Streamlit.

2. Key Components:

- LLM: llama-3.1-70b-Versatile open source model using Groq API

- Embeddings: HuggingFaceEmbeddings with "AkshitaS/bhasha-embed-v0" open source model (especially designed for hindi and romanised hindi

- Vector Store: FAISS (cpu)

- Document Loaders: PyPDFLoader and Docx2txtLoader

- Text Splitter: RecursiveCharacterTextSplitter (chunk size = 500, chunk overlap=50)

- Currently using parichay (both Hindi and English), samvaad 1, 2, arthshastha, adhyatmavad, samvidhan, Bhautikvaad and 6 books and documents written by students.

3. Functionality:
   
- Checks if vectorstore present in file system then it loads it, else it creates and saves a new vector store with the following steps:
 
- Checks if embeddings are present in the file system then it loads it, else  it downloads embeddings from huggingface and saves it
 
- Loads PDFs and documents using document loaders
 
- Splits the loaded documents into an array of elements of 500 characters size
 
- To each spilt, we add a metadata, which is a system prompt we specifically designed for each uploaded book so that we can instruct the LLM about how to use the respective book in proper context
 
- Use FAISS vector database with our saved embeddings to create a vectorstore from the documents
 
- Create a custom retriever to use the vectorstore as retriever, we get the relevant documents for the given user query and return the docs array file with each element’s corresponding metadata, which is the specific system prompt for each uploaded book, appended to the element’s page content, adding the metadata with the name of system prompt for the LLM to interpret.
 
- Create a question answer prompt (qa_prompt) which has a system prompt for the guidelines of how the chatbot should behave and also with the <context> variable for the retrieved split documents, messages placeholder for chat history and human input.
 
- Create a contextualized q prompt whose role is not to answer the question but reformulate the current question and the previous messages to form a standalone question which can carry the previous context as well.
 
- Create history aware retriever and question answer chain to serve in creating retrieval chain (rag chain)
 
- Invoke the chain with every user input question.
 
4. User Interface:
   - Streamlit-based web interface for user input and displaying responses

5. Key Features:
   - Caches models and vector store for improved performance
   - Handles multiple document types (PDF, DOCX)
   - Incorporates system prompts for better context understanding
   - Uses a conversational RAG (Retrieval-Augmented Generation) chain

6. Notable Aspects:
   - Handles potential OCR and encoding issues in documents
   - Prioritizes original teachings over student interpretations
   - Provides simplified explanations of complex philosophical concepts

This code creates a specialized chatbot that can answer questions about Madhyasth Darshan philosophy using a combination of language models, document retrieval, and conversation history management.

