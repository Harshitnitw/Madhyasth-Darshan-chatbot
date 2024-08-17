import os
import pickle
import streamlit as st
from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader
from typing import List
from langchain.schema import BaseRetriever, Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from typing import List
from sentence_transformers import SentenceTransformer
from langchain_groq import ChatGroq

from dotenv import load_dotenv
load_dotenv()

llm = ChatGroq(groq_api_key=os.getenv("GROQ_API_KEY"), model_name="llama-3.1-70b-Versatile")

if "store" not in st.session_state:
    st.session_state["store"] = {}


def get_or_create_vectorstore(cache_folder="./vectorstore_cache"):
    cache_path = os.path.join(cache_folder, "vectorstore.pkl")    
    if os.path.exists(cache_path):
        print(f"Loading vectorstore from cache: {cache_path}")
        with open(cache_path, 'rb') as f:
            vectorstore = pickle.load(f)
        return vectorstore
    else:
        documents = []
        books=[
            {
                "system_prompt":"Introduction to Madhyasth Darshan philosophy is provided in the form of a book ‘jeevan vidya - ek parichay’ by the propounder, A. Nagraj. It may contain human error as the document is edited manually. A. Nagraj was born in the year 1920 in hassan district in karnataka and was from a family of vedic scholars and wrote the book during the 1980s, because of these and some other reasons, his hindi language style and dialect is different from the style and dialect commonly spoken these days making it difficult for most people to understand. To make the interpretation of his writings more clear for the LLM application, later on I would also add some documents and PDFs written by the students of the philosopher, which are in day to day english or hindi dialect. Remember that the students have yet not reached the expertise in the philosophy and have delivered their interpretations only, so if there is contradiction then the final truth should be considered of the original propounder only.",
                "pdf_docx_paths":["Jeevan Vidya - Ek Parichay.docx"]
            },
            {
                "system_prompt":"This is the english translation of the book ‘Jeevan Vidya - ek parichay” shared before. The translation has been done from the propounder’s peculiar hindi dialect to day to day english dialect by a student of Madhyasth Darshan philosophy who has not yet reached the expertise, and has delivered his interpretations only, so if there is contradiction then the final truth should be considered of the original propounder only. The translation is provided to aid the LLM application to understand the dialect of Madhyasth Darshan’s original propounder, A. Nagraj. This will help the LLM to interpret the dialect and language style of the books written originally by the propounder. Note that in the translation, hindi words are also there sometimes and they may have encoding issues, try to understand them in the text’s context or ignore them if not interpretable.",
                "pdf_docx_paths":["jeevan vidya an introduction (english translation of 'jeevan vidya - ek parichay').pdf"]
            },
            {
                "system_prompt":"This is a transcription of one to one discussions between the propounder A. Nagraj and disciples who have just been introduced to Madhyasth Darshan philosophy. So here the propounder has tried to clarify the queries of the students in somewhat easier, day to day language. Use these conversations to understand his dialect, his ideas, and the content of the philosophy. For these documents, PyMuPDF has been used to extract content from some old OCR files, because of which at some places, the text has got misrepresented and at places abrupt line breaks have come, also at some places it is putting wrong alphabets. which the LLM would have to interpret, also numbers like 1 written in hindi are changed to 4 at places during extraction, there maybe more such examples, so interpret the content by comparing the information from other uploaded documents.",
                "pdf_docx_paths":["Nagraj ji ke saath samvaad 1.docx","Nagraj ji ke saath samvaad 2.docx"]
            },
            {
                "system_prompt":"The educational content of Madhyasth Darshan philosophy is provided in the form of books by the propounder, A. Nagraj. They may contain human error as the document is edited manually. A. Nagraj was born in the year 1920 in hassan district in karnataka and was from a family of vedic scholars and wrote the book during the 1980s, because of these and some other reasons, his hindi language style and dialect is different from the style and dialect commonly spoken these days making it difficult for most people to understand. To make the interpretation of his writings more clear for the LLM application, I have added before and later on would also add some documents and PDFs written by the students of the philosopher, which are in day to day english or hindi dialect. Remember that the students have yet not reached the expertise in the philosophy and have delivered their interpretations only, so if there is contradiction then the final truth should be considered of the original propounder only. ",
                "pdf_docx_paths":["samadhanatmak bhoutikvad.docx","anubhavatmak adhyatmvad.docx","avartansheel arthshastra.docx","manviya samvidhan.docx"]
            },
            {
                "system_prompt":"These documents are written by students of Madhyasth Darshan philosophy who have not yet reached the expertise, and have delivered his interpretations only, so if there is contradiction then the final truth should be considered of the original propounder only. The translation is provided to aid the LLM application to understand the dialect of Madhyasth Darshan’s original propounder, A. Nagraj. This will help the LLM to interpret the dialect and language style of the books written originally by the propounder. Note that in the translation, hindi words are also there sometimes and they may have encoding issues, try to understand them in the text’s context or ignore them if not interpretable.",
                "pdf_docx_paths":["A. Jeevan Vidya Workshop - a companion reader.pdf","B. About Co-existential Philosophy.pdf","C. What is Madhyasth Darshan - small intro.pdf","D. Study Method in Madhyasth Darshan - small intro.pdf"]
            },
            {
                "system_prompt":"These documents are written by students of Madhyasth Darshan philosophy who have not yet reached the expertise, and have delivered his interpretations only, so if there is contradiction then the final truth should be considered of the original propounder only. The translation is provided to aid the LLM application to understand the dialect of Madhyasth Darshan’s original propounder, A. Nagraj. This will help the LLM to interpret the dialect and language style of the books written originally by the propounder. Note that in the translation, hindi words are also there sometimes and they may have encoding issues, try to understand them in the text’s context or ignore them if not interpretable. Note that these documents were mostly written in hindi and were having encoding issues, so they were converted to OCR and then to docx in unicode. I used convert_from_path() from pdf2image module to convert each page of the PDF to an image, and then used image_to_string() from pytesseract module to extract text from each image, and then cleaned the text to allow only abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789 \n\r\t,.'\";:!?|–—।॥ set of characters, so keep this in memory while deciphering from these documents that other characters are absent, which you may have to estimate to understand the context. Also, some charts were there in the original docs, only text has been extracted from it during conversion, so keep that in memory incase of finding abrupt texts.",
                "pdf_docx_paths":["मध्यस्थ दर्शन परिचय - श्रीराम.docx","अध्ययन सम्बंधित सामान्य स्पष्टीकरण.docx"]
            },
        ]

        def load_document(file_path, system_prompt):
            _, extension = os.path.splitext(file_path)
            if extension == ".pdf":
                loader = PyPDFLoader(file_path)
            elif extension in [".docx", ".doc"]:
                loader = Docx2txtLoader(file_path)
            else:
                print(f"Unsupported file format: {extension}")
                return []

            docs = loader.load()

            for doc in docs:
                doc.metadata["system_prompt"] = system_prompt

            return docs

        def split_documents(documents):
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
            splits = []
            
            for doc in documents:
                doc_splits = text_splitter.split_documents([doc])
                for split in doc_splits:
                    # Ensure each split has the same metadata as its parent document
                    split.metadata = doc.metadata or {}
                    splits.append(split)
            
            return splits

        def get_or_create_embeddings(model_name, cache_folder="./model_cache"):
            # Create the cache folder if it doesn't exist
            os.makedirs(cache_folder, exist_ok=True)
            
            # Check if the model is already cached
            model_path = os.path.join(cache_folder, model_name.replace("/", "_"))
            
            if os.path.exists(model_path):
                print(f"Loading model from cache: {model_path}")
                model = SentenceTransformer(model_path)
            else:
                print(f"Downloading model: {model_name}")
                model = SentenceTransformer(model_name)
                # Save the model to cache
                model.save(model_path)
                print(f"Model saved to cache: {model_path}")
            
            # Create HuggingFaceEmbeddings instance
            embeddings = HuggingFaceEmbeddings(model_name=model_path)
            
            return embeddings

        for collection in books:
            system_prompt=collection["system_prompt"]
            for path in collection["pdf_docx_paths"]:
                path = path.strip()
                path = "books/" + path
                if os.path.exists(path):
                    docs = load_document(path, system_prompt)
                    if docs:
                        documents.extend(docs)
                        print(f"Loaded {len(docs)} documents from {path}.")
                    else:
                        print(f"No documents loaded from {path}.")
                else:
                    print(f"File {path} not found.")

        # Debugging: Print the total number of documents loaded
        print(f"Total documents loaded: {len(documents)}")

        # Ensure there are documents before attempting to create or add to the vector store
        if documents:
            splits=split_documents(documents)
            model_name = "AkshitaS/bhasha-embed-v0"
            embeddings = get_or_create_embeddings(model_name)

            vectorstore = FAISS.from_documents(documents=splits, embedding=embeddings)
            
            print("System prompt and documents added successfully!")
            os.makedirs(cache_folder, exist_ok=True)
            with open(cache_path, 'wb') as f:
                pickle.dump(vectorstore, f)
            print(f"Vectorstore saved to cache: {cache_path}")
            return vectorstore
        else:
            print("No valid documents were found. Vector store creation skipped.")


class CustomRetriever(BaseRetriever):
    
    def get_relevant_documents(self, query: str) -> List[Document]:
        if "retriever" not in st.session_state:
            st.session_state["retriever"] = get_or_create_vectorstore().as_retriever()
        docs = st.session_state["retriever"].get_relevant_documents(query)
        for doc in docs:
            doc.page_content = f"System Prompt: {doc.metadata['system_prompt']}\n\nContent: {doc.page_content}"
        return docs

    async def aget_relevant_documents(self, query: str) -> List[Document]:
        return self.get_relevant_documents(query)

   
session_id = "default_session"

custom_retriever = CustomRetriever()


system_prompt=(
    """
    You are an AI assistant specializing in the Madhyasth darshanphilosophy propounded by A. Nagraj. Your primary sources are documents written by A. Nagraj, along with interpretations and translations by his students. Follow these guidelines:

        Understanding language style and dialect: A. Nagraj's Hindi dialect is different from contemporary spoken Hindi. When answering, simplify the language to make it more accessible to modern Hindi speakers. Explain any uncommon words or phrases.
        Understanding context: Use the writings of the students to understand the dialect of A. Nagraj’s philosophical writings, to make your interpretation of his original ideas more clear. 
        Source priority: Prioritize information from A. Nagraj's original writings over student interpretations. If there's a contradiction, defer to A. Nagraj's original teachings.
        Terminology explanation: When using terms specific to Madhyasth Darshan, provide brief explanations or contemporary equivalents to aid understanding.
        Clarity in explanation: If using any unconventional words or if conventional words are used with new meanings in the philosophy, explicitly explain these to the user.
        Error handling: Be aware that some words may appear misspelled, gibberish, absurd or out of context due to encoding issues, OCR errors or manual editing errors, try to interpret them in the text’s context or ignore them if not interpretable.
        Answer the user's question based on the retrieved documents. Do not speculate or infer information that is not directly stated in the context. If you're unsure or if the information needed to answer the question is not present in the context, respond saying that the provided context does not contain sufficient information to answer the question.

    Use the following context to answer questions, keeping in mind the specific instructions associated with each piece of information. Answer the question using ONLY the information provided in the context below. Do not use any external knowledge or make assumptions beyond what is explicitly stated in the given context.:\n\n
        {context}
    """
)


qa_prompt = ChatPromptTemplate.from_messages(
    [
        ("system",system_prompt),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{input}")
    ]
)

contextualize_q_system_prompt=(
    "Given a chat history and the latest user question"
    "which might reference context in the chat history, "
    "formulate a standalone question which can be understood "
    "without the chat history. Do NOT answer the question, "
    "just reformulate it if needed and otherwise return it as is."
)
contextualize_q_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", contextualize_q_system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ]
)

history_aware_retriever = create_history_aware_retriever(llm, custom_retriever,contextualize_q_prompt)
question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)
rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)

def get_session_history(session_id: str) -> BaseChatMessageHistory:
    if session_id not in st.session_state["store"]:
        st.session_state["store"][session_id] = ChatMessageHistory()
    return st.session_state["store"][session_id]

conversational_rag_chain = RunnableWithMessageHistory(
    rag_chain,
    get_session_history,
    input_messages_key="input",
    history_messages_key="chat_history",
    output_messages_key="answer"
)

st.title("Chatbot for madhyasth darshan")

user_input = st.text_input("Your question:")
if user_input:
    session_history = get_session_history(session_id)
    response = conversational_rag_chain.invoke(
        {"input": user_input},
        config={
            "configurable": {"session_id": session_id}
        },
    )
    st.write("Assistant:", response['answer'])
