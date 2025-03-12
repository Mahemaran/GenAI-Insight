import streamlit as st
import pandas as pd
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.llms.huggingface_hub import HuggingFaceHub
from langchain_community.vectorstores import FAISS
from langchain_openai import ChatOpenAI
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_groq import ChatGroq
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from transformers import AutoModelForCausalLM, AutoTokenizer

st.set_page_config(page_title="QA_Chatbot", page_icon="📝", layout="centered")

# Helper Functions for File Reading
def read_pdf(file):
    reader = PdfReader(file)
    text = "".join([page.extract_text() or "" for page in reader.pages])
    return text

def read_csv(file):
    df = pd.read_csv(file)
    return df.to_string()

def read_excel(file, sheet_name):
    df = pd.read_excel(file, sheet_name=sheet_name, engine='openpyxl')
    return df.to_string()

def read_text(file):
    return file.read().decode("utf-8")

# File Handling Function
def load_files(uploaded_files, file_type, sheet_name=None):
    combined_text = ""
    for uploaded_file in uploaded_files:
        if file_type == "Text":
            combined_text += read_text(uploaded_file)
        elif file_type == "PDF":
            combined_text += read_pdf(uploaded_file)
        elif file_type == "CSV":
            combined_text += read_csv(uploaded_file)
        elif file_type == "Excel":
            combined_text += read_excel(uploaded_file, sheet_name)
    return combined_text

# Streamlit UI
st.title("📝 AI-Powered Question Answering System 🤖")

# File Upload Section
uploaded_files = st.file_uploader("Upload Files (Text, PDF, CSV, Excel)", type=["txt", "pdf", "csv", "xlsx"],
                                  accept_multiple_files=True)
if not uploaded_files:
    st.warning("Please upload file")
# File Type Selection
with st.sidebar:
    file_type = st.sidebar.selectbox("Select File Type", ["Text", "PDF", "CSV", "Excel"])

# Excel Sheet Selection
    sheet_name = None
    if file_type == "Excel":
        sheet_name = st.sidebar.text_input("Enter Sheet Name (for Excel files)", "Sheet1")
    if file_type == "CSV":
        sheet_name = st.sidebar.text_input("Enter Sheet Name (for CSV files)", "Sheet1")

# Model Selection
with st.sidebar:
    model_name = st.sidebar.selectbox(
    "Select LLM Model",
    ["gpt-3.5-turbo (OpenAI)", "gemini-pro (Google)", "tiiuae/falcon-7b-instruct (HuggingFace)", "llama3-8b-8192 (Groq)", "gpt2"])

# API Key Input
with st.sidebar:
    api_key = st.text_input("OpenAI API Key", key="chatbot_api_key", type="password")
    if not api_key:
        st.warning("please enter your API key 🔑")

if uploaded_files:
    question = st.text_input("ask your question 😊")
    st.write(question)
    if not question:
        st.error("Please ask the question.")
    else:
        try:
            # Load and combine file data
            combined_text = load_files(uploaded_files, file_type, sheet_name)
            # Text Splitting
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
            documents = text_splitter.create_documents([combined_text])
            embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
            db = FAISS.from_documents(documents, embedding=embeddings)
            # with st.sidebar:
            #     k_value = st.number_input("top matches", min_value=1, value=5)
            #     k_value = int(k_value)
            # if not k_value:
            #     st.warning("please enter top matches")
            # else:
            try:
                retriever = db.similarity_search(question, k=1)
                context_text = "\n".join([doc.page_content for doc in retriever])
                retriever = db.as_retriever()
                with st.spinner("Fetching comments...😊"):
                    # Model Selection
                    if "gpt-3.5" in model_name:
                        llm = ChatOpenAI(api_key=api_key, model="gpt-3.5-turbo")
                    elif "gemini-pro" in model_name:
                        llm = ChatGoogleGenerativeAI(google_api_key=api_key, model="gemini-pro")
                    elif "falcon" in model_name:
                        llm = HuggingFaceHub(repo_id="tiiuae/falcon-7b-instruct", huggingfacehub_api_token=api_key,
                                             model_kwargs={"temperature": 0.9, "max_new_tokens": 50, "top_p": 0.9})

                    elif "Groq" in model_name:
                        llm = ChatGroq(groq_api_key=api_key, model_name="llama3-8b-8192")
                    elif "gpt2" in model_name:
                        model_name = "gpt2"
                        tokenizer = AutoTokenizer.from_pretrained(model_name)
                        model = AutoModelForCausalLM.from_pretrained(model_name)

                        # Tokenize the prompt
                        inputs = tokenizer(context_text, return_tensors="pt", truncation=True, max_length=512)

                        # Generate a response
                        outputs = model.generate(
                            **inputs,
                            max_new_tokens=50,  # Control response length
                            temperature=0.8,  # Control randomness (lower = more deterministic)
                            top_p=0.85,  # Nucleus sampling
                            do_sample=True,  # Enable sampling
                            repetition_penalty=1.2,  # Penalize repetition
                            pad_token_id=tokenizer.eos_token_id,  # Padding token
                            eos_token_id=tokenizer.eos_token_id  # End-of-sequence token
                        )
                        # Decode and print the response
                        response_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
                    else:
                        st.error("Selected model is not available.")
                        st.stop()
                    # Prompt Template
                    system_prompt = (""" Use the given context to answer the question. 
                        If you don't know the answer, say you don't know. 
                        Use three sentence maximum and keep the answer concise. 
                        <context>
                        {context} 
                        </context>
                        question: {input} """)

                    # prompt = system_prompt.format(context=context_text, question=question)
                    prompt = ChatPromptTemplate.from_template(system_prompt)
                    # llm = create_retrieval_chain(context_text, question_answer_chain)
                    # Invoke the model
                    st.write("Generating response...")
                    if "gpt2" in model_name:
                        response = response_text
                    else:
                        question_answer_chain = create_stuff_documents_chain(llm, prompt)
                        llm = create_retrieval_chain(retriever, question_answer_chain)
                        response = llm.invoke({"input": question})
                        # response = llm.invoke(question)
                    # Display Answer
                    st.subheader("✅ Answer:")
                    if "falcon" in model_name:
                        st.write(response)
                    elif "gpt2" in model_name:
                        st.write(
                            "**Note: The answer may be less accurate when using GPT-2. No API key is required for this model...** \n")
                        st.write(str(response))
                    else:
                        st.write(response["answer"])
            except ValueError as e:
                st.error(f"Please enter a valid number for the top matches (k): {e}")
        except Exception as e:
            st.error(f"An error occurred: {e}")