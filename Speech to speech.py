from PyPDF2 import PdfReader
from langchain_community.chat_models import ChatOpenAI
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
import speech_recognition as sr
import pyttsx3

# question = "Maran's Technical skills"
recognizer = sr.Recognizer()
with sr.Microphone() as source:
    print("Listening for your question...")
    audio = recognizer.listen(source)
    question = recognizer.recognize_google(audio)
    print(f"You asked: {question}")

def read(file):
    reader = PdfReader(file)
    text = "".join([i.extract_text() or '' for i in reader.pages])
    return text

read_ = read(file=r"D:\Maran\Resume\Maran_Resume.pdf")
# print(read_)
text_split = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=500*0.25)
doc = text_split.create_documents([read_])
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
vector_db = FAISS.from_documents(doc, embeddings)
retriever = vector_db.similarity_search(query=question, k=1)
context = "\n".join([i.page_content for i in retriever])
llm = ChatOpenAI(model='gpt-3.5-turbo', api_key="")
prompt = f"""
use the given context and answer the question.
context: {context}
question: {question}
"""
generated_text = llm.invoke(prompt)

engine = pyttsx3.init()
answer = "\n".join(generated_text.content.splitlines())
engine.say(answer)
engine.runAndWait()