from flask import Flask, request, render_template
from PyPDF2 import PdfReader
from langchain_openai import OpenAIEmbeddings, OpenAI
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain

# Replace with your actual OpenAI API key
OPENAI_API_KEY = "sk-None-hvfTell8ftyI3Q6x0dBBT3BlbkFJRR7Ru0ydE7T78hQ86U51"

# Initialize Flask app
app = Flask(__name__)

# Load and prepare the data
pdfreader = PdfReader(r'C:\cpp\coding\Vetaron\data\annavaram.pdf')

# Read text from PDF
raw_text = ''
for i, page in enumerate(pdfreader.pages):
    content = page.extract_text()
    if content:
        raw_text += content

# Split the text into smaller chunks
text_splitter = CharacterTextSplitter(
    separator="\n",
    chunk_size=800,
    chunk_overlap=200,
    length_function=len,
)
texts = text_splitter.split_text(raw_text)

# Create embeddings
embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)

# Create FAISS vector store
document_search = FAISS.from_texts(texts, embeddings)

# Load the QA chain
qa_chain = load_qa_chain(OpenAI(openai_api_key=OPENAI_API_KEY), chain_type="stuff")

def generate_answer(input_query):
    """Process the input query and return an answer."""
    # Search for relevant documents
    docs = document_search.similarity_search(input_query)
    
    # Generate the answer using the QA chain
    answer = qa_chain.run(input_documents=docs, question=input_query)
    
    return answer

@app.route("/")
def index():
    return render_template('chat.html')

@app.route("/get", methods=["GET", "POST"])
def chat():
    msg = request.form["msg"]
    response = generate_answer(msg)
    return str(response)

if __name__ == '__main__':
    app.run(host="127.0.0.1", port=8080, debug=True)
