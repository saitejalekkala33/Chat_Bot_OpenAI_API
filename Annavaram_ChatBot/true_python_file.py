
from PyPDF2 import PdfReader
from langchain_openai import OpenAIEmbeddings, OpenAI
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain_community.llms import OpenAI

OPENAI_API_KEY = "sk-None-hvfTell8ftyI3Q6x0dBBT3BlbkFJRR7Ru0ydE7T78hQ86U51"

pdfreader = PdfReader(r'C:\cpp\coding\Vetaron\data\annavaram.pdf')

raw_text = ''
for i, page in enumerate(pdfreader.pages):
    content = page.extract_text()
    if content:
        raw_text += content

text_splitter = CharacterTextSplitter(
    separator="\n",
    chunk_size=800,
    chunk_overlap=200,
    length_function=len,
)
texts = text_splitter.split_text(raw_text)

embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)

document_search = FAISS.from_texts(texts, embeddings)

chain = load_qa_chain(OpenAI(openai_api_key=OPENAI_API_KEY), chain_type="stuff")

query = "What are the different types of darsanas?"
docs = document_search.similarity_search(query)

answer = chain.run(input_documents=docs, question=query)
print(answer)
