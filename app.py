import os
from flask import Flask, request, render_template
from langchain.document_loaders import TextLoader, PyMuPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.llms import Ollama
from langchain.chains import RetrievalQA

app = Flask(__name__)
UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
vectorstore = None  # Holds our FAISS DB

@app.route("/", methods=["GET", "POST"])
def index():
    global vectorstore
    answer = ""

    if request.method == "POST":
        if "file" in request.files:
            file = request.files["file"]
            if file.filename:
                filepath = os.path.join(UPLOAD_FOLDER, file.filename)
                file.save(filepath)

                ext = os.path.splitext(filepath)[-1].lower()

                if ext == ".txt":
                    loader = TextLoader(filepath)
                elif ext == ".pdf":
                    loader = PyMuPDFLoader(filepath)
                else:
                    return "Unsupported file type", 400
            

                docs = loader.load()
                splitter = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=50)
                chunks = splitter.split_documents(docs)

                embedding = HuggingFaceEmbeddings(
                model_name="local_models/all-MiniLM-L6-v2",
                model_kwargs={"local_files_only": True}
                )
                if os.path.exists("vectorstore.index"):
                    vectorstore = FAISS.load_local("vectorstore", embedding)
                else:
                    vectorstore = FAISS.from_documents(chunks, embedding)
                    vectorstore.save_local("vectorstore")

        if "query" in request.form and vectorstore:
            query = request.form["query"]
            llm = Ollama(model="mistral")
            qa = RetrievalQA.from_chain_type(llm=llm, retriever=vectorstore.as_retriever())
            answer = qa.run(query)

    return render_template("index.html", answer=answer)

if __name__ == "__main__":
    app.run(debug=True)
