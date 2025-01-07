# 🦙 Interactive Text Generation with Llama 3.2 Model Using LangChain and Ollama 🚀

This project demonstrates how to build an **interactive text generation system** using **Meta's Llama 3.2 model** with **LangChain** and **Ollama integration**. Additionally, it features retrieval-augmented QA capabilities using **FAISS vector databases** for handling documents like PDFs. This setup is ideal for custom question-answering, joke generation, and general text interaction tasks.

---

## 🌟 Key Features

- **Llama 3.2 Model Integration**: Advanced language model powered by LangChain and Ollama.
- **PDF Document Processing**: Load, split, and embed PDFs into FAISS for efficient retrieval-based QA.
- **FAISS Vector Store**: Persistent vector database for storing and retrieving document embeddings.
- **Interactive Text Generation**: Direct interaction with the model for generating jokes or answering questions.
- **LangChain-Orchestrated QA**: Use RetrievalQA for combining language model reasoning with document knowledge.

---

## 🛠️ Installation

### Install Required Libraries
Run the following commands to install the dependencies:
```bash
pip install langchain langchain-community sentence-transformers faiss-gpu pypdf langchain_ollama
Additional Setup
Install Colab Xterm for running interactive terminal commands:

bash
Copy code
pip install colab-xterm
Log in to Hugging Face:

python
Copy code
from huggingface_hub import login
login("your_huggingface_token")
🚀 Project Workflow
1️⃣ Process PDF Document
Load a sample PDF document and split it into chunks:

python
Copy code
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import CharacterTextSplitter

# Load document
loader = PyPDFLoader("/content/sample_document.pdf")
documents = loader.load()

# Split document into chunks
text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=30, separator="\n")
docs = text_splitter.split_documents(documents=documents)
2️⃣ Create FAISS Vector Store
Embed the document chunks and save the embeddings using FAISS:

python
Copy code
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS

# Load HuggingFace embeddings
embedding_model_name = "sentence-transformers/all-mpnet-base-v2"
model_kwargs = {"device": "cuda"}
embeddings = HuggingFaceEmbeddings(model_name=embedding_model_name, model_kwargs=model_kwargs)

# Create FAISS vector store
vectorstore = FAISS.from_documents(docs, embeddings)

# Save vector store locally
vectorstore.save_local("faiss_index_")

# Load vector store
persisted_vectorstore = FAISS.load_local("faiss_index_", embeddings, allow_dangerous_deserialization=True)

# Create retriever
retriever = persisted_vectorstore.as_retriever()
3️⃣ Load Llama Model with Ollama
Use Ollama for efficient loading of the Llama model:

python
Copy code
from langchain_community.llms import Ollama

llm = Ollama(model="llama3.2")
response = llm.invoke("Tell me a joke")
print(response)
🧠 Retrieval-Based Question Answering
Set up RetrievalQA with LangChain
Use LangChain’s RetrievalQA to query the document using Llama 3.2:

python
Copy code
from langchain.chains import RetrievalQA

qa = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=retriever)

# Interactive QA loop
while True:
    query = input("Type your query (or type 'Exit' to quit): \n")
    if query.lower() == "exit":
        break
    result = qa.run(query)
    print(result)
Example Queries
Input: What is this document about?
Output: This document appears to be a sample PDF document created for testing purposes, specifically to test document reading and indexing functions in natural language processing workflows.

Input: What are its contents?
Output: The document's content covers topics such as Technology, Health, and Environment.

🎯 Customization Options
Change Embedding Model: Replace "sentence-transformers/all-mpnet-base-v2" with another Hugging Face embedding model.
Fine-Tune Llama 3.2: Modify the Llama model using your dataset for domain-specific tasks.
Increase Chunk Size: Adjust chunk_size and chunk_overlap in CharacterTextSplitter to handle larger or smaller sections of the document.
Add Custom Chains: Extend LangChain’s functionality with custom pipelines for additional reasoning.
📦 Model Persistence
Save Vector Store and Model
python
Copy code
# Save FAISS vector store
vectorstore.save_local("faiss_index_")

# Save fine-tuned model (if applicable)
model.save_pretrained("/content/drive/MyDrive/fine_tuned_model")
tokenizer.save_pretrained("/content/drive/MyDrive/fine_tuned_model")
Load Saved Assets
python
Copy code
# Load vector store
persisted_vectorstore = FAISS.load_local("faiss_index_", embeddings, allow_dangerous_deserialization=True)

# Load fine-tuned model
from transformers import AutoModelForCausalLM, AutoTokenizer
model = AutoModelForCausalLM.from_pretrained("/content/drive/MyDrive/fine_tuned_model")
tokenizer = AutoTokenizer.from_pretrained("/content/drive/MyDrive/fine_tuned_model")
📚 Example Interaction
Joke Generation

Input: Tell me a joke.
Output: Here's one: What do you call a fake noodle? An impasta.
Question Answering

Input: What is this document about?
Output: This document appears to be a sample PDF document created for testing purposes.
