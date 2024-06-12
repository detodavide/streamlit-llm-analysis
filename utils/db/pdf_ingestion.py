import os
import joblib
from llama_parse import LlamaParse

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings.fastembed import FastEmbedEmbeddings
from langchain_community.document_loaders import UnstructuredMarkdownLoader
from langchain_community.vectorstores import Chroma

from dotenv import load_dotenv, find_dotenv
load_dotenv(find_dotenv())

def load_or_parse_data(uploaded_file):
    data_file = "./data/parsed_data.pkl"
    
    if os.path.exists(data_file):
        # Load the parsed data from the file
        parsed_data = joblib.load(data_file)
    else:
        # Perform the parsing step and store the result in llama_parse_documents
        parsingInstructionUber10k = """The provided document is a quarterly report filed by Uber Technologies, 
        Inc. with the Securities and Exchange Commission (SEC). 
        This form provides detailed financial information about the company's performance for a specific quarter. 
        It includes unaudited financial statements, management discussion and analysis, and other relevant disclosures required by the SEC.
        It contains many tables.
        Try to be precise while answering the questions"""
        
        parser = LlamaParse(api_key=os.getenv("LLAMA_PARSE_API_KEY"), result_type="markdown", parsing_instruction=parsingInstructionUber10k)
        
        # Load data from the uploaded PDF file
        with open(uploaded_file.name, "wb") as f:
            f.write(uploaded_file.getbuffer())
        llama_parse_documents = parser.load_data(uploaded_file.name)
        
        # Save the parsed data to a file
        joblib.dump(llama_parse_documents, data_file)
        
        # Set the parsed data to the variable
        parsed_data = llama_parse_documents
    
    return parsed_data

# Create vector database
def create_vector_database(llama_parse_documents):
    """
    Creates a vector database using document loaders and embeddings.

    This function loads urls,
    splits the loaded documents into chunks, transforms them into embeddings using FastEmbedEmbeddings,
    and finally persists the embeddings into a Chroma vector database.

    """
    # Call the function to either load or parse the data
    print(llama_parse_documents[0].text[:100])
    
    with open('data/output.md', 'a', encoding='utf-8') as f:  # Open the file in append mode ('a')
        for doc in llama_parse_documents:
            f.write(doc.text + '\n')
    
    markdown_path = "data/output.md"
    loader = UnstructuredMarkdownLoader(markdown_path)

    documents = loader.load()
    # Split loaded documents into chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=50)
    docs = text_splitter.split_documents(documents)
    print(f"length of documents loaded: {len(documents)}")
    print(f"total number of document chunks generated :{len(docs)}")

    # Initialize Embeddings
    embeddings = FastEmbedEmbeddings(model_name="BAAI/bge-base-en-v1.5")

    # Create and persist a Chroma vector database from the chunked documents in batches
    batch_size = 50
    for i in range(0, len(docs), batch_size):
        batch_docs = docs[i:i+batch_size]
        vectorstore = Chroma.from_documents(
            documents=batch_docs,
            embedding=embeddings,
            persist_directory="./chromadb1",
            collection_name="full_documents"
        )
    
    return vectorstore, embeddings

def query_vectorstore(vectorstore: Chroma):
    query = "what is the agenda of Financial Statements for 2022?"
    found_docs = vectorstore.similarity_search(query, k=3)
    return found_docs

# Example usage
# llama_parse_documents should be a list of Document objects with a 'text' attribute
# vectorstore, embeddings = create_vector_database(llama_parse_documents)
# found_docs = query_vectorstore(vectorstore)
# for doc in found_docs:
#     print(doc.text[:100])
