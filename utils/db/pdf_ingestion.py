import os
import joblib
from llama_parse import LlamaParse
import streamlit as st

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings.fastembed import FastEmbedEmbeddings
from langchain_community.document_loaders import UnstructuredMarkdownLoader
from langchain_community.vectorstores import Chroma
from llm.llm_model import get_llm, LLMConfig
from langchain.prompts import PromptTemplate
from langchain.chains.retrieval_qa.base import RetrievalQA
from langchain_community.vectorstores import Qdrant
from qdrant_client import QdrantClient


from dotenv import load_dotenv, find_dotenv
load_dotenv(find_dotenv())

def load_or_parse_data(uploaded_file):

    file_name: str = uploaded_file.name.split('.')[0]
    data_file = f"./data/{file_name}.pkl"
    
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
        
        parser = LlamaParse(api_key=st.secrets["LLAMA_PARSE_API_KEY"], result_type="markdown", parsing_instruction=parsingInstructionUber10k)
        
        # Load data from the uploaded PDF file
        with open(f"pdfs/{uploaded_file.name}", "wb") as f:
            f.write(uploaded_file.getbuffer())
        llama_parse_documents = parser.load_data(f"pdfs/{uploaded_file.name}")
        
        # Save the parsed data to a file
        joblib.dump(llama_parse_documents, data_file)
        
        # Set the parsed data to the variable
        parsed_data = llama_parse_documents
    
    return parsed_data

# Create vector database
def create_vector_database(llama_parse_documents, uploaded_file):
    """
    Creates a vector database using document loaders and embeddings.

    This function loads urls,
    splits the loaded documents into chunks, transforms them into embeddings using FastEmbedEmbeddings,
    and finally persists the embeddings into a Chroma vector database.

    """
    markdown_path = f"data/{uploaded_file.name.split('.')[0]}.md"
    file_name: str = uploaded_file.name.split('.')[0]

    with open(markdown_path, 'w', encoding='utf-8') as f:  # Open the file in append mode ('a')
        for doc in llama_parse_documents:
            f.write(doc.text + '\n')
    
    loader = UnstructuredMarkdownLoader(markdown_path)

    documents = loader.load()
    # Split loaded documents into chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=100)
    docs = text_splitter.split_documents(documents)
    print(f"length of documents loaded: {len(documents)}")
    print(f"total number of document chunks generated :{len(docs)}")

    # Initialize Embeddings
    embeddings = FastEmbedEmbeddings()

    # IF it crash use the batch add chunks
    # vectorstore = Chroma(persist_directory="./chromadb1", collection_name=f"{file_name}", embedding_function=embeddings)

    # batch_size = 50
    # for i in range(0, len(docs), batch_size):
    #     print(f"Adding batch of chunks from {i} to {i+batch_size}")
    #     batch_docs = docs[i:i+batch_size]
    #     vectorstore.add_documents(documents=batch_docs)

    # vectorstore = Chroma.from_documents(
    #         documents=docs,
    #         embedding=embeddings,
    #         persist_directory="./chromadb1",
    #         collection_name=f"{file_name}"
    #     )
    
    # vectorstore.persist()

    qdrant = Qdrant.from_documents(
        documents=docs,
        embedding=embeddings,
        url=st.secrets["QDRANT_URL"],
        collection_name=file_name,
        api_key=st.secrets["QDRANT_API_KEY"]
    )

def query_vectorstore(query, uploaded_file):
    
    file_name: str = uploaded_file.name.split('.')[0]

    embeddings = FastEmbedEmbeddings()
    # vectorstore = Chroma(embedding_function=embeddings,
    #                 persist_directory="./chromadb1",
    #                 collection_name=f"{file_name}"
    # )
    client = QdrantClient(api_key=st.secrets["QDRANT_API_KEY"], url=st.secrets["QDRANT_URL"],)
    vectorstore = Qdrant(client=client, embeddings=embeddings, collection_name=file_name)

    config=LLMConfig(llm_provider="Groq")
    llm = get_llm(config)

    retriever=vectorstore.as_retriever(search_kwargs={'k': 3})

    prompt = PromptTemplate(
        template="""<|begin_of_text|><|start_header_id|>system<|end_header_id|>
            You are a helpful assistant that is really good on retrieving data from docuemnts.

            <|eot_id|><|start_header_id|>user<|end_header_id|>
            Use the following pieces of information to answer the user's question.
            If you don't know the answer, just say that you don't know, don't try to make up an answer.

            Context: {context}
            Question: {question}

            Only return the helpful answer below and nothing else, and contextualize the answer.
            Helpful answer:
            <|eot_id|>
            <|start_header_id|>assistant<|end_header_id|>
            """,
            input_variables=["context", "question"],
    )
    
    qa = RetrievalQA.from_chain_type(llm=llm,
                               chain_type="stuff",
                               retriever=retriever,
                               return_source_documents=True,
                               chain_type_kwargs={"prompt": prompt})

    response = qa.invoke({"query": query})
    return response["result"]

# Example usage
# llama_parse_documents should be a list of Document objects with a 'text' attribute
# vectorstore, embeddings = create_vector_database(llama_parse_documents)
# found_docs = query_vectorstore(vectorstore)
# for doc in found_docs:
#     print(doc.text[:100])
