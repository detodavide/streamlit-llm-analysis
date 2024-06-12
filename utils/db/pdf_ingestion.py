import os
import joblib
from llama_parse import LlamaParse

from dotenv import load_dotenv,find_dotenv
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
