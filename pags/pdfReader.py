import streamlit as st
from utils.db.pdf_ingestion import load_or_parse_data

def app():
    st.title("PDF Parsing")
    
    uploaded_file = st.file_uploader("Upload a PDF file", type="pdf")
    
    if uploaded_file is not None:
        parsed_data = load_or_parse_data(uploaded_file)
        
        # Display each document's content
        if parsed_data:
            st.write("PDF Parsed Successful!")


