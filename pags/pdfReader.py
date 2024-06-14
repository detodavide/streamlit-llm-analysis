import streamlit as st
from utils.db.pdf_ingestion import load_or_parse_data, query_vectorstore

def app():
    st.title("PDF Parsing")

    parsing_prompt = st.text_area("Before uploading the file add a prompt for the parsing:", """The provided document is a quarterly report filed by Uber Technologies, 
        Inc. with the Securities and Exchange Commission (SEC). 
        This form provides detailed financial information about the company's performance for a specific quarter. 
        It includes unaudited financial statements, management discussion and analysis, and other relevant disclosures required by the SEC.
        It contains many tables.
        Try to be precise while answering the questions""")
    uploaded_file = st.file_uploader("Upload a PDF file", type="pdf")

    if uploaded_file is not None and parsing_prompt:
        if st.button("Load Data"):
            load_or_parse_data(uploaded_file, parsing_prompt)

        st.markdown("## Ask your Documents!")
        query = st.text_input("Enter your query:", "what is the Balance of UBER TECHNOLOGIES, INC.as of December 31, 2021?")
        if st.button("Send message"):
            answer = query_vectorstore(query, uploaded_file)
            st.markdown(f'<div class="answer">{answer}</div>', unsafe_allow_html=True)
