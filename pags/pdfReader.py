import streamlit as st
from utils.db.pdf_ingestion import load_or_parse_data, create_vector_database, query_vectorstore

def app():
    st.title("PDF Parsing")

    parsing_prompt = st.text_area("Before uploading the file add a prompt for the parsing:", """The provided document is a quarterly report filed by Uber Technologies, 
        Inc. with the Securities and Exchange Commission (SEC). 
        This form provides detailed financial information about the company's performance for a specific quarter. 
        It includes unaudited financial statements, management discussion and analysis, and other relevant disclosures required by the SEC.
        It contains many tables.
        Try to be precise while answering the questions""",
        height=200
    )
    uploaded_file = st.file_uploader("Upload a PDF file", type="pdf")

    if uploaded_file is not None:
        parsed_data = load_or_parse_data(uploaded_file, parsing_prompt)

        if parsed_data is None:
            st.write("Parsing the data")


        if st.button("Process PDF"):
            create_vector_database(parsed_data, uploaded_file)
            st.write("PDF Parsed Successfully!")
        
        note = "If the pdf has already been processed then no need to re-process it."
        st.markdown(f"**Note:** {note}")

        st.markdown("## Ask your Documents!")
        query = st.text_input("Enter your query:", "what is the Balance of UBER TECHNOLOGIES, INC.as of December 31, 2021?")
        if st.button("Send message"):
            answer = query_vectorstore(query, uploaded_file)
            st.markdown(f'<div class="answer">{answer["result"]}</div>', unsafe_allow_html=True)

            with st.expander("Document Similarity Search"):
                for i, doc in enumerate(answer["source_documents"]):
                    st.markdown(f"### Document {i+1} \n{doc.page_content}")
                    st.write("---------------------------------")