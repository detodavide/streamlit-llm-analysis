import streamlit as st
from utils.db.pdf_ingestion import load_or_parse_data, create_vector_database, query_vectorstore

def app():
    st.title("PDF Parsing")

    uploaded_file = st.file_uploader("Upload a PDF file", type="pdf")

    if uploaded_file is not None:
        parsed_data = load_or_parse_data(uploaded_file)

        if parsed_data is None:
            st.write("Parsing the data")

        # Display each document's content
        if parsed_data:
            st.write("PDF Parsed Successfully!")

            if st.button("Create Vector Database"):
                create_vector_database(parsed_data, uploaded_file)
                st.write("Vector Database Created Successfully!")
            
            note = "If the vector database has been already instantiated no need to rerun the creation."
            st.markdown(f"**Note:** {note}")

            st.markdown("## Ask your Documents!")
            query = st.text_input("Enter your query:", "what is the Balance of UBER TECHNOLOGIES, INC.as of December 31, 2021?")
            if st.button("Send message"):
                answer = query_vectorstore(query, uploaded_file)
                st.markdown(f'<div class="answer">{answer}</div>', unsafe_allow_html=True)
