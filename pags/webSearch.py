import streamlit as st

def app():
    request = st.text_area("Insert web search query", "")  

    if st.button("Submit"):  
        st.write(f"Request submitted: {request}")  


