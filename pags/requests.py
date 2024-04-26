import streamlit as st

def app():
    request = st.text_area("Insert a request", "")  

    if st.button("Submit"):  
        st.write(f"Request submitted: {request}")  


