import streamlit as st
from streamlit_option_menu import option_menu
import pags.csvAnalysis, pags.pdfReader

st.set_page_config(
    page_title="AI"
)

st.markdown(
    """
    <style>
    .answer {
        font-size: 24px;
        color: black;
        background-color: #f0f0f0;
        padding: 10px;
        border-radius: 5px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

class MultiApp:
    def __init__(self) -> None:
        self.apps = []

    def add_app(self, title, func):
        self.apps.append({
            "title": title,
            "function": func
        })

    def run():
        with st.sidebar:
            app = option_menu(
                menu_title="Table of Content",
                options=['CSV Analysis', "PDF Reader"],
                default_index=1,
                styles={
                "container": {"padding": "4!important","background-color":'black'},
                "icon": {"color": "white", "font-size": "23px"}, 
                "nav-link": {"color":"white","font-size": "18px", "text-align": "left", "margin":"0px", "--hover-color": "blue"},
                "nav-link-selected": {"background-color": "#02ab21", "font-size": "18px"},
                }
            )
        if app == "CSV Analysis":
            pags.csvAnalysis.app()
        if app == "PDF Reader":
            pags.pdfReader.app()  

    run()      