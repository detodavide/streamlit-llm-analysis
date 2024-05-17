import streamlit as st
import pandas as pd
from pandas.core.series import Series
import plotly.express as px
from pandasai import SmartDataframe
from pandasai.llm import OpenAI
from llm.langgraph.csv_point_selection.workflows import full_node_schema, short_node_schema
from llm.langgraph.csv_user_question.workflows import schema_builder as question_builder
from llm.llm_model import get_ollama_models, get_llm, LLMConfig, get_openai_models

from dotenv import load_dotenv,find_dotenv
load_dotenv(find_dotenv())

def load_data(uploaded_file):
    if uploaded_file.type == 'application/vnd.ms-excel':
        df = pd.read_excel(uploaded_file)
    else:
        df = pd.read_csv(uploaded_file)
    return df

def call_pandas_agent(df):
    llm = OpenAI()
    sdf = SmartDataframe(df, config={"llm": llm})
    return sdf

def get_llm_model(provider: str):
    if provider == "Ollama":
        models = get_ollama_models()
        model = st.selectbox("Select Local model", models)
    elif provider == "OpenAI":
        models = get_openai_models()
        model = st.selectbox("Select Local model", models)
    else:
        model=None

    config=LLMConfig(llm_provider=provider, model=model)
    LLM = get_llm(config)
    return LLM

def get_llm_provider():
    providers = ["Groq", "Ollama", "OpenAI"]
    provider = st.selectbox("Select LLM Provider", providers)
    return provider

def app():
    st.title("CSV Analysis with AI Assistant")
    
    #LLM from Local Ollama Server
    provider = get_llm_provider()
    if provider:
        LLM = get_llm_model(provider)

    # File upload
    uploaded_file = st.file_uploader("Upload CSV or XLSX file", type=["csv", "xlsx"])
    if uploaded_file is not None:
        df = load_data(uploaded_file)
        df['id'] = df.index
        columns = ['id'] + [col for col in df.columns if col != 'id']
        df = df[columns]

        conversion_errors = []
        for col in df.columns:
            if df[col].dtype == 'object':
                df[col] = df[col].str.replace(',', '')  
            try:
                df[col] = df[col].astype(float)
            except ValueError:
                conversion_errors.append(col)

        # Display the dataframe and info
        st.dataframe(df, hide_index=True)

        if df is not None:
            column_options = [col for col in df.columns if col != 'id']
            x = st.selectbox('Select x-axis for scatter plot:', column_options)
            y = st.selectbox('Select y-axis for scatter plot:', column_options)
            color = st.selectbox('Select color dimension:', column_options)
            size = st.selectbox('Select size dimension:', column_options)

            fig = px.scatter(df, x=x, y=y, color=color, size=size, hover_data=df.columns, title=f"Scatter Plot of {x} vs {y}, Colored by {color}")
            event_data = st.plotly_chart(fig, on_select="rerun", use_container_width=True)

            ai_assistant = st.checkbox("Check to start the AI Assistant (WARNING: Be careful, checking this will start the API call to the llm)")

            if ai_assistant:
                chat_container = st.container()
                bot_message = st.chat_message("Assistant")
                user_input = st.chat_message("User")

                with chat_container:

                    with user_input:
                        chat = st.chat_input("Ask a question about the data")

                    with bot_message:
                        print(event_data['select'])
                        input_data: Series = df.iloc[event_data['select']['points'][0]['customdata'][0]]
                        transposed_input = input_data.to_frame().T
                        st.dataframe(transposed_input)

                        if event_data and not chat:
                            
                            # Compile the graph
                            graph_app = short_node_schema(llm=LLM)

                            # Graph inputs
                            inputs = ({"df": df, "input_data": input_data, "num_steps": 0, "critics_steps": 0})

                            # Run
                            res_analysis = graph_app.invoke(inputs)
                            st.write(res_analysis["summary"])

                        if chat:

                            # Compile the graph
                            graph_app = question_builder(llm=LLM)

                            # Graph inputs
                            inputs = ({"df": df, "input_data": input_data, "question": chat, "num_steps": 0})

                            # Run
                            res_analysis = graph_app.invoke(inputs)
                            st.write(res_analysis["answer"])

