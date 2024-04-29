import streamlit as st
import pandas as pd
import plotly.express as px
from langchain.chat_models.openai import ChatOpenAI
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_experimental.agents import create_pandas_dataframe_agent
from langchain.agents.agent_types import AgentType
from pandasai import Agent, SmartDataframe
from pandasai.llm import OpenAI
from langchain_community.chat_models import ChatOllama
from llm.setup import PlotlyData, AISetup
from langchain_core.output_parsers import StrOutputParser
from llm.langgraph.workflows.csv_point_selection_flow import schema_builder

from langchain_groq import ChatGroq
from langchain_core.runnables import RunnableSerializable
from dotenv import load_dotenv,find_dotenv
load_dotenv(find_dotenv())

import re

def extract_questions(text):
    text_after_colon = text.split(':', 1)[1] if ':' in text else text   
    pattern = r'(?<=\?)\s*(?=[A-Z0-9])'
    questions = re.split(pattern, text_after_colon)
    questions = [question.strip() for question in questions if question.strip().endswith('?')]
    return questions

def load_data(uploaded_file):
    df = pd.read_csv(uploaded_file)
    return df

def call_pandas_agent(df):
    llm = OpenAI()
    sdf = SmartDataframe(df, config={"llm": llm})
    return sdf

def app():
    st.title("CSV Analysis with AI Assistant")

    # File upload
    uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])
    if uploaded_file is not None:
        df = load_data(uploaded_file)
        st.dataframe(df)

        if df is not None:
            column_options = df.columns.tolist()
            x = st.selectbox('Select x-axis for scatter plot:', column_options, index=column_options.index('total_bill') if 'total_bill' in column_options else 0)
            y = st.selectbox('Select y-axis for scatter plot:', column_options, index=column_options.index('tip') if 'tip' in column_options else 0)
            color = st.selectbox('Select color dimension:', column_options, index=column_options.index('day') if 'day' in column_options else 0)

            fig = px.scatter(df, x=x, y=y, color=color, title=f"Scatter Plot of {x} vs {y}, Colored by {color}")
            event_data = st.plotly_chart(fig, on_select="rerun")

            ai_assistant = st.checkbox("Check to start the AI Assistant (WARNING: Be careful, checking this will start the API call to the llm)")

            if ai_assistant:
                chat_container = st.container()
                bot_message = st.chat_message("Assistant")
                user_input = st.chat_message("User")

                with chat_container:

                    with user_input:
                        chat = st.chat_input("Ask a question about the data")

                    with bot_message:

                        setup = AISetup(llm=ChatGroq(temperature=0, model_name="llama3-8b-8192"))
                        input_data = PlotlyData(df=df, x=x, y=y, color=color)

                        if event_data and not chat:

                            data = event_data.select["points"][0]
                            input_data = f"- {x}: {data['x']}\n- {y}: {data['y']}\n- {color}: {data['legendgroup']}"
                            st.write(input_data)
                            
                            # Compile the graph
                            graph_app = schema_builder()

                            # Graph inputs
                            inputs = ({"df": df, "input_data": input_data, "num_steps": 0})

                            # Run
                            res_analysis = graph_app.invoke(inputs)
                            st.write(res_analysis["summary"])

                        if chat:

                            agent = setup.get_pandas_agent("question", input_data=input_data)

                            res = agent.invoke(f"Question: {chat}")
                            st.markdown(f'<p style="color:white; font-size:16px;">Question: {chat}</p>', unsafe_allow_html=True)
                            st.write(res["output"])

