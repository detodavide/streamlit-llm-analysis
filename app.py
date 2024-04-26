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

def call_selection_agent(df, x="None", y="None", color="None"):
    llm = ChatOllama(model='phi3:3.8b', temperature=0.0)
    agent = create_pandas_dataframe_agent(
        prefix=f"""You are a helpful data analyst expert that gives elaborate insight data.""" ,
        llm=llm,
        df=df,
        agent_type=AgentType.OPENAI_FUNCTIONS,
        verbose=True
    )
    agent.handle_parsing_errors=True
    return agent

def call_question_agent(df, x="None", y="None", color="None"):
    llm = ChatOpenAI(model='gpt-3.5-turbo', temperature=0.0)
    agent = create_pandas_dataframe_agent(
        prefix=f"""You are a helpful data analyst expert that gives elaborate insight in the context of the Dataframe, based on the user question.""" ,
        llm=llm,
        df=df,
        agent_type=AgentType.OPENAI_FUNCTIONS,
        verbose=True
    )
    agent.handle_parsing_errors=True
    return agent

def call_pandas_agent(df):
    llm = OpenAI()
    sdf = SmartDataframe(df, config={"llm": llm})
    return sdf

# Main Streamlit app
def main():
    st.title("CSV Analysis with AI Assistant")

    # File upload
    uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])
    if uploaded_file is not None:
        df = load_data(uploaded_file)
        st.dataframe(df)

        agent_options = ["LangChain Agent", "PandasAI Agent"]

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
                        selected_agent = st.selectbox("Select the AI Agent to assist you", agent_options)
                        if selected_agent == "LangChain Agent":
                            if event_data and not chat:
                                data = event_data.select["points"][0]
                                agent = call_selection_agent(df,x,y,color)
                                st.write(f"- {x}: {data['x']}\n- {y}: {data['y']}\n- {color}: {data['legendgroup']}")
                                questions = agent.invoke(f"Write some insightful questions about this specific data:\n- {x}: {data['x']}\n- {y}: {data['y']}\n- {color}: {data['legendgroup']} with respect to the dataframe, but keep the focus on the specific data.")
                                extracted_questions = extract_questions(questions["output"])
                                answer_list = []
                                for i, question in enumerate(extracted_questions, start=1):
                                    res = agent.invoke(f"{question}")
                                    answer_list.append(res["output"])
                                answers = "\n".join(answer_list)
                                final_res = agent.invoke(f"Given this context:\n{answers}\nWrite a concise summary of the context in a discorsive way and in italian.")
                                st.write(final_res["output"])

                            if chat:
                                agent = call_question_agent(df,x,y,color)
                                res = agent.invoke(f"Question: {chat}")
                                st.markdown(f'<p style="color:white; font-size:16px;">Question: {chat}</p>', unsafe_allow_html=True)
                                st.write(res["output"])
                        if selected_agent == "PandasAI Agent":
                            if event_data and not chat:
                                data = event_data.select["points"][0]
                                sdf = call_pandas_agent(df)
                                st.write(f"- {x}: {data['x']}\n- {y}: {data['y']}\n- {color}: {data['legendgroup']}")
                                row = sdf.chat(f"Given this data that represent a dataframe row:\n-{x}: {data['x']}\n- {y}: {data['y']}\n- {color}: {data['legendgroup']}\n\nFind the row representing this data.")
                                st.write(row)
                                res = sdf.chat(f"Given the row: {row}. There are any correlation or trend with other rows??")
                                st.write(res)

                            if chat:
                                agent = call_question_agent(df,x,y,color)
                                res = agent.invoke(f"Question: {chat}")
                                st.markdown(f'<p style="color:white; font-size:16px;">Question: {chat}</p>', unsafe_allow_html=True)
                                st.write(res["output"])



    
if __name__ == "__main__":
    main()
