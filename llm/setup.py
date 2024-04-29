from langchain_community.chat_models import ChatOllama
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_openai import ChatOpenAI
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_experimental.agents import create_pandas_dataframe_agent
from langchain.agents.agent_types import AgentType
from pandasai import Agent, SmartDataframe
from langchain.agents import AgentExecutor
from typing import Any
from pydantic import BaseModel
from langchain_core.runnables import RunnableSerializable
import llm.templates as templates
import pandas as pd

class PlotlyData(BaseModel):
    df: pd.DataFrame
    x: str = "None"
    y: str = "None"
    color: str = "None"

    class Config:
        arbitrary_types_allowed = True

class AISetup:
    def __init__(self, llm: BaseChatModel):
        if not isinstance(llm, (ChatOllama, ChatOpenAI, ChatGroq)):
            raise ValueError("Invalid model type. llm must be an instance of ChatOllama, ChatOpenAI or ChatGroq.")
        self.llm = llm

    def get_analyst_executor(self, agent_mode: str, input_data: PlotlyData):
        """
        High order function to call the agent/chain/smartdataframe based on the selected mode
        """
        agent_modes = {
            "selection": templates.SELECTION_AGENT,
            "question": templates.QUESTION_AGENT
        }
        template = agent_modes.get(agent_mode)
        if not template:
            raise ValueError("Unsupported agent type specified. Choose 'selection' or 'question'.")
        return self._call_agent(input_data, template)

    def _call_agent(self, input_data: PlotlyData, template) -> (SmartDataframe | AgentExecutor | RunnableSerializable):
        """
        Return an agent or a smartDataFrame depending on the LLM instanciated
        """

        if isinstance(self.llm, ChatOpenAI):
            agent_type = AgentType.OPENAI_FUNCTIONS

            agent = create_pandas_dataframe_agent(
            prefix=template,
            llm=self.llm,
            df=input_data.df,
            agent_type=agent_type,
            verbose=True
        )
        if isinstance(self.llm, ChatGroq):
            output = StrOutputParser()  
            chain = self.llm | output
            return chain

        agent.handle_parsing_errors = True
        return agent