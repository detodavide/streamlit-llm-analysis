from typing_extensions import TypedDict
from typing import List
import pandas as pd
from utils.logger import logger


class GraphState(TypedDict):
    """
    Represents the state of the graph.

    Attributes:
        df: A pandas DataFrame containing the data used in the graph.
        input_data: A string representing the source or nature of the input data.
        num_steps: An integer indicating the number of processing steps applied to the data.
        questions: A string representing questions that relate to the data.
        answers: A string representing answers related to the questions posed.
        summary: A string summarizing the key insights or results derived from the data.
        summary_critics: A string giving a critical reviewon how to rewrite the summary.
        critics_steps: An integer indicating the number of critical reviews given.
    """
    df: pd.DataFrame
    input_data: str
    num_steps: int
    critics_steps: int
    questions: str
    answers: str
    summary: str
    summary_critics: str