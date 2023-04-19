import os, streamlit as st
import pandas as pd
# Uncomment to specify your OpenAI API key here (local testing only, not in production!), or add corresponding environment variable (recommended)
# os.environ['OPENAI_API_KEY']= ""

from pathlib import Path
from llama_index import download_loader
from llama_index import GPTSimpleVectorIndex, SimpleDirectoryReader, LLMPredictor, PromptHelper
from langchain import OpenAI
from langchain.document_loaders import DataFrameLoader


# This example uses text-davinci-003 by default; feel free to change if desired
llm_predictor = LLMPredictor(llm=OpenAI(temperature=0, model_name="text-davinci-003"))

# Configure prompt parameters and initialise helper
max_input_size = 4000
num_output = 256
max_chunk_overlap = 20

prompt_helper = PromptHelper(max_input_size, num_output, max_chunk_overlap)

# Load documents from the 'data' directory
#documents = SimpleDirectoryReader('data').load_data()


#PandasExcelReader = download_loader("PandasExcelReader")

#loader = PandasExcelReader()
#documents = loader.load_data(file=Path('./data/jebo.xlsx'), column_name="nickname", column_name="contents", column_name="date", column_name="latitude", column_name="longitude", pandas_config={"sheet_name":"Sheet1"})
#documents = loader.load_data(file=Path('./data/jebo.xlsx'), column_name=("nickname", "contents", "date", "latitude", "longitude"), pandas_config={"sheet_name":"Sheet1"})

# PandasCSVReader = download_loader("PandasCSVReader")


# df=pd.read_csv('./data/jebo.csv')


#loader = DataFrameLoader(df, page_content_column="nickname")
#documents = loader.load()
#loader = PandasCSVReader()

simpleCSVReader = download_loader("SimpleCSVReader")

loader = SimpleCSVReader()
documents = loader.load_data(file=Path('./data/jebo.csv'))



index = GPTSimpleVectorIndex(
    documents, llm_predictor=llm_predictor, prompt_helper=prompt_helper
)

# Define a simple Streamlit app
st.title("Jebo Data_case_998")
query = st.text_input("제보에는 무슨 일이 일어났을까요?", "")

if st.button("Submit"):
    response = index.query(query)
    st.write(response)
