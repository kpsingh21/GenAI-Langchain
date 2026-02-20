from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv
import os
import streamlit as st
from langchain_core.prompts import PromptTemplate,load_prompt

load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id="mistralai/Mistral-7B-Instruct-v0.2",
    task="text-generation",
    # huggingfacehub_api_token=os.getenv("HUGGINGFACEHUB_API_TOKEN"),
)
chat_model = ChatHuggingFace(llm=llm)
st.header("I know Everything!")

userInput= st.text_input("Ask your Doubt")
if st.button("Can I Answer!!!"):
    result = chat_model.invoke(userInput)
    st.write(result.content)
       
