from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv
import os

load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id="mistralai/Mistral-7B-Instruct-v0.2",
    task="text-generation",
    # huggingfacehub_api_token=os.getenv("HUGGINGFACEHUB_API_TOKEN"),
)

chat_model = ChatHuggingFace(llm=llm)

result = chat_model.invoke("What is the capital of India?")
print(result.content)


