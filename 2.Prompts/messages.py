from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv

load_dotenv()


llm = HuggingFaceEndpoint(
    repo_id="mistralai/Mistral-7B-Instruct-v0.2",
    task="text-generation",
    # huggingfacehub_api_token=os.getenv("HUGGINGFACEHUB_API_TOKEN"),
)
chat_model = ChatHuggingFace(llm=llm)


messages=[
    SystemMessage(content='You are a helpful assistant'),
    HumanMessage(content='Tell me about LangChain')
]

result = chat_model.invoke(messages)

messages.append(AIMessage(content=result.content))

print(messages)