from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv

load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id="mistralai/Mistral-7B-Instruct-v0.2",
    task="text-generation",
    # huggingfacehub_api_token=os.getenv("HUGGINGFACEHUB_API_TOKEN"),
)
chat_model = ChatHuggingFace(llm=llm)


# we need chat history so it will take the ref from previous
chat_history=[]

while True:
    user_input =input('You:')
    chat_history.append(user_input)
    if user_input == 'exit':
        break
    # result=chat_model.invoke(user_input)
    result = chat_model.invoke(chat_history)
    chat_history.append(result.content)
    print("AI: ",result.content)
