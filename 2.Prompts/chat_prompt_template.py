from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage


chat_template1 = ChatPromptTemplate([
    # SystemMessage(content='You are a helpful {domain} expert')
    # HumanMessage(content='Explain in simple terms, what is {topic}')
    # but as if we are passing like this then it will not replace the dynamic var we need to pass like this as below
    ('system', 'You are a helpful {domain} expert'),
    ('human', 'Explain in simple terms, what is {topic}')
])


# ChatPromptTemplate.from_messages this also gives same output

chat_template = ChatPromptTemplate.from_messages([
    # SystemMessage(content='You are a helpful {domain} expert')
    # HumanMessage(content='Explain in simple terms, what is {topic}')
    # but as if we are passing like this then it will not replace the dynamic var we need to pass like this as below
    ('system', 'You are a helpful {domain} expert'),
    ('human', 'Explain in simple terms, what is {topic}')
])

prompt = chat_template.invoke({'domain':'cricket','topic':'Dusra'})

print(prompt)