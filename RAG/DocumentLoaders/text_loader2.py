from langchain_community.document_loaders import TextLoader
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableSequence,RunnableParallel,RunnablePassthrough,RunnableLambda,RunnableBranch
from langchain_core.prompts import PromptTemplate


load_dotenv()

loader = TextLoader('RAG/DocumentLoaders/cricket.txt',encoding='utf-8') # it looks file from here


prompt = PromptTemplate(
    template='Write a summary for the following poem - \n {poem}',
    input_variables=['poem']
)


llm = HuggingFaceEndpoint(
    repo_id="mistralai/Mistral-7B-Instruct-v0.2",
    task="text-generation",
    max_new_tokens=200,

)

model = ChatHuggingFace(llm=llm)

parser=StrOutputParser()

chain = RunnableSequence(prompt,model,parser)


docs = loader.load()

print(type(docs))

print(len(docs))

print(docs[0].page_content)

print(docs[0].metadata)

# chain = prompt | model | parser

print(chain.invoke({'poem':docs[0].page_content}))