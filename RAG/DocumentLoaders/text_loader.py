from langchain_community.document_loaders import TextLoader
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableSequence,RunnableParallel,RunnablePassthrough,RunnableLambda,RunnableBranch
from langchain_core.prompts import PromptTemplate


load_dotenv()

loader = TextLoader('RAG/DocumentLoaders/text.txt',encoding='utf-8') # it looks file from here


prompt=PromptTemplate(
    template='{name} is from which city',
    input_variables=['name']
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

res = chain.invoke({'name': docs[0].page_content})

print(docs)
print(type(docs)) #list of document
print(len(docs))
# print(docs[0])
# print(type(docs[0]))
# print(docs[0].page_content)
# print(docs[0].metadata)
print(res)




