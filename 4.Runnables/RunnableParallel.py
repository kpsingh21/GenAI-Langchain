# Runnable Parallel is a runnable primitive that allows multiple runnables to execute in parallel 
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableSequence,RunnableParallel
load_dotenv()

prompt1 = PromptTemplate(
    template='Generate a tweet about {topic}',
    input_variable=['topic']
)

prompt2 = PromptTemplate(
    template='Generate a linkedin post about {topic}',
    input_variable=['topic']
)

llm = HuggingFaceEndpoint(
    repo_id="mistralai/Mistral-7B-Instruct-v0.2",
    task="text-generation",
    max_new_tokens=200,

)

model = ChatHuggingFace(llm=llm)

parser=StrOutputParser()

parallel_chain = RunnableParallel({
    'tweet':RunnableSequence(prompt1,model,parser),
    'linkedin':RunnableSequence(prompt1,model,parser)
})


result =parallel_chain.invoke({'topic':'AI'})
print(result)