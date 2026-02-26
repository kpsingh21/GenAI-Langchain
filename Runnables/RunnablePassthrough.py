# RunnablePassthrough is a special Runnable primitive that simply return input as a output without modifying it.
# Usecase: suppose you want to create a joke explainer but with SeqRunnable we are able to do that like in last output we are not able to see JOKE 
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableSequence,RunnableParallel,RunnablePassthrough
load_dotenv()

prompt=PromptTemplate(
    template='write a joke about {topic}',
    input_variables=['topic']
)

prompt2 =PromptTemplate(
    template='explain this joke {joke}',
    input_variables=['joke']
)

llm = HuggingFaceEndpoint(
    repo_id="mistralai/Mistral-7B-Instruct-v0.2",
    task="text-generation",
    max_new_tokens=200,

)

model = ChatHuggingFace(llm=llm)

parser=StrOutputParser()

joke_gen_chain =RunnableSequence(prompt,model,parser)

parallel_chain = RunnableParallel({
    'Joke':RunnablePassthrough(),
    'JokeExplanation':RunnableSequence(prompt2,model,parser)
})


final_chian = RunnableSequence(joke_gen_chain,parallel_chain)
result =final_chian.invoke({'topic': 'Arrange marriage' })

print(result)