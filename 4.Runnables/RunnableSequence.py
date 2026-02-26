# actually Runnables are nothing but Kind of abstract class is created and all the other components will inherit that abstract class

#Runnablesequence is asequential chain of runnables in Langchain that executes each ste[ one after another, passing the o/p of one step as the i/p of the next.

from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableSequence,RunnableParallel
load_dotenv()

# Define the model
llm = HuggingFaceEndpoint(
    repo_id="mistralai/Mistral-7B-Instruct-v0.2",
    task="text-generation",
    max_new_tokens=200,

)

prompt=PromptTemplate(
    template='write a joke about {topic}',
    input_variables=['topic']
)
model = ChatHuggingFace(llm=llm)


parser=StrOutputParser()
chain = RunnableSequence(prompt,model,parser)
result = chain.invoke({'topic':'Office'})
print(result)


