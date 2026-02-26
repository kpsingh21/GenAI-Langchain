# RunnableBranch is kind of if /else in langchain
#  in a  RunnableBranch we will pass condtion in tuple , every if condtion is a tuple and that tuple contains ( condition , runnable ) and one default condition
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableSequence,RunnableParallel,RunnablePassthrough,RunnableLambda,RunnableBranch
load_dotenv()

prompt=PromptTemplate(
    template='write a report on the {topic}',
    input_variables=['topic']
)

prompt2 =PromptTemplate(
    template='summerize this  {report}',
    input_variables=['report']
)

llm = HuggingFaceEndpoint(
    repo_id="mistralai/Mistral-7B-Instruct-v0.2",
    task="text-generation",
    max_new_tokens=200,

)


model = ChatHuggingFace(llm=llm)

parser=StrOutputParser()

gen_report = RunnableSequence(prompt,model,parser)

branch_chain =RunnableBranch(
    (lambda x: len(x.split())>300, prompt2 | model | parser),
    RunnablePassthrough()  # this default condition
)


final_chain = RunnableSequence(gen_report,branch_chain)
print(final_chain.invoke({'topic':'Russia vs Ukraine'}))



# LCEL :langchain expression language
# As we observe  RunnableSequence is used many times or we are using it multiple times so 
#  they created a synatx :  s1 | s2 | s3