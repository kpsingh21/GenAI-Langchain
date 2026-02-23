from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableParallel,RunnableBranch, RunnableLambda

from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field
from typing import Literal


load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id="mistralai/Mistral-7B-Instruct-v0.2",
    task="text-generation",
)

model = ChatHuggingFace(llm=llm)
parser=StrOutputParser()

class Feedback(BaseModel):

    sentiment: Literal['positive', 'negative'] = Field(description='Give the sentiment of the feedback')

parser2 = PydanticOutputParser(pydantic_object=Feedback)


prompt =PromptTemplate(
    template='Classify the sentiment of the following feedback text into postive or negative \n {feedback} \n {format_instruction}',
    input_variables=['feedback'],
    partial_variables={'format_instruction':parser2.get_format_instructions()}
)


# classifier_chain = prompt | model | parser
classifier_chain = prompt | model | parser2

customer_feedback = "I had a wonderful experience at this restaurant. The food was absolutely delicious, fresh, and full of flavor. Every dish was beautifully presented and exceeded my expectations. The staff was friendly, attentive, and made us feel very welcome from the moment we arrived. The service was quick and professional, and the atmosphere was warm and comfortable. Overall, it was a fantastic dining experience, and I would definitely recommend this restaurant to friends and family. I’m looking forward to visiting again soon!"
result = classifier_chain.invoke(customer_feedback).sentiment
# as we see here the result is kind of para or we don't have any control over this to structure and validate we will use pydantic Parser
print(result)

prompt2 = PromptTemplate(
    template='Write an appropriate response to this positive feedback \n {feedback}',
    input_variables=['feedback']
)

prompt3 = PromptTemplate(
    template='Write an appropriate response to this negative feedback \n {feedback}',
    input_variables=['feedback']
)


# RunnableBranch : you send tuple in which there is condtion and what to execute on this condition

# branch_chain =RunnableBranch(
#     (condition1,chain1),
#     (condition2,chain2),
#     defaultchain
# )

# lambda arguments: expression
branch_chain = RunnableBranch(
    (lambda x:x.sentiment == 'positive', prompt2 | model | parser),
    (lambda x:x.sentiment == 'negative', prompt3 | model | parser),
    RunnableLambda(lambda x: "could not find sentiment")
)

chain = classifier_chain | branch_chain

print(chain.invoke({'feedback': 'This is a terrible phone'}))

chain.get_graph().print_ascii()