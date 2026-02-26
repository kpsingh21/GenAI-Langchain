# RunnableLambda is a Runnable primitive that  allows us to apply custom python function within a AI pipeline.
# It acts as a middleware between different AI Components , enabling preprocessing . transformation , API calls , filtering and post preprocessing in langchain workflow
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableSequence,RunnableParallel,RunnablePassthrough,RunnableLambda
load_dotenv()


def word_count(text):
    return len(text.split())


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
    'word_count': RunnableLambda(word_count)
    # 'word_count':lambda x: len(x.split())
})


final_chian = RunnableSequence(joke_gen_chain,parallel_chain)
result =final_chian.invoke({'topic': 'love marriage' })

print(result)