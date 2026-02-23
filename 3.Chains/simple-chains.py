from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

# Define the model
llm = HuggingFaceEndpoint(
    repo_id="mistralai/Mistral-7B-Instruct-v0.2",
    task="text-generation",
    # huggingfacehub_api_token=os.getenv("HUGGINGFACEHUB_API_TOKEN"),
)

model = ChatHuggingFace(llm=llm)

template = PromptTemplate(
    template='Write a detailed report on {topic}',
    input_variables=['topic']
)

parser =  StrOutputParser()

chain = template | model | parser
result = chain.invoke({ 'topic':'cricket'})
print(result)
chain.get_graph().print_ascii() \
# prominput -> PromptTemplate -> chatHuggingFace ->  StrOutputParser -> StrOutputParserOutput