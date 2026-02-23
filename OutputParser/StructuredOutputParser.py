from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
# from langchain_core.output_parsers import JsonOutputParser
# StructuredOutputParser is prsent in langchain.outputparser not in  langchain_core , langchain_core is kind of lib from lagchain which carrries reusable libraries
# from langchain.output_parsers import StructuredOutputParser, ResponseSchema
from langchain_core.output_parsers import StructuredOutputParser , ResponseSchema
load_dotenv()

# Define the model
llm = HuggingFaceEndpoint(
    repo_id="mistralai/Mistral-7B-Instruct-v0.2",
    task="text-generation",
    # huggingfacehub_api_token=os.getenv("HUGGINGFACEHUB_API_TOKEN"),
)

model = ChatHuggingFace(llm=llm)

schema = [
    ResponseSchema(name='fact_1', description='Fact 1 about the topic'),
    ResponseSchema(name='fact_2', description='Fact 2 about the topic'),
    ResponseSchema(name='fact_3', description='Fact 3 about the topic'),
]

parser = StructuredOutputParser.from_response_schemas(schema)

template = PromptTemplate(
    template='Give 3 fact about {topic} \n {format_instruction}',
    input_variables=['topic'],
    partial_variables={'format_instruction':parser.get_format_instructions()}
)


promt= template.invoke({'topic':'black hole'})
result = model.invoke(promt)
final_res =parser.parse(result.content)

print(final_res)

# with chain
chain = template | model | parser
result = chain.invoke({'topic':'black hole'})
print(result)