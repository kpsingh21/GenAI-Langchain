from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser

load_dotenv()

# Define the model
llm = HuggingFaceEndpoint(
    repo_id="mistralai/Mistral-7B-Instruct-v0.2",
    task="text-generation",
    # huggingfacehub_api_token=os.getenv("HUGGINGFACEHUB_API_TOKEN"),
)
chat_model = ChatHuggingFace(llm=llm)

# model = ChatHuggingFace(llm=llm)

parser = JsonOutputParser()

template = PromptTemplate(
    template='Give me the name ,version, and company of  5 LLM\n {format_instruction}',
    input_variables=[],
    partial_variables={'format_instruction': parser.get_format_instructions()}
)

promt =template.format()
# print(promt) // Give me the name ,version, and company of  5 LLM Return a JSON Object

# without chain 
# result = chat_model.invoke(promt)
# final_result = parser.parse(result.content)

#with chain
chain = template | chat_model | parser
result = chain.invoke({}) 
# invoke fun require a dict even if there is no input variable {}

# print(result.content)
print(result)


# JsonOutputParser does not enforce any specific schema i.e why we need StructuredOutputParser
