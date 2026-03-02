# webbaseloader is a document loader in langchain used to load and extract text content from webpages(URL)
# it uses BeautifulSoup under the hood to parse HTML and extract visible text.

#limitations:
# Load only static content (what's in the HTML .not what loads after the page rendering)
# Doesn't handle js-heavy pages well
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_community.document_loaders import WebBaseLoader
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv
from bs4 import BeautifulSoup #pip install bs4

load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id="mistralai/Mistral-7B-Instruct-v0.2",
    task="text-generation",
    max_new_tokens=200,

)

model = ChatHuggingFace(llm=llm)

prompt = PromptTemplate(
    template='Answer the following question \n {question} from the following text - \n {text}',
    input_variables=['question','text']
)

parser = StrOutputParser()

url = 'https://www.ogilvy.com/ideas/ogilvy-air-episode-1-i-believe-i-can-fly-w-richard-browning'
loader = WebBaseLoader(url)

docs = loader.load()


chain = prompt | model | parser

print(chain.invoke({'question':'What is the topic of this article?', 'text':docs[0].page_content}))


# a chrome plugin that lead you to ask about the page content based on llm using this document loader