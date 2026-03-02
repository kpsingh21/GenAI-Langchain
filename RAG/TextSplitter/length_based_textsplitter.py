# from langchain.text_splitter import CharacterTextSplitter
from langchain_text_splitters import CharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader


text = 'When to use agents: Agents can be used for open-ended problems where it’s difficult or impossible to predict the required number of steps, and where you can’t hardcode a fixed path. The LLM will potentially operate for many turns, and you must have some level of trust in its decision-making. Agents' 
loader = PyPDFLoader('RAG/DocumentLoaders/file-sample.pdf')
docs = loader.load()


splitter = CharacterTextSplitter(
    chunk_size=200,
    chunk_overlap=0, #how many charcter will overlap between your two chunks , it will help to maintain context like text spliiter will cut text abruptly to prevent context , ideally we can keep 10-20% of chunk size as a chunk overlap
    separator=''
)

# result = splitter.split_text(text)
result = splitter.split_documents(docs)

print(result)
print(result[0])
print(result[0].page_content)



