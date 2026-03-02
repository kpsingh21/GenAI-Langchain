#Directory Loader is a document loader that lets you load multiple documnets from a direcctory (folder) of files.
from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader

loader = DirectoryLoader(
    path='RAG/DocumentLoaders/books',
    glob='*.pdf',
    loader_cls=PyPDFLoader
)

docs = loader.load() # lazy_load() also available


print(docs[1].page_content)
for document in docs:
    print(document.metadata)
