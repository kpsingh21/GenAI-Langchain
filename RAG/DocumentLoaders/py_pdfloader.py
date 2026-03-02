# PyPDFLoader is a document loader in Langchain used to load content from PDF files and convert each page into a Document object.

# [
#     Document(page_content='Text from page 1',metadata={'page':0,'source':'file.pdf'}),
#     Document(page_content='Text from page 1',metadata={'page':0,'source':'file.pdf'})

# ]

# Limitations: It uses the PyPDF lib under the hood - not great with scanned PDFs and complex layouts
# pip install pypdf

from langchain_community.document_loaders import PyPDFLoader

loader = PyPDFLoader('RAG/DocumentLoaders/file-sample.pdf')

docs = loader.load()

print(len(docs))

print(docs[0].page_content)
print(docs[1].metadata)