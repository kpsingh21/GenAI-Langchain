# Document loaders are comp in langchain used to load data from various sources into a standardized format ( usually a Document objects) which can then be used for chunking ,embedding ,retrival and generation

# Document(
#     page_content = 'The actual text content',
#     metadata= 'source': 'filename.pdf'
# )

#  Data wil come from any source like pdf,csv ,DB,S3 but we need a Standard format Data will come in specific format and that format is above mention format i.e Document

# TextLoader is a simple and commonly used document loader in langchain that reads plain text (.txt) files and converts them into Langchain Document Objects.


# we can create a custom doc loader
#  you see all this doc loaders are created by community that why it is available in
#  langchain_community.document_loaders
