from langchain_community.vectorstores import Qdrant
from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain_community.embeddings import GPT4AllEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter

def load_data():
    dirpath = 'dataset'
    papers = []
    loader = DirectoryLoader(dirpath, glob="./*.pdf", loader_cls=PyPDFLoader)
    try:
        papers = loader.load()
    except Exception as e:
        print(f"Error loading file: {e}")
    print("Total number of pages loaded:", len(papers)) 

    # Concatenate all pages' content into a single string
    full_text = ''
    for paper in papers:
        full_text += paper.page_content

    # Remove empty lines and join lines into a single string
    full_text = " ".join(line for line in full_text.splitlines() if line)
    print("Total characters in the concatenated text:", len(full_text)) 

    # Split the text into chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    paper_chunks = text_splitter.create_documents([full_text])

    # Create Qdrant vector store
    qdrant = Qdrant.from_documents(
        documents=paper_chunks,
        embedding=GPT4AllEmbeddings(),
        path="./tmp/local_qdrant",
        collection_name="govt_schemes",
    )

    retriever = qdrant.as_retriever()
    return retriever