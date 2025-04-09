"""
Module to work with vectors of embeddings
"""
import os 

# These next three lines swap the stdlib sqlite3 lib with the pysqlite3 package
# This is required by the Chroma package and must appear BEFORE the import.
__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
import chromadb
from models import get_embedding_model


CHROMA_DIR = os.environ.get("CHROMA_DIR", "./chroma_hylo_db")

# An example to test with
PDF_EX = "/home/jstubbs/project-data/tapis/docs_2_pdf/build/latex/tapis2pdf_SHORT.pdf"
query = "What is a Tapis associate site?"


def get_chromadb(persist_directory=CHROMA_DIR):
    """"
    Return the raw chromadb object; can use methods like list_collections(), etc. to 
    inspect the db instance. 
    """
    return chromadb.PersistentClient(path=persist_directory) 


def get_vector_store(collection, persist_directory=CHROMA_DIR):
    """
    Returns a Chroma vector store for the `collection`; 
    Optionally pass the `persist_directory` for the location to write the chroma db file.
    """
    
    return Chroma(collection_name=collection, 
                  persist_directory=persist_directory,
                  embedding_function=get_embedding_model())


def lazy_load_pdf(path_to_pdf):
    """
    Load a pdf from disk using LangChain's lazy loader
    """
    loader = PyPDFLoader(path_to_pdf)
    pages = []
    for page in loader.lazy_load():
        pages.append(page)
    return pages 


def get_text_splitter(chunk_size=1000, chunk_overlap=200, add_start_index=True):
    """
    Returns a text splitter
    """
    return RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,  # characters
        chunk_overlap=chunk_overlap,  # characters
        add_start_index=add_start_index,  # track index in original document
    )


def index_pdf(path_to_pdf, collection, persist_directory=CHROMA_DIR):
    """
    Index a pdf file at `path_to_pdf` into the vector store in the `collection`. 
    """
    # load the docs 
    docs = lazy_load_pdf(path_to_pdf)

    # split the docs 
    splitter = get_text_splitter()
    all_splits = splitter.split_documents(docs)

    # persist the embeddings to the db instance
    db = Chroma.from_documents(all_splits, 
                               get_embedding_model(),
                               collection_name=collection,
                               persist_directory=persist_directory)
    
    # return the db instance
    return db 
    

def test():
    # get the raw chroma database object
    dbr = get_chromadb()

    # list collections to see whether the test collection already exists
    dbr.list_collections()

    # delete the test collection if needed 
    # note: loading the pdf will into a db instance that already contains embeddings of the pdf 
    #       will simply result in the pdf docs being embedded twice. 
    try: dbr.delete_collection("test")
    except: pass 

    # load the test pdf; this will also create the test collection
    index_pdf(PDF_EX, "test")

    # get the LangChain instance of the vector store, which provides a high-level API
    # for similarity search against queries. 
    db = get_vector_store("test")

    # perform a similarity search with the example query 
    docs = db.similarity_search(query)

    # the result, `docs`, is an array of pages, sorted from most similar, with page having a `page_content`
    # attribute with the raw content. 
    print(f"Content of the most similar page:\n{docs[0].page_content}") 

    return dbr, db, docs


if __name__ == "__main__":
    
    # load the data 
    dbr, db, docs = test()
    
    # inspect the collection
    c = dbr.get_collection("test")
    
    # get the total number of documents
    print(f"Total number of documents: {c.count()}")