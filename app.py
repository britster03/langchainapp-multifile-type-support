#streamlit for creating gui in python
import streamlit as st
from dotenv import load_dotenv
import dill
import pickle
import threading
import os
from PyPDF2 import PdfReader
import csv
from streamlit_extras.add_vertical_space import add_vertical_space
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain.vectorstores.faiss import FAISS
from langchain.llms import OpenAI
from langchain.chains.question_answering import load_qa_chain
from langchain.callbacks import get_openai_callback

with st.sidebar:
    st.title('😋💬LLM Chat App')
    st.markdown('''
                ## About 
                This app is an LLM-powered chatbot built using:
                - [Straemlit](https://streamlit.io/)
                - [LangChain](https://python.langchain.com/)
                - [OpenAI](https://platform.openai.com/docs/models) LLM model
                ''')
    add_vertical_space(5)
    st.write('Made with ❤️ by [Ronit Virwani](https://linkedin.com/in/ronitvirwani)')

load_dotenv()


def read_pdf(pdf_file):
    pdf_reader = PdfReader(pdf_file)
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text()
    return text

def read_csv(csv_file):
    # Read the content of the CSV file
    content = csv_file.getvalue().decode('utf-8')

    # Use csv.reader on the content
    reader = csv.reader(content.splitlines())
    text = '\n'.join([' '.join(row) for row in reader])
    return text


def main():
    st.header("Chat with Document 💬")
    
    # Upload a document (PDF or CSV)
    document = st.file_uploader("Upload your Document", type=['pdf', 'csv'])
    
    if document is not None:
        if document.type == 'application/pdf':
            text = read_pdf(document)
        elif document.type == 'text/csv':
            text = read_csv(document)

        #uploading part completed above
        
        
        #now next we will take the pages in document and split it into chunks so that our LLM can process them
        #LLMs are large language models and have limited context window, so we have to divide our pages into smaller chunks
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,   #process  the text in chunks of this size (default: 1000).
            #there will be a overlap of 200 tokens between the conseutive chunks
            chunk_overlap=200,
            length_function=len
        )
        chunks = text_splitter.split_text(text=text)
        #now for these chunks we whave to create embeddings
        #first understand what is embedding? - embedding is a process of creating vectors using deep learning. an embedding is the output of this process -- in other words, the vector that is created by a DL model for the purpose of similarity searches by that model.
        #now what are embeddings? - in simple words embeddings are numerical representations of information, such as text, images, documents and audio. they represent each character as vector representation
        # Vector embeddings are numerical representations of data that capture the meaning and relationships of words, phrases, and other data types.
        # The distance between two vectors measures their relatedness. Small distances suggest high relatedness, and large distances suggest low relatedness. 
        # In natural language processing (NLP), a word embedding is a representation of a word. The embedding is used in text analysis. The representation is a real-valued vector that encodes the meaning of the word. Words that are closer in the vector space are expected to be similar in meaning. 
        #vector embedding that we will be using will take each chunk and we will use openai embeddings for that
        
        #embeddings

        
        #langchain supports a lot of vector stores
        #One of the most common ways to store and search over unstructured data is to embed it and store the resulting embedding vectors, and then at query time to embed the unstructured query and retrieve the embedding vectors that are 'most similar' to the embedded query. 
        #A vector store takes care of storing embedded data and performing vector search for you.
        
        #here we will use the FAISS vector database
        #here we have to pass chunks(documents) along with embeddings

        store_name = document.name[:-4]  #took the file name and dropped the last 4 letters that are corresponding to .pdf and then we can pass this on store_name
        st.write(f"{store_name}")
        embeddings = OpenAIEmbeddings()
        VectorStore = FAISS.from_texts(chunks, embedding=embeddings)

        # we will use a text input to take query from the user
        query = st.text_input("Ask questions about your PDF file:")
        # st.write(query)
        ## till now we created a knowledge base by using the files and the embeddings generated from it
        ## then we are able to accept questions from the user as well
        ## now we need to compute embeddings based on the search and do a semantic search
        ## when we do a semantic search the result is going to be a bunch of documents which the system thinks is similar to our query
        ## now this is the operation which we want to perform our knowledge base / vector store and not LLM
        
        if query:
            docs = VectorStore.similarity_search(query=query, k=3) # here we find the similar docs , 
            #'k' is the max number of documents you want to return, this is very important when we are dealing with LLM
            # this is where the concept of context window comes in because the above step returns us the most relevant documents 
            # then you take these most relevant documents and feed them through the LLMs so that becomes the context for the LLMs
            # and finally along with the query and above context the LLM tries to generate the response
            
            # so now we have got the documents that are most relevant to the question asked by the user and these will be passed to the LLM
            # we are using the openai LLM
            # and we will be feeding the questions as well as the context to the LLM and for this purpose we are using the chains from LangChain

            llm = OpenAI()
            chain = load_qa_chain(llm=llm, chain_type="stuff") 
            #so we will be using a question-answer(qa) chain , it needs an LLM and in our case we have OpenAI LLM, and also it needs the type of chain
            #we can also define the type of llm we want to use by default it will be using the da-vinci model but we can also change it to gpt 3.5 turbo
            with get_openai_callback() as cb:
                response = chain.run(input_documents=docs, question=query)
                print(cb)
            st.write(response)

if __name__ == '__main__':
    main()


# note that there is a optimum solution to use this without charge 
       # st.write(f'{store_name}')
 
       # if os.path.exists(f"{store_name}.pkl"):
       #    with open(f"{store_name}.pkl", "rb") as f:
       #        VectorStore = pickle.load(f)
            # st.write('Embeddings Loaded from the Disk')
       # else:
       #    embeddings = OpenAIEmbeddings()
       #    VectorStore = FAISS.from_texts(chunks, embedding=embeddings)
       #    with open(f"{store_name}.pkl", "wb") as f:
       #        pickle.dump(VectorStore, f)

# ------------------------------------------------------------------------------------------------------------------------------------  

   
# here if we do this then we then we dont have to generate the embeddings again and again because here we will already create a pkl file which will have the embeddings within them, but it is giving an error which says

# Traceback (most recent call last):
#  File "C:\Users\user\AppData\Local\Programs\Python\Python310\lib\site-packages\streamlit\runtime\scriptrunner\script_runner.py", line 552, in _run_script       
#    exec(code, module.__dict__)
#  File "C:\Users\user\langchainapp\app.py", line 133, in <module>
#    main()
#  File "C:\Users\user\langchainapp\app.py", line 102, in main
#    pickle.dump(embeddings, f)
# TypeError: cannot pickle '_thread.RLock' object 
