# import all the required libraries
from dotenv import load_dotenv
import os

# to read pdf
from PyPDF2 import PdfReader

# to start the streamlit window
import streamlit as st

# import all necessary methods from langchain library
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain_community.llms import OpenAI
from langchain_community.callbacks import get_openai_callback

# import entire library to avoid any version error
import langchain
langchain.verbose = False


# load env variables
load_dotenv()


# Utility function
def preprocess_text(text):
    """
    Utility function to split the pdf text into chunks/batches using langchain
    Parameter: text (str) - input text extracted from the pdf
    Output: knowledge_base(faiss object) - trained object with knowledge base built with text
    """
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len,
    )

    # generate chunks from the split text using text_splitter
    chunks = text_splitter.split_text(text)

    # convert the chunks of text into embeddings to form a knowledge base
    embeddings = OpenAIEmbeddings(openai_api_type=os.environ.get("OPENAI_API_KEY"))

    knowledge_base = FAISS.from_texts(chunks, embeddings)

    return knowledge_base



def main():

    """
    Main function to process the PDF and answer queries
    """

    # create a title for the streamlit app
    st.title("Chat with my PDF")

    # ask the user to upload the input PDF
    pdf = st.file_uploader("Upload your PDF file", type="pdf")

    # check if pdf is correctly uploaded and process the knowledge base
    if pdf is not None:
        # read the pdf and store it in an object
        pdf_reader = PdfReader(pdf)
        print(type(pdf_reader))

        # store the pdf text in a variable
        text = ""

        for page in pdf_reader.pages:
            text += page.extract_text()

        
        # create a knowledge base object
        knowledge_base = preprocess_text(text)

        # ask the user for the query to be answered by the module
        query = st.text_input("Ask question to PDF:")
        
        # add a cancel button to stop the process
        cancel_button = st.button("Cancel")

        if cancel_button:
            st.stop()

        if query:
            # check for the possible document similar to the query
            docs = knowledge_base.similarity_search(query)

            # run the LLM to extract answer
            llm = OpenAI(openai_api_key=os.environ.get("OPENAI_API_KEY"))

            chain = load_qa_chain(llm, chain_type="stuff")

            with get_openai_callback() as cost:
                response = chain.invoke(input=
                                        {"question": query,
                                         "input_documents": docs})
                print(cost)

                st.write(response["output_text"])

# call the main function to run the streamlit app
if __name__ == "__main__":
    main()