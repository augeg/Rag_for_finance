import configparser

from langchain_community.llms import CTransformers
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers.string import StrOutputParser

"""
===============================================================================

    RAG Module for finance : LLM Setup 
    Using Langchain pipeline
    
    INPUT : 
        - Vector embedding model for document embedding
        - Pre trained Large Language model for Q&A task and analysis

    OUTPUT : 
        - LLM Class 

===============================================================================
"""

config = configparser.ConfigParser()
config.read("C:/Sauvegarde/Trading_house/Rag_for_finance/config.ini")

class llm :
    def __init__(self):
        self.path_model = config.get("PATH", "path_model")
        self.path_vector = config.get("PATH", "path_vector")
    
    def build_llm(self):
        # local CTransformers model - no Online model
        return CTransformers(model = self.path_model,
                             model_type = "llama",
                             config = {"max_new_tokens" : 256,
                                       'temperature' : 0.1})
    
    def add_data(self, docs:list):
        return RecursiveCharacterTextSplitter(chunk_size=100, # Can be bigger
                                              chunk_overlap = 20,
                                              length_function=len,
                                              is_separator_regex=False).create_documents(docs)
    
    def get_embeddings(self):
        return HuggingFaceEmbeddings(model_name = self.path_vector,
                                     encode_kwargs = {"normalize_embeddings" : True},
                                     model_kwargs = {"device" : 'gpu'}) # Can be CPU - depending on FAISS
        
    def get_vectorstore(self, docs:list):
        """
        INPUT :
            docs : list of text
            
        OUTPUT :
            Docs vectorized with embedding
        """
        text = self.add_data(docs)
        embeddings = self.get_embeddings()
        return FAISS.from_documents(text, embeddings)
    
    def get_prompt(self, prompt):
        return PromptTemplate.from_template(prompt)
    
    
    def run(self, docs:list, prompt, question) :
        
        """
        Question should be chosen wisely. 
        May finetune the prompt as well to improve result, depending to the choice of th pretrained LLM
        """
        
        def format_docs(ds):
            return "\n\n".join(d.page_content for d in ds)
        
        print("Docs vectorization ...")
        Vectorstore = self.get_vectorstore(docs)
        prompt = self.get_prompt(prompt)
        llm = self.build_llm()
        rag_chain = (
            {"context": Vectorstore.as_retriever() | format_docs, "question" : RunnablePassthrough()}
            | prompt
            | llm
            | StrOutputParser()
            )
        print("Answering to prompt ...")        
        return rag_chain.invoke(question)
        
if __name__=="__main__":
    model_llm = llm()        
        
        
        
        
        