import streamlit as st
import os
import dotenv
import langchain
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.tools.retriever import create_retriever_tool
from langgraph.checkpoint.memory import InMemorySaver
import uuid
from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from langchain.agents import create_agent




dotenv.load_dotenv()
api_key = os.getenv("HUGGINGFACE_API_KEY")


st.set_page_config(page_title="RAG Chatbot", layout="wide")

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

def main():
    st.title("RAG Chatbot with Document Memory")
    
    # sidebar
    with st.sidebar:
        st.header("Configuration")
        
        # API Token input
        hf_token = st.text_input("HuggingFace Token (Optional)", type="password")
        if hf_token:
            os.environ["HUGGINGFACEHUB_API_TOKEN"] = hf_token
            
        st.divider()
        
        uploaded_file = st.file_uploader("Upload a document (PDF only)", type=["pdf"])
        
        st.divider()
        
        st.subheader("Model Settings")
        embedding_model_name = st.text_input("Embedding Model", value="Qwen/Qwen3-Embedding-0.6B")
        repo_id = st.text_input("Chat Model Repo ID", value="zai-org/GLM-4.7")
        
        process_btn = st.button("Process Document")

    st.subheader("Chat Area")

    if uploaded_file and process_btn:


        with st.spinner("Processing document..."):

            try:

                with open(uploaded_file.name, "wb") as f:
                    f.write(uploaded_file.getbuffer())
                
                # Load document
                if uploaded_file.name.endswith(".pdf"):
                    loader = PyPDFLoader(uploaded_file.name)
                    docs = loader.load()
                else:
                    st.error("Please upload a pdf.")
                
                total_chars = sum(len(doc.page_content) for doc in docs)
                size = total_chars/5

                if size>5000:
                    size = 1000
                

                # split, embed, store
                text_splitter = RecursiveCharacterTextSplitter(chunk_size=size, chunk_overlap=(size/3))
                splits = text_splitter.split_documents(docs)
                
                
                embeddings = HuggingFaceEmbeddings(model_name=embedding_model_name)
                
                
                vectorstore = FAISS.from_documents(documents=splits, embedding=embeddings)
                


                # Create Retriever Tool
                retriever = vectorstore.as_retriever(k=2)
                tool = create_retriever_tool(
                    retriever,
                    "search_document",
                    "Searches and returns excerpts from the uploaded document.",
                )
                
                st.session_state.tools = [tool]
                st.session_state.vectorstore = vectorstore
                os.remove(uploaded_file.name)
                
                st.success(f"Processed {len(splits)} chunks!")
                
            except Exception as e:
                st.error(f"Error processing file: {e}")



    if "checkpointer" not in st.session_state:
        st.session_state.checkpointer = InMemorySaver()

    if "thread_id" not in st.session_state:
        st.session_state.thread_id = str(uuid.uuid4())

    
    for message in st.session_state.chat_history:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Chat input
    if prompt := st.chat_input("Ask a question about your document..."):

        st.session_state.chat_history.append({"role": "user", "content": prompt})

        with st.chat_message("user"):
            st.markdown(prompt)
            
        # Response logic
        with st.chat_message("assistant"):

            if "tools" not in st.session_state:

                st.warning("Please upload and process a document first.")
                response_text = "Please upload and process a document first."


            else:

                with st.spinner("Thinking..."):
                   
                   try:
                       
                       llm = HuggingFaceEndpoint(
                           repo_id=repo_id, 
                           temperature=0.5, 
                           huggingfacehub_api_token=api_key,
                           max_new_tokens=1024,
                           stop_sequences=["<|endoftext|>", "</s>", "[/ASSIST]"]
                       )
                       
                       chat_model = ChatHuggingFace(llm=llm)

                       
                       agent = create_agent(
                           chat_model, 
                           st.session_state.tools, 
                           checkpointer=st.session_state.checkpointer
                       )
                       
                       # retrieve the relevant docs
                       retriever = st.session_state.vectorstore.as_retriever(k=1)
                       retrieved_docs = retriever.invoke(prompt)
                       
                       context = "\n\n".join([doc.page_content for doc in retrieved_docs])
                       
                       # prompt
                       # prompt
                       enhanced_prompt = f"""RETRIEVED CONTEXT:
{context}

USER QUESTION:
{prompt}

INSTRUCTIONS:
NEVER say something like: "from the retrieved context".
check if the RETRIEVED CONTEXT is relevant to the USER QUESTION.
- If the context contains the answer, use it.
- If the context is irrelevant (e.g. asking about general topics, or general greeting), IGNORE the context and answer 
from your own knowledge. You do not need to tell if the context provided is irrelevant.
Just answer the question and give a regular answer.
"""
                       
                       config = {"configurable": {"thread_id": st.session_state.thread_id}}
                       
                       result = agent.invoke( 
                           {"messages": [{"role": "user", "content": enhanced_prompt}]},
                           config=config
                       )
                       
                       
                       
                       response_text = result["messages"][-1].content
                       
                       # Clean up potential artifacts if the model kept generating
                       if "[/ASSIST]" in response_text:
                           response_text = response_text.split("[/ASSIST]")[-1].strip()
                       
                   except Exception as e:
                       st.error(str(e))
            
            st.markdown(response_text)
            
        st.session_state.chat_history.append({"role": "assistant", "content": response_text})

if __name__ == "__main__":
    main()
