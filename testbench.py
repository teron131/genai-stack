import os
import streamlit as st
from dotenv import load_dotenv
from langchain_community.embeddings.openai import OpenAIEmbeddings
from langchain_community.chat_models import ChatOpenAI
from langchain_community.vectorstores.chroma import Chroma
from langchain_core.documents.base import Document
from langchain.chains import RetrievalQAWithSourcesChain
from langchain.chains.qa_with_sources import load_qa_with_sources_chain
from langchain_core.prompts.chat import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)
from langchain_community.graphs import Neo4jGraph
from chains import (
    configure_llm_only_chain,
    configure_qa_rag_chain,
)

def configure_qa_rag_chroma_chain_test(llm, embeddings, general_system_template):
    # RAG response
    #   System: Always talk in pirate speech.
    general_user_template = "Question:```{question}```"
    messages = [
        SystemMessagePromptTemplate.from_template(general_system_template),
        HumanMessagePromptTemplate.from_template(general_user_template),
    ]
    qa_prompt = ChatPromptTemplate.from_messages(messages)

    qa_chain = load_qa_with_sources_chain(
        llm,
        chain_type="stuff",
        prompt=qa_prompt,
    )

    # ChromaDB Knowledge Database response
    chromadb = Chroma(persist_directory="data_chroma", embedding_function=embeddings)

    kb_qa = RetrievalQAWithSourcesChain(
        combine_documents_chain=qa_chain,
        retriever=chromadb.as_retriever(search_kwargs={"k": 2}),
        reduce_k_below_max_tokens=False,
        max_tokens_limit=3375,
    )
    return kb_qa

# Define the system message templates
general_system_template_baseline = "{summaries}"
general_system_template_original = """Use the following pieces of context to answer the question at the end.
The context contains question-answer pairs and their links from Stackoverflow.
You should prefer information from accepted or more upvoted answers.
Make sure to rely on information from the answers and not on questions to provide accuate responses.
When you find particular answer in the context useful, make sure to cite it in the answer using the link.
If you don't know the answer, just say that you don't know, don't try to make up an answer.
----
{summaries}
----
Each answer you generate should contain a section at the end of links to 
Stackoverflow questions and answers you found useful, which are described under Source value.
You can only use links to StackOverflow questions that are present in the context and always
add links to the end of the answer in the style of citations.
Generate concise answers with references sources section of links to 
relevant StackOverflow questions only at the end of the answer."""
general_system_template_custom = """Use the following pieces of context to answer the question at the end.
The context contains question-answer pairs and their links from Stackoverflow.
Make sure to rely on information from the answers and not on questions to provide accuate responses.
When you find particular answer in the context useful, make sure to cite it in the answer using the link.
----
{summaries}
----
Do not include the keyword "Answer:".
Each answer you generate should contain a section at the end of links to 
Stackoverflow questions and answers you found useful, which are described under Source value.
You can only use links to StackOverflow questions that are present in the context and always
add links to the end of the answer in the style of citations.
Generate concise answers with references sources section of links to 
relevant StackOverflow questions only at the end of the answer."""

# Prepare the environment
st.title("RAG Models Testbench")
load_dotenv()
url = os.getenv("NEO4J_URI")
username = os.getenv("NEO4J_USERNAME")
password = os.getenv("NEO4J_PASSWORD")
llm=ChatOpenAI(temperature=0, model_name="gpt-4", streaming=True)
embeddings = OpenAIEmbeddings()
neo4j_graph = Neo4jGraph(url=url, username=username, password=password)
chromadb = Chroma(persist_directory="data_chroma", embedding_function=embeddings)

question = st.text_input("Enter your question:", "Is there any way to import the data to `neo4j` desktop")
prompt = st.text_area("Enter your custom prompt:", general_system_template_custom, height=200)

submit_button = st.button('Submit Question')
custom_prompt_button = st.button('Get Answer with Custom Prompt')

# Creating tabs for each model's output
tab1, tab2, tab3, tab4 = st.tabs(["LLM only", "RAG with Neo4j", "RAG with ChromaDB (baseline)", "RAG with ChromaDB (custom prompt)"])

if submit_button and question:
    with tab1:
        with st.spinner('Fetching answer...'):
            # Configure the LLM only model
            llm_chain = configure_llm_only_chain(llm)
            st.session_state['llm_answer'] = llm_chain({"question": question}, callbacks=[])["answer"]
            st.subheader("LLM only")
            st.write(st.session_state['llm_answer'])

    with tab2:
        with st.spinner('Fetching answer...'):
            # Configure the RAG model with Neo4j
            rag_chain_neo4j = configure_qa_rag_chain(
                llm, embeddings, embeddings_store_url=url, username=username, password=password
            )
            st.session_state['neo4j_answer'] = rag_chain_neo4j({"question": question})["answer"]
            st.subheader("RAG model with Neo4j")
            st.write(st.session_state['neo4j_answer'])

    with tab3:
        with st.spinner('Fetching answer...'):
            # Configure the RAG model with Chroma with the baseline prompt
            rag_chroma_chain_baseline = configure_qa_rag_chroma_chain_test(
                llm=llm, embeddings=embeddings, general_system_template=general_system_template_baseline
            )
            st.session_state['baseline_prompt_answer'] = rag_chroma_chain_baseline({"question": question})["answer"]
            st.subheader("RAG model with ChromaDB and baseline prompt (baseline)")
            st.write(st.session_state['baseline_prompt_answer'])

if custom_prompt_button and question:
    with tab4:
        with st.spinner('Fetching answers...'):
            # Configure the RAG model with Chroma with the custom prompt
            rag_chroma_chain_custom = configure_qa_rag_chroma_chain_test(
                llm=llm, embeddings=embeddings, general_system_template=prompt
            )
            st.session_state['custom_prompt_answer'] = rag_chroma_chain_custom({"question": question})["answer"]
            st.subheader("RAG model with ChromaDB and custom prompt")
            st.write(st.session_state['custom_prompt_answer'])