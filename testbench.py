import os
from dotenv import load_dotenv
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.chat_models import ChatOpenAI
from langchain.vectorstores.chroma import Chroma
from langchain.chains import RetrievalQAWithSourcesChain
from langchain.chains.qa_with_sources import load_qa_with_sources_chain
from langchain.prompts.chat import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)
from langchain.graphs import Neo4jGraph
from chains import (
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
general_system_template_original = """
    Use the following pieces of context to answer the question at the end.
    The context contains question-answer pairs and their links from Stackoverflow.
    Make sure to rely on information from the answers and not on questions to provide accuate responses.
    ----
    {summaries}
    ----
    Generate concise answers with references sources section of links to 
    relevant StackOverflow questions only at the end of the answer.
    DO NOT include the keywords "Question:", "Answer:", "Score:", "Link:" in your answer.
    """
general_system_template_with_instructions = """
    Use the following pieces of context to answer the question at the end.
    The context contains question-answer pairs in the following formmat:
        Question: title
        content
        Answer: content
        Score: score
        Link: link
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
    relevant StackOverflow questions only at the end of the answer.
"""

# Prepare the environment
load_dotenv()
url = os.getenv("NEO4J_URI")
username = os.getenv("NEO4J_USERNAME")
password = os.getenv("NEO4J_PASSWORD")
llm=ChatOpenAI(temperature=0, model_name="gpt-4", streaming=True)
embeddings = OpenAIEmbeddings()
neo4j_graph = Neo4jGraph(url=url, username=username, password=password)
chromadb = Chroma(persist_directory="data_chroma", embedding_function=embeddings)

# Configure the RAG model with ChromaDB baseline prompt
rag_chroma_chain_baseline = configure_qa_rag_chroma_chain_test(
    llm=llm, embeddings=embeddings, general_system_template=general_system_template_baseline
)

# Configure the RAG model with ChromaDB and additional instructions prompt
rag_chroma_chain_test = configure_qa_rag_chroma_chain_test(
    llm=llm, embeddings=embeddings, general_system_template=general_system_template_original
)

# Define a question
question = "Is there anyway to import the data to `neo4j` desktop"

# Simulate vector search
query_vector = embeddings.embed_query(question)
retrieved_docs = chromadb.similarity_search_by_vector_with_relevance_scores(query_vector, k=2)
for doc in retrieved_docs:
    print(doc[0])
    print(doc[1])

print("-"*100)

# Get the answer from the baseline RAG model
print("## RAG model with ChromaDB baseline prompt ##")
print(rag_chroma_chain_baseline({"question": question})["answer"])

print("-"*100)

# Get the answer from the RAG model with additional instructions
print("## RAG model with additional instructions prompt ##")
print(rag_chroma_chain_test({"question": question})["answer"])


# Configure the RAG model with Neo4j
rag_chain = configure_qa_rag_chain(
    llm, embeddings, embeddings_store_url=url, username=username, password=password
)

print("-"*100)

# Get the answer from the baseline RAG model
print("## RAG model with Neo4j ##")
print(rag_chain({"question": question})["answer"])