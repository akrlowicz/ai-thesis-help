from llama_index.core import SummaryIndex, VectorStoreIndex, SimpleDirectoryReader, StorageContext, Settings, load_index_from_storage
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.tools import QueryEngineTool
from llama_index.core.query_engine.router_query_engine import RouterQueryEngine
from llama_index.core.selectors import LLMSingleSelector

from llama_index.embeddings.huggingface import HuggingFaceEmbedding
#from llama_index.llms.ollama import Ollama

from dotenv import load_dotenv
import os

load_dotenv()


PERSIST_DIR = "./storage"

def loader():

    if not os.path.exists(PERSIST_DIR):
        
        documents = SimpleDirectoryReader("data").load_data()

        splitter = SentenceSplitter(chunk_size=1024) # experiment with chunk size
        nodes = splitter.get_nodes_from_documents(documents)

        vector_index = VectorStoreIndex(nodes)
        summary_index = SummaryIndex(nodes)

        # store the index
        vector_index.storage_context.persist(persist_dir=PERSIST_DIR+ "/vector")
        summary_index.storage_context.persist(persist_dir=PERSIST_DIR+ "/summary")
    else:
        vector_storage_context = StorageContext.from_defaults(persist_dir=PERSIST_DIR+ "/vector")
        summary_storage_context = StorageContext.from_defaults(persist_dir=PERSIST_DIR+ "/summary")
        
        vector_index = load_index_from_storage(vector_storage_context)
        summary_index = load_index_from_storage(summary_storage_context)

    return vector_index, summary_index   

def main():

    # Settings.llm = Ollama(model="llama2")
    Settings.embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-base-en-v1.5")

    # load documents
    vector_index, summary_index = loader()

    summary_query_engine = summary_index.as_query_engine(response_mode="tree_summarize")
    vector_query_engine = vector_index.as_query_engine()

    summary_tool = QueryEngineTool.from_defaults(
        query_engine=summary_query_engine,
        description=("Useful for summarization questions related to the papers."),
    )

    vector_tool = QueryEngineTool.from_defaults(
        query_engine=vector_query_engine,
        description=("Useful for retrieving specific context from all the papers."),
    )

    # define routing engine
    query_engine = RouterQueryEngine(
        selector=LLMSingleSelector.from_defaults(),
        query_engine_tools=[
            summary_tool,
            vector_tool,
        ],
        verbose=True
    )

    response = query_engine.query("What are the challenges of membership inference attack?")

    print(str(response))

    # for n in response.source_nodes:
    #     print(n.metadata)


if __name__ == "__main__":
    main()