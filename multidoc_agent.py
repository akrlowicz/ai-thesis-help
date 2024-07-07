from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, StorageContext, Settings, load_index_from_storage
from llama_index.core import SummaryIndex, VectorStoreIndex

from llama_index.core.agent import FunctionCallingAgentWorker
from llama_index.core.agent import AgentRunner, FunctionCallingAgentWorker
from llama_index.core.objects import ObjectIndex

from llama_index.embeddings.huggingface import HuggingFaceEmbedding
# from llama_index.llms.ollama import Ollama
from llama_index.llms.openai import OpenAI

from utils import get_doc_tools
from pathlib import Path

import os
from dotenv import load_dotenv

load_dotenv()



def load_papers(data_path):

    papers = os.listdir(data_path)

    paper_to_tools_dict = {}
    for paper in papers:
        vector_tool, summary_tool = get_doc_tools(os.path.join(data_path,paper), Path(paper.split('.pdf')[0].replace('.','')).stem)
        paper_to_tools_dict[paper] = [vector_tool, summary_tool]

    return paper_to_tools_dict
    

def main():

    # Settings.llm = Ollama(model="gpt2")
    Settings.embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-base-en-v1.5")

    llm = OpenAI(model="gpt-3.5-turbo")

    papers = os.listdir('data/')
    paper_to_tools_dict = load_papers('data')

    all_tools = [t for paper in papers for t in paper_to_tools_dict[paper]]

    obj_index = ObjectIndex.from_objects(
        all_tools,
        index_cls=VectorStoreIndex,
    )

    obj_retriever = obj_index.as_retriever(similarity_top_k=3)

    agent_worker = FunctionCallingAgentWorker.from_tools(
        tool_retriever=obj_retriever,
        llm=llm, 
        system_prompt=""" \
            You are an agent designed to answer queries over a set of given papers.
            Please always use the tools provided to answer a question. Do not rely on prior knowledge.\
            """,
        verbose=True,
    )
    
    agent = AgentRunner(agent_worker)


    response = agent.query("What is membership inference attack?")
    print(str(response))

    response = agent.query("What are common apporaches to membership inference attack for diffusion models?")
    print(str(response))

    response = agent.query("Which datasets are used for the experiment of 2305.18355v2?")
    print(str(response))

    response = agent.query("What is Proximal Initialization Attack (PIA)?")
    print(str(response))

    response = agent.query("What is the summary of file 2312.08207v4 regarding problem, methodology and the results?")
    print(str(response))


if __name__ == "__main__":
    main()