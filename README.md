# ai thesis help

This project is a thesis/research help. By utilizing RAG and agents, the system is querying and summarizing research papers you desire. Each paper has its query vector engine and summarizing engine. The system uses tool routing to pick between appropriate papers and the routing between querying/summarizing option.

# Steps

Install requirements file

```sh
pip install -r requirements.txt
```
Put your research papers in pdf format in the `data` directory. It's best if the papers contain a meaningful name.

Create `.env` file and put there your `OPENAI_API_KEY`.

Specify queries in the script.

Then run the script.
```sh
python multidoc_agent.py
```


# Use cases/capabilties

This system allows you to gain information based on your dumped papers. Example queries:

- General questions about the concept that is in papers

    *What is membership inference attack?*

    *What are common apporaches to membership inference attack for diffusion models?*


- Questions about specific article (best specify a filename)

    *Which datasets are used for the experiment of example_file_name.pdf?*

    *What is Proximal Initialization Attack (PIA)?*

- Summary of specific artle (by file name)

    *What is the summary of example_file_name.pdf regarding problem, methodology and the results?*



# Limitations

With growing number of papers it will become slower. Tool routing is implemented to aid in this shortcoming. 

# Future work

Would be nice if there is possibility to integrate with Zotero to import papers, provide some interface and query the file by the paper title instead of file name.
Possibly to test open-source models e.g. llama2/3 provided by Ollama



