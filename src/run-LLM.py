import torch, click
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceInstructEmbediings
from langchain.llms import HuggingFacePipeline
from langchain.chains import RetrievalQA
from transformers import AutoModelForCasualLM, AutoTokenizer, pipeline


def load_model(device):
    """Model can be selected from huggingface. It will download the model
    for first execution.
    It will use the model from the disk for next iteration of runs
    """
    model = 'tiiuae/falcon-7b-instruct'
    if device == "cuda":
        tokenizer = AutoTokenizer.from_pretrained(model)
    else:
        tokenizer = AutoTokenizer.from_pretrained(model)
        model = AutoModelForCasualLM.from_pretrained(model,trust_remote_code=True)
    Pipe = pipeline('text-generation', tokenizer=tokenizer, model=model,torch_dtype=torch.float32 if device=="cpu" else torch.bfloat16,
                    device_map=device if device=="cpu" else "auto", max_length=2048, temoerature=0, top_p=0.90, top_k=10,
                    repetition_penalty=1.15, num_return_sequences=1, pad_token_id=tokenizer.eos_token_id)
    local_llm = HuggingFacePipeline(pipeline=Pipe)
    return local_llm

@click.command()
@click.option('--device_type', default='cuda', help='delect gpu or cpu for execution')

def main(device_type,):
    # Load the instructorEmbeddings
    if device_type in ['cpu', 'CPU']: device='cpu'
    else: device='cuda'
    print(f"Running on: {device}")
    embeddings = HuggingFaceInstructEmbediings(model_name="hkunlp/instructor-base", model_kwargs={"device":device})
    # Load the vectorstore from disk which was saved earlier
    database = FAISS.load_local('faiss_index', embeddings)
    retriever = database.as_retriever()
    # Load the llm for returning the responses to the questions asked 
    llm = load_model(device=device_type)
    query = RetrievalQA.form_chain_type(llm, retriever, chain_type="stuff",return_source_documents=True)
    while True:
        query = input("\nEnter the query:")
        if query=="exit":
            break
        # Get the answer from the question & answer chain
        result = query(query)
        answer, docs = result['result'], result['source_documents']
        print("\n\n> Question:")
        print(query)
        print("\n> Answer:")
        print(answer)
        # Print the relevant source which was used for answerinng
        for document in docs:
            print("\n" + document.metadata["source"]+ ":")


if __name__=="__main__":
    main()