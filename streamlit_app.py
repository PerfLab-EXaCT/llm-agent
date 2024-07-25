import streamlit as st
import torch
import transformers
from langchain_huggingface import HuggingFacePipeline
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
import bs4
from langchain_community.document_loaders import WebBaseLoader

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.cluster import KMeans


### Statefully manage chat history ###
store = {}


@st.cache_resource
def load_files():
    loader = WebBaseLoader(
        web_paths=("https://dl.acm.org/doi/fullHtml/10.1145/3624062.3624064",),
        bs_kwargs=dict(
            parse_only=bs4.SoupStrainer(
                class_=("journal-title", "authorGroup", "abstract", "body")
            )
        ),
    )
    return loader.load()


@st.cache_resource
def get_retriever(HF_TOKEN=None, embedding_model_id=None):
    docs = load_files()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # https://huggingface.co/spaces/mteb/leaderboard
    model_kwargs = {'device': device, 'trust_remote_code' : True, 'token': HF_TOKEN}
    encode_kwargs = {'normalize_embeddings': False}
    embedding_model = HuggingFaceEmbeddings(
        model_name=embedding_model_id,
        model_kwargs=model_kwargs,
        encode_kwargs=encode_kwargs
    )

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    splits = text_splitter.split_documents(docs)
    vectorstore = Chroma.from_documents(documents=splits, embedding=embedding_model)
    return vectorstore.as_retriever()


@st.cache_resource
def get_llm(HF_TOKEN=None, llm_model_id=None):
    # set quantization configuration to load large model with less GPU memory
    # https://huggingface.co/blog/4bit-transformers-bitsandbytes
    # https://colab.research.google.com/drive/1ge2F1QSK8Q7h0hn3YKuBCOAS0bK8E0wf?usp=sharing
    bnb_config = transformers.BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True
    )

    tokenizer = transformers.AutoTokenizer.from_pretrained(
        llm_model_id,
        token=HF_TOKEN
    )

    model = transformers.AutoModelForCausalLM.from_pretrained(
        llm_model_id,
        device_map='auto',
        quantization_config=bnb_config,
        token=HF_TOKEN
    )

    terminators = [
        tokenizer.eos_token_id,
        tokenizer.convert_tokens_to_ids("<|eot_id|>")
    ]

    pipe = transformers.pipeline(
        task='text-generation',
        model=model,
        tokenizer=tokenizer,
        eos_token_id=terminators,
        pad_token_id=tokenizer.eos_token_id,
        return_full_text=False,  # langchain expects the full text
        # model parameters, optional, added so it's easier to modify in the future
        temperature=0.6,  # 'randomness' of outputs, 0.0 is the min and 1.0 the max #Default 0.6 Try 0.8
        max_new_tokens=512,  # max number of tokens to generate in the output
        repetition_penalty=1.1  # without this output begins repeating
    )
    pipe.tokenizer.pad_token_id = model.config.eos_token_id

    return HuggingFacePipeline(pipeline=pipe)


@st.cache_resource
def get_chain(HF_TOKEN=None, llm_model_id=None, embedding_model_id=None):
    retriever = get_retriever(HF_TOKEN=HF_TOKEN, embedding_model_id=embedding_model_id)
    llm = get_llm(HF_TOKEN=HF_TOKEN, llm_model_id=llm_model_id)

    ### Contextualize question ###
    contextualize_q_system_prompt = """<|begin_of_text|><|start_header_id|>system<|end_header_id|>
        Given the following conversation and a follow up question,
        rephrase the follow up question to be a standalone question, in its original language.
        Please don't include any commentary; respond with only the standalone question. 

        Let me share a couple examples.

        If you do not see any chat history, you MUST return the "Follow Up Input" as is:
        ```
        Chat History:
        Follow Up Input: How is Lawrence doing?
        Standalone Question:
        How is Lawrence doing?
        ```

        If this is the second question onwards, you should properly rephrase the question like this:
        ```
        Chat History:
        Human: How is Lawrence doing?
        Assistant:
        Lawrence is injured and out for the season.
        Follow Up Input: What was his injury?
        Standalone Question:
        What was Lawrence's injury?
        ```

        Now, with those examples, here is the actual chat history and input question.

        Chat History:
        {chat_history}

        Follow Up Input: {input}

        Standalone Question:
        [your response here]
        <|eot_id|><|start_header_id|>assistant<|end_header_id|>
    """

    contextualize_q_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", contextualize_q_system_prompt),
        ]
    )
    history_aware_retriever = create_history_aware_retriever(
        llm, retriever, contextualize_q_prompt
    )

    ### Answer question ###
    system_prompt = """<|begin_of_text|><|start_header_id|>system<|end_header_id|>
        You are an assistant for question-answering tasks.
        Use the following pieces of retrieved context to answer the question.
        If the context is not relevant, ignore the retrieved context. Don't need to incorporate it into your response.
        Instead, please think rationally and answer from your own knowledge base.
        If the context is not relevant, you do not need to state that it's not relevant.
        If you are answering from your own knowledge base, you do not need to state that you are answering from your own knowledge base.
        Please just respond with the answer, no commentary necessary.

        Context (may NOT be relevant): {context}

        Chat History:
        {chat_history} 
        <|eot_id|><|start_header_id|>user<|end_header_id|>
        {input}
        <|eot_id|><|start_header_id|>assistant<|end_header_id|>
    """

    qa_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
        ]
    )
    question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)
    rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)

    # Session-specific message storage
    class SessionStorage:
        def __init__(self):
            self.chat_history = []

        def add_message(self, message):
            self.chat_history.append(message)

        def get_chat_history(self):
            return self.chat_history


    def get_session_history(session_id: str) -> BaseChatMessageHistory:
        if session_id not in store:
            store[session_id] = ChatMessageHistory()
        return store[session_id]


    conversational_rag_chain = RunnableWithMessageHistory(
        rag_chain,
        get_session_history,
        input_messages_key="input",
        history_messages_key="chat_history",
        output_messages_key="answer",
        return_source_documents=True,
    )

    return conversational_rag_chain


def ask_question(chain, query):
    response = chain.invoke(
        {"input": query},
        config={"configurable": {"session_id": "default"}}
    )
    return response


def show_ui(chain, prompt_to_user=None):
    if "messages" not in st.session_state.keys():
        st.session_state.messages = [{"role": "assistant", "content": prompt_to_user}]

    # Display chat messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.write(message["content"])

    # User-provided prompt
    if query := st.chat_input():
        st.session_state.messages.append({"role": "user", "content": query})
        with st.chat_message("user"):
            st.write(query)

    if "prompted" in st.session_state.keys():
        show_options(datasets.load_iris())

    # Generate a new response if last message is not from assistant
    if st.session_state.messages[-1]["role"] != "assistant":
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                response = ask_question(chain, query)
                st.markdown(response["answer"])
        message = {"role": "assistant", "content": response["answer"]}
        st.session_state.messages.append(message)
        if "prompted" not in st.session_state:
            st.session_state["prompted"] = True
            show_options(datasets.load_iris())


def show_options(dataset):
    with st.sidebar:
        st.header("Data Interpretation Techniques")
        with st.expander("K-Means Clustering"):
            user_selected_k = st.slider("K Value", 1, 10, 2)
            st.write("K: ", user_selected_k)

            X = dataset.data
            y = dataset.target

            # Apply K-Means clustering
            kmeans = KMeans(n_clusters=user_selected_k)
            kmeans.fit(X)

            # Get the cluster labels
            labels = kmeans.labels_

            # Visualize the clusters
            fig = plt.figure()
            plt.scatter(X[:, 0], X[:, 1], c=labels)
            plt.xlabel('Sepal Length')
            plt.ylabel('Sepal Width')
            plt.title("K-Means Clustering with K = " + str(user_selected_k))
            fig.patch.set_facecolor('none')
            st.pyplot(fig) # instead of plt.show()

            
def get_secret_or_input(secret_key, secret_name, info_link=None):
    if secret_key in st.secrets:
        st.write("Found %s secret" % secret_key)
        secret_value = st.secrets[secret_key]
    else:
        st.write(f"Please provide your {secret_name}")
        secret_value = st.text_input(secret_name, key=f"input_{secret_key}", type="password")
        if secret_value:
            st.session_state[secret_key] = secret_value
        if info_link:
            st.markdown(f"[Get an {secret_name}]({info_link})")
    return secret_value


def main():
    st.set_page_config(page_title="LLM Agent Demo", page_icon="pnnl-logo-1.png")
    st.title("LLM Agent Demo")
    st.logo("pnnl-logo-2.png")

    if "initialized" not in st.session_state or not st.session_state.initialized:
        ready = True

        HF_TOKEN = st.session_state.get("HF_TOKEN")

        with st.sidebar:
            if not HF_TOKEN:
                HF_TOKEN = get_secret_or_input('HF_TOKEN', "HuggingFace Hub API Token",
                                                            info_link="https://huggingface.co/docs/huggingface_hub/main/en/quick-start#authentication")

        if not HF_TOKEN:
            st.warning("Missing HF_TOKEN")
            ready = False

        if ready:
            llm_model_id = "meta-llama/Meta-Llama-3-8B-Instruct"
            embedding_model_id = "Linq-AI-Research/Linq-Embed-Mistral"
            chain = get_chain(HF_TOKEN=HF_TOKEN, llm_model_id=llm_model_id, embedding_model_id=embedding_model_id)
            if "chain" not in st.session_state:
                st.session_state["chain"] = chain
        else:
            st.stop()

        st.session_state.initialized = True  # Mark initialization as done to avoid re-running

    uploaded_files = st.file_uploader("Upload File(s) (Optional)", accept_multiple_files=True)
    if uploaded_files is not None:
        for uploaded_file in uploaded_files:
            st.write("File Accepted: ", uploaded_file.name)

    st.subheader("Ask me questions about scientific images")
    show_ui(st.session_state.chain, "How may I help you?")


if __name__ == "__main__":
    main()