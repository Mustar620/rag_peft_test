import streamlit as st
from langchain_core.messages import ChatMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_community.chat_models import ChatOllama
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.document_loaders import PyPDFLoader
from langchain.embeddings import HuggingFaceEmbeddings
from huggingface_hub import login

# PDF 파일 로드 및 분할
loader = PyPDFLoader("data\컴퓨터소프트웨어학과.pdf")
pages = loader.load_and_split()

# 텍스트 분할
text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
texts = text_splitter.split_documents(pages)

# HuggingFace Embeddings 설정
token_key = "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"
login(token=token_key)
model_name = "jhgan/ko-sbert-nli"
model_kwargs = {'device': 'cpu'}
encode_kwargs = {'normalize_embeddings': True}
hf = HuggingFaceEmbeddings(
    model_name=model_name,
    model_kwargs=model_kwargs,
    encode_kwargs=encode_kwargs
)

# 벡터스토어 생성
docsearch = Chroma.from_documents(texts, hf)
retriever = docsearch.as_retriever(search_type="mmr", search_kwargs={'k': 3, 'fetch_k': 10})

# ChatPromptTemplate 설정
template = """Answer the question based only on the following context:
{context}

Question: {question}
"""
prompt = ChatPromptTemplate.from_template(template)

# Streamlit 설정
st.set_page_config(page_title="피드백 데모", page_icon="🚀")
st.title("🚀 (peft+rag)customized model🚀")

# 초기 세션 상태 설정
if "messages" not in st.session_state:
    st.session_state["messages"] = []

if "store" not in st.session_state:
    st.session_state["store"] = dict()

# 메시지 출력 함수
def print_message():
    if 'messages' in st.session_state and len(st.session_state['messages']) > 0:
        for chatmessage in st.session_state['messages']:
            st.chat_message(chatmessage.role).write(chatmessage.content)

# 세션별 채팅 히스토리 가져오기 함수
def get_session_history(session_ids: str) -> BaseChatMessageHistory:
    if session_ids not in st.session_state["store"]:
        st.session_state["store"][session_ids] = ChatMessageHistory()
    return st.session_state["store"][session_ids]

print_message()

# 사용자의 입력 처리
if user_input := st.chat_input("메세지를 입력해 주세요"):
    # 사용자가 입력한 내용 출력
    st.chat_message("user").write(f"{user_input}")
    st.session_state['messages'].append(ChatMessage(role='user', content=user_input))

    # LLM 설정
    llm = ChatOllama(model="Llama3_ko_8b_q5:latest")

    # 검색 결과를 context로 설정
    context_documents = retriever.get_relevant_documents(user_input)
    context = "\n".join([doc.page_content for doc in context_documents])

    # 메시지 히스토리를 포함한 체인 생성
    chain = prompt | llm
    chain_with_memory = RunnableWithMessageHistory(
        chain,
        get_session_history,
        input_messages_key="question",
        history_messages_key="history",
    )

    # 체인 실행
    response = chain_with_memory.invoke(
        {"context": context, "question": user_input},
        config={"configurable": {"session_id": "abc123"}}
    )
    msg = response.content

    # AI 답변 출력
    with st.chat_message("assistant"):
        st.write(msg)
        st.session_state['messages'].append(ChatMessage(role='assistant', content=msg))
