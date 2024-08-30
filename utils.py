from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.llms.tongyi import Tongyi
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.chains import ConversationalRetrievalChain
import dashscope
from http import HTTPStatus
from langchain.embeddings import HuggingFaceEmbeddings
from sentence_transformers import SentenceTransformer

# 设置API密钥
DASHSCOPE_API_KEY = "sk-7cb7535b25a54d5a8f3af0066af95fd3"

model = Tongyi(
    model_name="qwen-turbo",
    max_tokens=1024,
    top_p=1,
    frequency_penalty=0,
    presence_penalty=0,
    api_key=DASHSCOPE_API_KEY
)

def qa_agent(memory, uploaded_file, question):
    # 对用户上传文档进行读取
    file_content = uploaded_file.read()
    temp_file_path = "temp.pdf"
    with open(temp_file_path, "wb") as temp_file:
        temp_file.write(file_content)

    loader = PyPDFLoader(temp_file_path)
    docs = loader.load()

    # 分隔符
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=50,
        separators=["\n", "。", "！", "？", "，", "、"]
    )

    # 文本分割
    texts = text_splitter.split_documents(docs)
    # 使用预训练的Sentence-BERT模型
    model_name = "sentence-transformers/all-MiniLM-L6-v2"
    embeddings = HuggingFaceEmbeddings(model_name=model_name)
    # 向量数据库
    db = FAISS.from_documents(texts, embeddings)

    # 检索器
    retriever = db.as_retriever()
    qa = ConversationalRetrievalChain.from_llm(
        llm=model,
        retriever=retriever,
        memory=memory
    )

    response = qa.invoke({"chat_history": memory, "question": question})
    return response