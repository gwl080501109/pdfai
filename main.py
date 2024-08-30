import streamlit as st
from utils import qa_agent
from langchain.memory import ConversationBufferMemory

st.title("海亮知识库")


with st.sidebar:
    mobile = st.text_input("请输入你的手机号")
    st.info("*******************")
if "memory" not in st.session_state:
    st.session_state["memory"] = ConversationBufferMemory(
        return_messages=True,
        memory_key="chat_history",
        output_key="answer"

    )
    st.session_state["messages"] = [{"role":"ai",
                                     "content":"你好，我是你的小助手，有什么可以帮你的吗"}]


upload_file = st.file_uploader("上传你的PDF文件: ", type="pdf")
question = st.text_input("对文件的内容进行提问吧", disabled=not upload_file)

if upload_file and question:
    with st.spinner("生成中,请稍等"):
        result = qa_agent(st.session_state["memory"], upload_file, question)
    st.write("###答案")
    st.write(result["answer"])

    st.session_state["chat_history"] = result["chat_history"]
if "chat_history" in st.session_state:
    with st.expander("历史消息"):
        for i in range(0, len(st.session_state["chat_history"]), 2):
            human_message = st.session_state["chat_history"][i]
            ai_message = st.session_state["chat_history"][i+1]
            st.write(human_message)
            st.write(ai_message)
            if i < len(st.session_state["chat_history"]) - 2:
                st.divider()