import streamlit as st
from langchain.chat_models import ChatOpenAI
from langchain.chains import ConversationChain
from langchain.chains.conversation.memory import ConversationEntityMemory
from langchain.chains.conversation.prompt import ENTITY_MEMORY_CONVERSATION_TEMPLATE

# To run: streamlit run --server.port 8502 MemoryBot.py
# Initialize session states
if "generated" not in st.session_state:
    st.session_state["generated"] = []  # Wyjście
if "past" not in st.session_state:
    st.session_state["past"] = []  # Przeszłość
if "input" not in st.session_state:
    st.session_state["input"] = ""
if "stored_session" not in st.session_state:
    st.session_state["stored_session"] = []

# Define function to get user input
def get_text():
    """
    Gets text from the user.
    Returns:
        (str): User-entered text
    """
    input_text = st.text_input("You: ", st.session_state["input"], key="input",
                               placeholder="Your AI assistant here! Ask me anything ...",
                               label_visibility='hidden')
    return input_text

# Define function to start a new chat
def new_chat():
    """
    Clears the session state and starts a new chat
    """
    save = []
    for i in range(len(st.session_state['generated'])-1, -1, -1):
        save.append("User:" + st.session_state["past"][i])
        save.append("Bot:" + st.session_state["generated"][i])
    st.session_state["stored_session"].append(save)
    st.session_state["generated"] = []
    st.session_state["past"] = []
    st.session_state["input"] = ""
    st.session_state.entity_memory.store({})
    st.session_state.entity_memory.buffer.clear()

st.title("Memory Bot")

# API
api = st.sidebar.text_input("API-Key", type="password")
MODEL = st.sidebar.selectbox(label='Model', options=['gpt-3.5-turbo'])
if api:

    # Creating a ChatOpenAI instance
    chat_model = ChatOpenAI(
        temperature=0,
        openai_api_key=api,
        model_name=MODEL,
    )

    # Create conversation memory
    if 'entity_memory' not in st.session_state:
        st.session_state.entity_memory = ConversationEntityMemory(llm=chat_model, k=10)

    # Create the Conversation Chain
    conversation = ConversationChain(
        llm=chat_model,
        prompt=ENTITY_MEMORY_CONVERSATION_TEMPLATE,
        memory=st.session_state.entity_memory
    )
else:
    st.error("No API found")

if st.sidebar.button("New Chat", on_click=new_chat, type='primary'):
    pass

#Get the user input
user_input = get_text()

#Generate the output using the ConversationChain object and the user input, and add
if user_input:
    output = conversation.run(input=user_input)

    st.session_state["past"].append(user_input)
    st.session_state["generated"].append(output)

with st.expander("Conversation"):
    for i in range(len(st.session_state['generated'])-1, -1, -1):
        st.info(st.session_state["past"][i])
        st.success(st.session_state["generated"][i])