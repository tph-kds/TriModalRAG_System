import streamlit as st

@st.cache_resource
def load_model():
    pass

def get_page_config():
        
    st.set_page_config(
        page_title="TriModal Retrieval Augmented Generation Chatbot Using Streamlit",
        page_icon="🧊",
        layout="wide",
        initial_sidebar_state="expanded",
        menu_items={
            'Get Help': 'https://www.extremelycoolapp.com/help',
            'Report a bug': "https://www.extremelycoolapp.com/bug",
            'About': "# This is a header. This is an *extremely* cool app!"
        }
    )



# st.image("https://huggingface.co/datasets/huggingface-course/documentation-images/resolve/main/chatbot.png", width=200)

# st.text_input("Enter your message", key="input", placeholder="Enter your message")


def main():

    get_page_config()
    st.title('TriModal Retrieval Augmented  Chatbot Using Streamlit') 

    pass

if __name__ == "__main__":
    main()