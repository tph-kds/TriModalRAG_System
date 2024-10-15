import os
import streamlit as st
import dotenv
from PIL import Image
from audio_recorder_streamlit import audio_recorder
import base64
from io import BytesIO
import random
import google.generativeai as genai
import time
from serving.streamlit_app.settings import (
    result_scenarios, 
    set_up_api_config, 
    settup_config_llm
)

dotenv.load_dotenv()
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))

google_models = [
    "gemini-1.5-flash-001",
    "gemini-1.5-flash",
    "gemini-1.5-pro",
]

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

# Function to convert the messages format from OpenAI and Streamlit to Gemini
def messages_to_gemini(messages):
    gemini_messages = []
    prev_role = None
    for message in messages:
        if prev_role and (prev_role == message["role"]):
            gemini_message = gemini_messages[-1]
        else:
            gemini_message = {
                "role": "model" if message["role"] == "assistant" else "user",
                "parts": [],
            }

        for content in message["content"]:
            if content["type"] == "text":
                gemini_message["parts"].append(content["text"])
            elif content["type"] == "image_url":
                gemini_message["parts"].append(base64_to_image(content["image_url"]["url"]))
            elif content["type"] == "audio_file":
                gemini_message["parts"].append(genai.upload_file(content["audio_file"]))

        if prev_role != message["role"]:
            gemini_messages.append(gemini_message)

        prev_role = message["role"]
        
    return gemini_messages


# Function to convert the messages format from OpenAI and Streamlit to Anthropic (the only difference is in the image messages)



# Function to query and stream the response from the LLM
def stream_llm_response(inputs, model_params, model_type="google", api_inputs=None):

    response_message = ""
    llm_config = settup_config_llm(model_params)
    api_config = set_up_api_config(api_inputs)
    
    if model_type == "google":
        ai_response, meta_repsonse = result_scenarios(
            question_str=inputs["text"],
            image_url=inputs["image"],
            video_url=inputs["audio"],
            query=inputs["query"], 
            serving_format="streamlit", 
            api_config=api_config, 
            llm_setup=llm_config
        )
        response_message = ai_response
        gemini_messages = messages_to_gemini(st.session_state.messages)




    st.session_state.messages.append({
        "role": "assistant", 
        "content": [
            {
                "type": "text",
                "text": response_message,
            }
        ]})


# Function to convert file to base64
def get_image_base64(image_raw):
    buffered = BytesIO()
    image_raw.save(buffered, format=image_raw.format)
    img_byte = buffered.getvalue()

    return base64.b64encode(img_byte).decode('utf-8')

def file_to_base64(file):
    with open(file, "rb") as f:

        return base64.b64encode(f.read())

def base64_to_image(base64_string):
    base64_string = base64_string.split(",")[1]
    
    return Image.open(BytesIO(base64.b64decode(base64_string)))

def init():
        pdf_upload = None
        image_path = None
        audio_path = None
        pdf_target_url = None
        image_target_url = None
        audio_target_url = None
        pdf_file_added = False
        image_file_added = False
        audio_file_added = False
        return pdf_upload, image_path, audio_path, pdf_target_url, image_target_url, audio_target_url, pdf_file_added, image_file_added, audio_file_added


def main():

    # --- Page Config ---
    get_page_config()     
    pdf_upload = None
    image_path = None
    audio_path = None
    pdf_target_url = None
    image_target_url = None
    audio_target_url = None
    pdf_file_added = False
    image_file_added = False
    audio_file_added = False

    # --- Header ---
    st.html("""<h1 style="text-align: center; color : #6ca395">TriModal Retrieval Augmented  Chatbot Using Streamlit</h1>""") 

    # --- Side Bar ---
    with st.sidebar:

        st.header("> TOKEN API KeEYS")

        # with cols_keys[0]:
        default_google_api_key = os.getenv("GOOGLE_API_KEY") if os.getenv("GOOGLE_API_KEY") is not None else ""  # only for development environment, otherwise it should return None
        with st.popover("🔐 Google"):
            google_api_key = st.text_input("Introduce your Google API Key (https://aistudio.google.com/app/apikey)", value=default_google_api_key, type="password")

        default_langchain_api_key = os.getenv("LANGCHAIN_API_KEY") if os.getenv("LANGCHAIN_API_KEY") is not None else ""  # only for development environment, otherwise it should return None
        with st.popover("🔐 LangChain"):
            langchain_api_key = st.text_input("Introduce your LangChain API Key (https://aistudio.google.com/app/apikey)", value=default_langchain_api_key, type="password")


    # --- Main Content ---
    # Checking if the user has introduced the OpenAI API Key, if not, a warning is displayed
    if ((google_api_key == "" or google_api_key is None) or (langchain_api_key == "" or langchain_api_key is None)):
    # if (openai_api_key == "" or openai_api_key is None or "sk-" not in openai_api_key) and (google_api_key == "" or google_api_key is None) and (anthropic_api_key == "" or anthropic_api_key is None):
        st.write("#")
        st.warning("⬅️ Please introduce an API Key to continue...")

        with st.sidebar:
            st.write("#")
            st.write("#")
            st.write(" 📋[How to use this Streamlit App ](https://github.com/tph-kds/TriModalRAG_System/tree/main/serving/guide_application.md)")   

    else:
        # client = OpenAI(api_key=openai_api_key)
        if "info" not in st.session_state:
            st.session_state.info = {}
        if "messages" not in st.session_state:
            st.session_state.messages = []

        # Displaying the previous messages if there are any
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                for content in message["content"]:
                    if content["type"] == "text":
                        st.write(content["text"])
                    elif content["type"] == "image_url":      
                        st.image(content["image_url"]["url"])
                    elif content["type"] == "pdf_file":
                        st.write(f" Upload a PDF file successfully with an url: {content['pdf_url']['url']}")
                    elif content["type"] == "audio_record":
                        st.audio(content["audio_record"])
                    elif content["type"] == "audio_url":
                        st.write(f" Upload a audio file successfully with an url: {content['audio_url']['url']}")




        # Side bar model options and inputs
        with st.sidebar:

            st.divider()
            
            # available_models = [] + (anthropic_models if anthropic_api_key else []) + (google_models if google_api_key else []) + (openai_models if openai_api_key else [])
            available_models = [] + (google_models if google_api_key else [])
            model = st.selectbox("Select a model:", available_models, index=0)
            model_type = None
            if model.startswith("gemini"): model_type = "google"
            # if model.startswith("gpt"): model_type = "openai"
            # elif model.startswith("gemini"): model_type = "google"
            # elif model.startswith("claude"): model_type = "anthropic"
            
            with st.popover("⚙️ Model parameters"):
                model_temp = st.slider("Temperature", min_value=0.0, max_value=2.0, value=0.3, step=0.1)
                model_tokens = st.slider("Max tokens", min_value=0, max_value=4096, value=1024, step=64)
                model_retries = st.slider("Max retries", min_value=0, max_value=10, value=3, step=1)
                model_stop = st.text_input("Stop sequence", value="\n")

            # audio_response = st.toggle("Audio response", value=False)
            # if audio_response:
            #     cols = st.columns(2)
            #     with cols[0]:
            #         tts_voice = st.selectbox("Select a voice:", ["alloy", "echo", "fable", "onyx", "nova", "shimmer"])
            #     with cols[1]:
            #         tts_model = st.selectbox("Select a model:", ["tts-1", "tts-1-hd"], index=1)

            model_params = {
                "model_name": model,
                "temperature": model_temp,
                "max_tokens": model_tokens,
                "max_retries": model_retries,
                "stop": model_stop,
            }


            def reset_conversation():
                if "messages" in st.session_state and len(st.session_state.messages) > 0:
                    st.session_state.pop("messages", None)

                

            reset_conversation_col = st.button(
                "🗑️ Reset conversation", 
                on_click=reset_conversation,
            )
            # if reset_conversation_col:
            #     pdf_upload = None
            #     image_path = None
            #     audio_path = None

            st.divider()
            # File Upload

            # Image Upload
            if model in ["gemini-1.5-flash-001", "gemini-1.5-flash", "gemini-1.5-pro"]:
                    
                # st.write(f"### **🖼️ Add an image{' or a video file' if model_type=='google' else ''}:**")
                def add_pdf_to_messages():
                    # if st.session_state.uploaded_pdf:
                    #     pdf_type = st.session_state.uploaded_pdf.type if st.session_state.uploaded_pdf else "file/pdf"
                    if st.session_state.uploaded_pdf:
                        # pdf_file_added = True
                        full_path_pdf = ROOT_DIR + f"\\data\\pdf\\pdf_{st.session_state.uploaded_pdf.name}" 
                        # with open(full_path_pdf, "wb") as f:
                        #     f.write(st.session_state.uploaded_pdf.getbuffer())
                        # time.sleep(1)
                        st.session_state.messages.append(
                            {
                                "role": "user", 
                                "content": [{
                                    "type": "pdf_file",
                                    "pdf_url": {"url": f"{full_path_pdf}"}
                                }]
                            }
                        )
                        st.session_state.info["pdf_target_url"] = full_path_pdf

                def add_image_to_messages():
                    if st.session_state.uploaded_img or ("camera_img" in st.session_state and st.session_state.camera_img):
                        img_type = st.session_state.uploaded_img.type if st.session_state.uploaded_img else "image/jpeg"

                        #     # save the image file
                        # if st.session_state.uploaded_img:
                        #     full_path_image = ROOT_DIR +  f"\\data\\image\\image_{st.session_state.uploaded_img.name}" 
                        #     with open(full_path_image, "wb") as f:
                        #         f.write(st.session_state.uploaded_img.getbuffer())
                        #     time.sleep(1)
                        raw_img = Image.open(st.session_state.uploaded_img or st.session_state.camera_img)
                        img = get_image_base64(raw_img)
                        st.session_state.messages.append(
                            {
                                "role": "user", 
                                "content": [{
                                    "type": "image_url",
                                    "image_url": {"url": f"data:{img_type};base64,{img}"}
                                }]
                            }
                        )
                def add_audio_to_messages():
                    if st.session_state.uploaded_audio_file:
                        # audio_file_added = True
                        full_path_audio = ROOT_DIR +  f"\\data\\audio\\audio_{st.session_state.uploaded_audio_file.name}" 
                        # with open(full_path_audio, "wb") as f:
                        #     f.write(st.session_state.uploaded_audio_file.getbuffer())
                        # time.sleep(1)
                        st.session_state.messages.append(
                            {
                                "role": "user", 
                                "content": [{
                                    "type": "audio_url",
                                    "audio_url": {"url": f"{full_path_audio}"}
                                }]
                            }
                        )
                # PDF Uploader
                st.write(f"### **📖 Add a pdf file:**")
                with st.popover("📁 Upload"):
                    pdf_upload = st.file_uploader(
                        f"Upload an image:", 
                        type=["pdf"], 
                        accept_multiple_files=False,
                        key="uploaded_pdf",
                        on_change=add_pdf_to_messages,
                    )
                    # Save the uploaded pdf on the local machine
                    if pdf_upload:
                        pdf_file_added = True
                        full_path_pdf = ROOT_DIR + f"\\data\\pdf\\pdf_{pdf_upload.name}" 
                        with open(full_path_pdf, "wb") as f:
                            f.write(pdf_upload.getbuffer())
                        time.sleep(1)
                    #     st.session_state.messages.append(
                    #         {
                    #             "role": "user", 
                    #             "content": [{
                    #                 "type": "pdf_file",
                    #                 "pdf_url": {"url": f"{full_path_pdf}"}
                    #             }]
                    #         }
                    #     )
                    #     st.session_state.info["pdf_target_url"] = full_path_pdf

                st.write(f"pdf_upload: {pdf_upload.name if pdf_upload else None}")
                
                st.divider()

                # Image Uploader
                image_file_added = False
                image_option = st.toggle("Image option", value=False)
                if image_option:
                    st.write(f"### **🖼️ Add an image{' or a video file' if model_type=='google' else ''}:**")
                    image_upload = None
                    with st.popover("📁 Upload"):
                        image_upload = st.file_uploader(
                            f"Upload an image{' or a video' if model_type == 'google' else ''}:", 
                            type=["png", "jpg", "jpeg"] + (["mp4"] if model_type == "google" else []), 
                            accept_multiple_files=False,
                            key="uploaded_img",
                            on_change=add_image_to_messages,
                        )
                        if image_upload:
                            full_path_image = ROOT_DIR +  f"\\data\\image\\image_{image_upload.name}" 
                            with open(full_path_image, "wb") as f:
                                f.write(image_upload.getbuffer())
                            time.sleep(1)
                        # image_path = image_upload.name if image_upload else None
                            st.session_state.info["image_target_url"] = full_path_image

                    if image_upload:
                        image_file_added = True

                    st.write(f"image_upload: {image_path if image_upload else None}")

                else:
                    st.write(f"### ** Add a video file:**")
                    activate_camera = False
                    with st.popover("📸 Camera"):
                        activate_camera = st.checkbox("Activate camera")
                        if activate_camera:
                            image_camera = st.camera_input(
                                "Take a picture", 
                                key="camera_img",
                                on_change=add_image_to_messages,
                            )

                            if image_camera:
                                image_file_added = True
                        # print(activate_camera)
                                full_path_image = ROOT_DIR +  f"\\data\\image\\image_{image_upload.name}" 

                                # image_path = image_camera.name if activate_camera else None
                                st.session_state.info["image_target_url"] = full_path_image

                    
                    st.write(f"image_upload: {image_path if activate_camera else None}")


            st.divider()
            # Audio Upload
            st.write("#")
            st.write(f"### **🎤 Add an audio:**")

            audio_prompt = None
            audio_file_added = False
            if "prev_speech_hash" not in st.session_state:
                st.session_state.prev_speech_hash = None


            audio_check_option = st.toggle("Audio Option", value=False)
            audio_path = None
            if audio_check_option:
                col_name = st.columns([3, 2])
                with col_name[0]:
                    speech_input = audio_recorder(
                        text="Record:", 
                        icon_size="2x",
                        neutral_color="#6ca395",
                        icon_name="headset",
                    )
                    if speech_input and st.session_state.prev_speech_hash != hash(speech_input):
                        # Save the audio to a temporary file and define the path
                        # import tempfile
                        # with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as temp_file:
                        #     temp_file.write(speech_input)
                        #     audio_path = temp_file.name  # Define the path to the temporary file

                        st.session_state.prev_speech_hash = hash(speech_input)
                        if model_type == "google":
                            # save the audio file
                            audio_id = random.randint(100000, 999999)
                            audio_path = ROOT_DIR + f"\\data\\audio\\audio_{audio_id}.mp3" 
                            with open(audio_path, "wb") as f:
                                f.write(speech_input)

                            time.sleep(1)
                            if audio_path:
                                st.session_state.messages.append(
                                    {
                                        "role": "user", 
                                        "content": [{
                                            "type": "audio_record",
                                            "audio_record": f"{audio_path}",
                                        }]
                                    }
                                )
                                st.session_state.info["audio_target_url"] = audio_path

                                audio_file_added = True
                        # st.audio(speech_input, format="audio/mp3", loop=True)
                        # st.write(f"audio_upload:  {f'{audio_path}' if speech_input else None}")

                with col_name[1]:
                        # Delete the audio file after user confirms
                    if st.button("Delete audio file"):
                        if audio_path is None:
                            st.write("No audio file to delete.")
                        elif os.path.exists(audio_path):
                            os.remove(audio_path)
                            st.session_state.prev_speech_hash = None
                            st.session_state.messages = []
                            st.write(f"Audio file {audio_path} deleted successfully!")
                        else:
                            st.write("File not found.")
            else:

                st.write("#")
                st.write(f"### **Or upload an audio:**")
                with st.popover("🔈 Upload"):
                        audio_upload = st.file_uploader(
                            f"Upload an audio file:", 
                            type=["mp3", "wav"] , 
                            accept_multiple_files=False,
                            key="uploaded_audio_file",
                            on_change=add_audio_to_messages,
                        )
                        if audio_upload:
                            audio_file_added = True
                            full_path_audio = ROOT_DIR +  f"\\data\\audio\\audio_{audio_upload.name}" 
                            with open(full_path_audio, "wb") as f:
                                f.write(audio_upload.getbuffer())
                            time.sleep(1)
                        # audio_path = audio_upload.name if audio_upload else None
                            st.session_state.info["audio_target_url"] = full_path_audio

                        speech_input = True
            # st.divider()
            st.write(f"audio_upload:  {f'{audio_path}' if speech_input else None}")

            st.divider()
            # --- LLM Response ---

        if prompt := st.chat_input("Hi! Ask me anything..."):
            # if not audio_file_added:
            st.session_state.messages.append(
                {
                    "role": "user", 
                    "content": [{
                        "type": "prompt",
                        "prompts": {
                            "prompt" : f"{prompt}:\n\n" ,
                             "prompt_pdf" : f" PDF: {st.session_state.info['pdf_target_url']} \n ",
                            "prompt_image" : f" Image: {st.session_state.info['image_target_url']}\n ",
                            "prompt_audio" : f" Audio: {st.session_state.info['audio_target_url']}\n ", 
                        }
                    }]
                }
            )
            
            # Display the new messages
            with st.chat_message("user"):
                st.markdown("Prompt which created from your inputs: \n\n" + \
                         st.session_state.messages[-1]["content"][0]["prompts"]["prompt_pdf"] + \
                         "\n\n" + \
                         st.session_state.messages[-1]["content"][0]["prompts"]["prompt_image"] + \
                         "\n\n" + \
                         st.session_state.messages[-1]["content"][0]["prompts"]["prompt_audio"])

            # else:
            #     # Display the audio file
            #     with st.chat_message("user"):
            #         st.audio(st.session_state.info["audio_target_url"], format="audio/mp3", loop=False, )

            with st.chat_message("assistant"):
                model2key = {
                    "GOOGLE_API_KEY": google_api_key, 
                    "LANGCHAIN_API_KEY" : langchain_api_key, 
                    "LANGCHAIN_ENDPOINT" : os.getenv("LANGCHAIN_ENDPOINT"), 
                    "LANGCHAIN_TRACING_V2" : os.getenv("LANGCHAIN_TRACING_V2"),
                    "LANGCHAIN_PROJECT" : os.getenv("LANGCHAIN_PROJECT"),
                }

                target_url = st.session_state.info
                print("Hung", prompt)
                inputs = {
                    "text": target_url["pdf_target_url"] if pdf_file_added else None,
                    "image": target_url["image_target_url"] if image_file_added else None,
                    "audio": target_url["audio_target_url"] if audio_file_added else None,
                    "query": prompt,
                }
                st.write_stream(
                    stream_llm_response(
                        inputs = inputs,
                        model_params=model_params, 
                        model_type=model_type, 
                        api_inputs=model2key
                    )
                )

            # --- Added Audio Response (optional) ---
            # if audio_response:
                # response =  client.audio.speech.create(
                #     model=tts_model,
                #     voice=tts_voice,
                #     input=st.session_state.messages[-1]["content"][0]["text"],
                # )
                # audio_base64 = base64.b64encode(response.content).decode('utf-8')
                # audio_html = f"""
                # <audio controls autoplay>
                #     <source src="data:audio/wav;base64,{audio_base64}" type="audio/mp3">
                # </audio>
                # """
                # st.html(audio_html)
                # st.audio("cat-purr.mp3", format="audio/mpeg", loop=True)



if __name__=="__main__":
    # reset data folder
    folder_main = ROOT_DIR + "\\" + "data"
    for filename in os.listdir(folder_main):
        file_path = os.path.join(folder_main, filename)
        for fp in os.listdir(file_path):
            if os.path.isfile(os.path.join(file_path, fp)) or os.path.islink(os.path.join(file_path, fp)):
                os.remove(os.path.join(file_path, fp))
    main()