import os
import streamlit as st
import dotenv
from PIL import Image
from audio_recorder_streamlit import audio_recorder
import base64
from io import BytesIO
import random
import google.generativeai as genai
from serving.settings import (
    result_scenarios, 
    set_up_api_config, 
    settup_config_llm
)

# dotenv.load_dotenv()


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
    llm_config=settup_config_llm(model_params)
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



def main():

    # --- Page Config ---
    get_page_config()     


    # --- Header ---
    st.html("""<h1 style="text-align: center; color : #6ca395">TriModal Retrieval Augmented  Chatbot Using Streamlit</h1>""") 

    # --- Side Bar ---
    with st.sidebar:

        st.header("> TOKEN API KeEYS")
        cols_keys = st.columns(2)
        # with cols_keys[0]:
        #     default_openai_api_key = os.getenv("OPENAI_API_KEY") if os.getenv("OPENAI_API_KEY") is not None else ""  # only for development environment, otherwise it should return None
        #     with st.popover("🔐 OpenAI"):
        #         openai_api_key = st.text_input("Introduce your OpenAI API Key (https://platform.openai.com/)", value=default_openai_api_key, type="password")
        
        with cols_keys[0]:
            default_google_api_key = os.getenv("GOOGLE_API_KEY") if os.getenv("GOOGLE_API_KEY") is not None else ""  # only for development environment, otherwise it should return None
            with st.popover("🔐 Google"):
                google_api_key = st.text_input("Introduce your Google API Key (https://aistudio.google.com/app/apikey)", value=default_google_api_key, type="password")

        # default_anthropic_api_key = os.getenv("ANTHROPIC_API_KEY") if os.getenv("ANTHROPIC_API_KEY") is not None else ""
        # with st.popover("🔐 Anthropic"):
        #     anthropic_api_key = st.text_input("Introduce your Anthropic API Key (https://console.anthropic.com/)", value=default_anthropic_api_key, type="password")
    
    # --- Main Content ---
    # Checking if the user has introduced the OpenAI API Key, if not, a warning is displayed
    if google_api_key == "" or google_api_key is None:
    # if (openai_api_key == "" or openai_api_key is None or "sk-" not in openai_api_key) and (google_api_key == "" or google_api_key is None) and (anthropic_api_key == "" or anthropic_api_key is None):
        st.write("#")
        st.warning("⬅️ Please introduce an API Key to continue...")

        with st.sidebar:
            st.write("#")
            st.write("#")
            st.write(" 📋[How to use this Streamlit App ](https://github.com/tph-kds/TriModalRAG_System/tree/main/serving/guide_application.md)")   

    else:
        # client = OpenAI(api_key=openai_api_key)

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
                    elif content["type"] == "video_file":
                        st.video(content["video_file"])
                    elif content["type"] == "audio_file":
                        st.audio(content["audio_file"])



        # Side bar model options and inputs
        with st.sidebar:

            st.divider()
            
            # available_models = [] + (anthropic_models if anthropic_api_key else []) + (google_models if google_api_key else []) + (openai_models if openai_api_key else [])
            available_models = [] + (google_models if google_api_key else [])
            model = st.selectbox("Select a model:", available_models, index=0)
            model_type = None
            if model.startswith("gpt"): model_type = "google"
            # if model.startswith("gpt"): model_type = "openai"
            # elif model.startswith("gemini"): model_type = "google"
            # elif model.startswith("claude"): model_type = "anthropic"
            
            with st.popover("⚙️ Model parameters"):
                model_temp = st.slider("Temperature", min_value=0.0, max_value=2.0, value=0.3, step=0.1)

            audio_response = st.toggle("Audio response", value=False)
            if audio_response:
                cols = st.columns(2)
                with cols[0]:
                    tts_voice = st.selectbox("Select a voice:", ["alloy", "echo", "fable", "onyx", "nova", "shimmer"])
                with cols[1]:
                    tts_model = st.selectbox("Select a model:", ["tts-1", "tts-1-hd"], index=1)

            model_params = {
                "model": model,
                "temperature": model_temp,
            }

            def reset_conversation():
                if "messages" in st.session_state and len(st.session_state.messages) > 0:
                    st.session_state.pop("messages", None)

            st.button(
                "🗑️ Reset conversation", 
                on_click=reset_conversation,
            )

            st.divider()
            # File Upload
            if model in ["gemini-1.5-flash", "gemini-1.5-pro"]:
                st.write(f"### **🖼️ Upload a pdf file :**")

            # Image Upload
            if model in ["gemini-1.5-flash", "gemini-1.5-pro"]:
                    
                st.write(f"### **🖼️ Add an image{' or a video file' if model_type=='google' else ''}:**")
                
                def add_image_to_messages():
                    if st.session_state.uploaded_img or ("camera_img" in st.session_state and st.session_state.camera_img):
                        img_type = st.session_state.uploaded_img.type if st.session_state.uploaded_img else "image/jpeg"
                        if img_type == "video/mp4":
                            # save the video file
                            video_id = random.randint(100000, 999999)
                            with open(f"video_{video_id}.mp4", "wb") as f:
                                f.write(st.session_state.uploaded_img.read())
                            st.session_state.messages.append(
                                {
                                    "role": "user", 
                                    "content": [{
                                        "type": "video_file",
                                        "video_file": f"video_{video_id}.mp4",
                                    }]
                                }
                            )
                        else:
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

                cols_img = st.columns(2)

                with cols_img[0]:
                    with st.popover("📁 Upload"):
                        st.file_uploader(
                            f"Upload an image{' or a video' if model_type == 'google' else ''}:", 
                            type=["png", "jpg", "jpeg"] + (["mp4"] if model_type == "google" else []), 
                            accept_multiple_files=False,
                            key="uploaded_img",
                            on_change=add_image_to_messages,
                        )

                with cols_img[1]:                    
                    with st.popover("📸 Camera"):
                        activate_camera = st.checkbox("Activate camera")
                        if activate_camera:
                            st.camera_input(
                                "Take a picture", 
                                key="camera_img",
                                on_change=add_image_to_messages,
                            )

            # Audio Upload
            st.write("#")
            st.write(f"### **🎤 Add an audio{' (Speech To Text)' if model_type == 'openai' else ''}:**")

            audio_prompt = None
            audio_file_added = False
            if "prev_speech_hash" not in st.session_state:
                st.session_state.prev_speech_hash = None

            speech_input = audio_recorder("Press to talk:", icon_size="3x", neutral_color="#6ca395", )
            if speech_input and st.session_state.prev_speech_hash != hash(speech_input):
                st.session_state.prev_speech_hash = hash(speech_input)
                if model_type != "google":
                    transcript = client.audio.transcriptions.create(
                        model="whisper-1", 
                        file=("audio.wav", speech_input),
                    )

                    audio_prompt = transcript.text

                elif model_type == "google":
                    # save the audio file
                    audio_id = random.randint(100000, 999999)
                    with open(f"audio_{audio_id}.wav", "wb") as f:
                        f.write(speech_input)

                    st.session_state.messages.append(
                        {
                            "role": "user", 
                            "content": [{
                                "type": "audio_file",
                                "audio_file": f"audio_{audio_id}.wav",
                            }]
                        }
                    )

                    audio_file_added = True

            st.divider()
            # --- LLM Response ---
        # col = st.columns([1, 4])
        # with col[0]:
        #     st.image("readme/images/logo.png")
        # with col[1]:
            # Chat input
        # with st.expander("💬 Chat", expanded=True):

        if prompt := st.chat_input("Hi! Ask me anything...") or audio_prompt or audio_file_added:
            if not audio_file_added:
                st.session_state.messages.append(
                    {
                        "role": "user", 
                        "content": [{
                            "type": "text",
                            "text": prompt or audio_prompt,
                        }]
                    }
                )
                
                # Display the new messages
                with st.chat_message("user"):
                    st.markdown(prompt)

            else:
                # Display the audio file
                with st.chat_message("user"):
                    st.audio(f"audio_{audio_id}.wav")

            with st.chat_message("assistant"):
                model2key = {
                    # "openai": openai_api_key,
                    "google": google_api_key,
                    # "anthropic": anthropic_api_key,
                }

                model_params = {
                    "model": model,
                    "temperature": temperature,
                    "max_tokens": max_tokens,
                    "top_p": top_p,
                    "frequency_penalty": frequency_penalty,
                    "presence_penalty": presence_penalty,
                }
                inputs = {
                    "text": pdf_url,
                    "image": image_url,
                    "audio": audio_url,
                    "query": prompt,
                }
                st.write_stream(
                    stream_llm_response(
                        inputs = inputs,
                        model_params=model_params, 
                        model_type=model_type, 
                        api_inputs=model2key[model_type]
                    )
                )

            # --- Added Audio Response (optional) ---
            if audio_response:
                response =  client.audio.speech.create(
                    model=tts_model,
                    voice=tts_voice,
                    input=st.session_state.messages[-1]["content"][0]["text"],
                )
                audio_base64 = base64.b64encode(response.content).decode('utf-8')
                audio_html = f"""
                <audio controls autoplay>
                    <source src="data:audio/wav;base64,{audio_base64}" type="audio/mp3">
                </audio>
                """
                st.html(audio_html)



if __name__=="__main__":
    main()