import streamlit as st
import requests

st.title("Gemini Answer Generator")

user_prompt = st.text_input("Enter your prompt:")

if st.button("Generate Answer"):
    if user_prompt.strip() == "":
        st.warning("Please enter a prompt.")
    else:
        with st.spinner("Generating..."):
            try:
                response = requests.post(
                    "http://localhost:8000/generate",
                    json={"prompt": user_prompt}
                )
                if response.status_code == 200:
                    result = response.json()
                    st.success("Answer:")
                    st.write(result["response"])
                else:
                    st.error(f"Error: {response.status_code} - {response.text}")
            except Exception as e:
                st.error(f"Request failed: {str(e)}")