from transformers import pipeline
import openai
import streamlit as st

# Setup page
st.set_page_config(page_title="Transformers", page_icon="🤖", layout="centered")
st.title("Transformers Pipelines")
st.write("Text Generation, Text Summarization, Sentiment Analysis, Question Answering, Translation...")

# Select pipeline
pipeline_type = st.sidebar.selectbox(
    "Select Pipeline",
    [
        "Text Generation",
        "Text Summarization",
        "Sentiment Analysis",
        "Question Answering",
        "Translation"
    ]
)

# Context for Question Answering
context = ""
if pipeline_type == "Question Answering":
    context = st.text_area("Enter the Context Here for Your Question")

# Select model name
model_name = st.sidebar.selectbox(
    "Select Model Name",
    ["gpt2", "gpt-3.5-turbo (OpenAI)"]
)

# OpenAI API Key
api_key = ""
if "OpenAI" in model_name:
    api_key = st.sidebar.text_input("Enter API Key", key="chatbot_api_key", type="password")

# User Question Input
question = st.chat_input("What do you want to ask?")

# Processing the Question
if question:
    generated_text = ""

    # GPT-2 Model
    if model_name == "gpt2":
        try:
            # Text Generation
            if pipeline_type == "Text Generation":
                generator = pipeline('text-generation', model=model_name)
                max_length = st.sidebar.number_input("Max Length", min_value=50, value=100)
                output = generator(question, max_length=max_length, num_return_sequences=1, truncation=True)
                generated_text = output[0]['generated_text']

            # Text Summarization
            elif pipeline_type == "Text Summarization":
                st.write("The initial response may take longer due to the use of GPT-2.")
                summarizer = pipeline('summarization', model='facebook/bart-large-cnn')
                max_length = st.sidebar.number_input("Max Length", min_value=50, value=100)
                summary = summarizer(question, max_length=max_length, min_length=25, do_sample=False)
                generated_text = summary[0]['summary_text']

            # Sentiment Analysis
            elif pipeline_type == "Sentiment Analysis":
                sentiment_analyzer = pipeline('sentiment-analysis')
                sentiment = sentiment_analyzer(question)
                generated_text = sentiment[0]['label']

            # Question Answering
            elif pipeline_type == "Question Answering":
                if context:
                    qa_pipeline = pipeline('question-answering', model='distilbert-base-uncased-distilled-squad')
                    response = qa_pipeline(question=question, context=context)
                    generated_text = response['answer']
                else:
                    st.error("Please enter context.")

            # Translation
            elif pipeline_type == "Translation":
                st.subheader("**Kindly choose OpenAI to proceed.**")

        except Exception as e:
            st.error(f"An error occurred: {str(e)}")

    # OpenAI Model
    elif model_name == "gpt-3.5-turbo (OpenAI)" and api_key:
        try:
            openai.api_key = api_key
            # Text Generation
            if "Text Generation" in pipeline_type:
                openai.api_key = api_key
                max_tokens = st.sidebar.number_input("max_length", min_value=50, value=100)
                response = openai.chat.completions.create(
                    model="gpt-3.5-turbo",
                    messages=[
                        {"role": "system", "content": "you are the helpful assistant"},
                        {"role": "user", "content": question}
                    ],
                    max_tokens=max_tokens,
                    n=1,
                    stop=None,
                    temperature=0.7
                )
                generated_text = response.choices[0].message.content.strip()
            # Text Summarization
            elif pipeline_type == "Text Summarization":
                max_tokens = st.sidebar.number_input("max_tokens", min_value=50, value=100)
                response = openai.chat.completions.create(
                    model="gpt-3.5-turbo",  # Specify the model (e.g., 'gpt-3.5-turbo', 'gpt-4')
                    messages=[
                        {"role": "system", "content": "Summarize the following text"},
                        {"role": "user", "content": question}  # User input text for summarization
                    ],
                    max_tokens=max_tokens,  # Control the length of the summary
                    temperature=0.5,  # Adjust creativity and randomness
                )
                generated_text = response.choices[0].message.content.strip()
            # Sentiment Analysis
            elif pipeline_type == "Sentiment Analysis":
                response = openai.chat.completions.create(
                    model="gpt-3.5-turbo",  # You can use other models like 'gpt-3.5-turbo' as well
                    messages=[
                        {"role": "system",
                         "content": "Classify the sentiment of the following text as Positive, Negative, or Neutral"},
                        {"role": "user", "content": question}  # User input text for summarization
                    ],
                    max_tokens=50,
                    temperature=0.0,  # We set temperature to 0 for more deterministic output
                )
                generated_text = response.choices[0].message.content.strip()
            # Question Answering
            elif pipeline_type == "Question Answering":
                max_tokens = st.sidebar.number_input("max_tokens", min_value=10, value=50)
                response = openai.chat.completions.create(
                    model="gpt-3.5-turbo",  # You can use other models like 'gpt-3.5-turbo' as well
                    messages=[
                        {"role": "system",
                         "content": f"Answer the questions from here {context}"},
                        {"role": "user", "content": question}  # User input text for summarization
                    ],
                    max_tokens=max_tokens,
                    temperature=0.0,  # We set temperature to 0 for more deterministic output
                )
                generated_text = response.choices[0].message.content.strip()

            # Translation
            elif pipeline_type == "Translation":
                max_tokens = st.sidebar.number_input("max_tokens", min_value=10, value=50)
                response = openai.chat.completions.create(
                    model="gpt-3.5-turbo",  # You can use other models like 'gpt-3.5-turbo' as well
                    messages=[
                        {"role": "system",
                         "content": f"translate this {question}"},
                        {"role": "user", "content": question}  # User input text for summarization
                    ],
                    max_tokens=50,
                    temperature=0.0,  # We set temperature to 0 for more deterministic output
                )
                generated_text = response.choices[0].message.content.strip()
        except Exception as e:
            st.error(f"An error occurred: {str(e)}")

    # Display Output
    try:
        st.subheader("✅ Answer:")
        st.write(generated_text)
    except NameError:
        st.error("No output generated. Please check your inputs.")
    except Exception as e:
        st.error(f"Unexpected error: {str(e)}")
