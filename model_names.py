def chatgpt_models():
    return [
        "GPT-3",
        "GPT-3.5",
        "GPT-4",
        "GPT-4 Turbo"
    ]

def gemini_models():
    return [
        "Gemini 1",
        "Gemini 1.5",
        "Gemini 1.5 Turbo"
    ]

def hugging_face_models():
    return [
        "BERT",
        "GPT-2",
        "T5",
        "DistilBERT",
        "RoBERTa",
        "Bloom",
        "GPT-NeoX"
    ]

def main():
    print("ChatGPT Models:")
    for model in chatgpt_models():
        print(f"- {model}")

    print("\nGemini Models:")
    for model in gemini_models():
        print(f"- {model}")

    print("\nHugging Face Models:")
    for model in hugging_face_models():
        print(f"- {model}")

if __name__ == "__main__":
    main()