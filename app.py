import streamlit as st
from transformers import AutoTokenizer, AutoModel
from sub1 import sub1_forward, sub1_backward

# Lista dostępnych modeli
MODELS = {
    "base-128": "/app/models/base-128",
    "large-128": "/app/models/large-128",
    "c-2048": "/app/models/c-2048"
}

st.title("QMoE Model Inference")

# Wybór modelu
selected_model = st.selectbox("Wybierz model:", list(MODELS.keys()))

# Wczytaj wybrany model i tokenizer
tokenizer = AutoTokenizer.from_pretrained(MODELS[selected_model])
model = AutoModel.from_pretrained(MODELS[selected_model])

# Umożliw użytkownikowi wprowadzenie tekstu do inferencji
input_text = st.text_area("Wprowadź tekst:")

# Przycisk wysyłania
send_button = st.button("Wyślij")

if send_button and input_text:
    # Tokenizuj tekst
    inputs = tokenizer(input_text, return_tensors="pt", truncation=True, padding=True, max_length=512)
    
    # Przeprowadź inferencję
    with st.spinner("Przetwarzanie..."):
        outputs = model(**inputs)
        # Możesz dodać dodatkowy kod do przetwarzania wyników, jeśli to konieczne
    
    # Wyświetl wysłaną wiadomość i odpowiedź
    st.write("Wysłana wiadomość:", input_text)
    st.write("Odpowiedź modelu:", outputs)

# Możesz dodać dodatkowe funkcje, takie jak wyświetlanie wizualizacji itp.
