
from langchain.vectorstores import Chroma
from langchain.embeddings import HuggingFaceEmbeddings
from mistralai.client import MistralClient
from mistralai import Mistral
from langchain_mistralai import ChatMistralAI
from langchain.prompts import PromptTemplate
from dataclasses import dataclass

from langchain.chat_models.base import SimpleChatModel
import os
from langchain.chains import RetrievalQA
import streamlit as st


@dataclass
class ChatMessage:
    role: str
    content: str



# 🔒 Mise en cache du modèle d'embedding
@st.cache_resource
def get_embedding_model():
    return HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")

# 🔒 Mise en cache de la base vectorielle Chroma
@st.cache_resource
def get_vectordb():
    embedding_model = get_embedding_model()
    vectordb = Chroma(persist_directory="chromadb", embedding_function=embedding_model)
    return vectordb

# ✨ Appel des fonctions dans ton app
embedding_model = get_embedding_model()
retriever = get_vectordb().as_retriever(search_kwargs={"k": 3})



os.environ["MISTRAL_API_KEY"] = "1ROdIMBXIGXrgxnD3cywOg4EhRaJufJA"

client = Mistral(api_key="1ROdIMBXIGXrgxnD3cywOg4EhRaJufJA")

mistral_model = "mistral-large-latest" # "open-mixtral-8x22b" 
llm = ChatMistralAI(model=mistral_model, temperature=0.2)


# 1. Ton prompt personnalisé
custom_prompt = PromptTemplate(
    input_variables=["context", "question"],
    template="""
Vous êtes un assistant juridique spécialisé en droit de la concurrence.
Utilisez les extraits de décisions suivants pour répondre précisément à la question posée.
Si la réponse ne figure pas dans les documents, indiquez-le clairement.

Contextes juridiques :
{context}

Question :
{question}

Réponse juridique argumentée :
"""
)

# 2. Création de la chaîne avec le prompt
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=retriever,
    return_source_documents=True,
    chain_type_kwargs={"prompt": custom_prompt}
)


# --- Interface utilisateur ---
st.title("Assistant juridique - Droit de la concurrence")
query = st.text_input("Posez votre question juridique :", value="")

if query:
    with st.spinner("Recherche en cours..."):
        result = qa_chain.invoke(query)  # 👈 La question est injectée ici

    # --- Affichage réponse ---
    st.subheader("Réponse générée :")
    st.write(result['result'])

    # --- Affichage des sources ---
    st.subheader("Sources utilisées :")
    for doc in qa_chain.invoke(query)['source_documents']:
        lien = doc.metadata.get("lien_pdf", "Lien inconnu")
        st.markdown(f"- **Lien** : {lien}")
        st.markdown(f"```{doc.page_content[:500]}...```")