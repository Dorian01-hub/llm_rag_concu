# import pandas as pd
# from langchain.schema import Document
# from langchain.embeddings import HuggingFaceEmbeddings
# from langchain.vectorstores import Chroma
# from tqdm import tqdm



# # Chemin vers votre fichier Feather
# chemin_fichier = 'base_rag_chunks_final.feather'

# # Lire le fichier Feather
# df = pd.read_feather(chemin_fichier)


# # Afficher les premières lignes du DataFrame
# print(df.head())

# documents = []

# for row in df.itertuples(index=False):
#     text = getattr(row, "chunks", "")
#     metadata = {
#         "titre": getattr(row, "titre", ""),
#         "date": getattr(row, "date", ""),
#         "lien_pdf": getattr(row, "url_pdf", ""),
#         "lien_page": getattr(row, "url_page", ""),
#         "source_type": getattr(row, "source_type", ""),
#         "chunk_id": getattr(row, "chunk_id", "")
#     }
#     documents.append(Document(page_content=text, metadata=metadata))


# # Initialiser les embeddings
# embeddings = HuggingFaceEmbeddings(
#     model_name="sentence-transformers/all-MiniLM-L6-v2",
#     encode_kwargs={"batch_size": 100, "show_progress_bar": True}
# )

# # Répertoire de la base
# persist_directory = "chromadb"

# # D'abord, créer la base avec le premier batch
# first_batch = documents[:10000]
# vectorstore = Chroma.from_documents(first_batch, embedding=embeddings, persist_directory=persist_directory)

# # Ajouter les autres batches
# for i in range(10000, len(documents), 10000):
#     batch = documents[i:i+10000]
#     vectorstore.add_documents(batch)
#     print(f"Batch {i//10000 + 1} ajouté.")

# # Sauvegarder la base vectorielle complète
# vectorstore.persist()
# print("Base vectorielle complète sauvegardée.")



## code pour ajouter des documents à une base existante
# vectorstore = Chroma(persist_directory="chromadb", embedding_function=embeddings)
# vectorstore.add_documents(nouveaux_documents)
# vectorstore.persist()