import os

# Here should a openai api key be provided


from llama_index.core import VectorStoreIndex, SimpleDirectoryReader



documents1 = SimpleDirectoryReader(input_files=["your path to/DiseasePrecaution.txt"]).load_data()
documents2 = SimpleDirectoryReader(input_files=["your path to/SymptomSeverity.txt"]).load_data()
documents3 = SimpleDirectoryReader(input_files=["your path to/disease.txt"]).load_data()
documents4 = SimpleDirectoryReader(input_files=["your path to/drug_info.txt"]).load_data()



index1 = VectorStoreIndex.from_documents(documents1)
index2 = VectorStoreIndex.from_documents(documents2)
index3 = VectorStoreIndex.from_documents(documents3)
index4 = VectorStoreIndex.from_documents(documents4)


# Persist the index for future use
index1.storage_context.persist()
index2.storage_context.persist()
index3.storage_context.persist()
index4.storage_context.persist()

