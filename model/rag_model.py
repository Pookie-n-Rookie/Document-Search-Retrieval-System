from langchain.document_loaders import PyPDFLoader, TextLoader, Docx2txtLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.llms import HuggingFacePipeline
from langchain.chains import RetrievalQA
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline
import os
import tempfile

class RAGModel:
    def __init__(self, model_name="google/flan-t5-base"):
        """Initialize the RAG model with components."""
        # Initialize the tokenizer and model
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

        # Create the text generation pipeline
        self.pipe = pipeline(
            "text2text-generation",
            model=self.model,
            tokenizer=self.tokenizer,
            max_length=512,
            min_length=30,
            do_sample=True,
            temperature=0.7,
            top_p=0.9
        )

        # Initialize the LLM for LangChain
        self.llm = HuggingFacePipeline(pipeline=self.pipe)

        # Initialize embeddings for vector store
        self.embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-mpnet-base-v2",
            model_kwargs={'device': 'cpu'}
        )

        # Text splitter for breaking documents into chunks
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=50
        )

        self.vector_store = None

    def _save_temp_file(self, file_content, file_extension):
        """Save the uploaded file content to a temporary file."""
        with tempfile.NamedTemporaryFile(delete=False, suffix=f'.{file_extension}') as tmp_file:
            tmp_file.write(file_content)
            return tmp_file.name

    def load_document(self, file_content, file_name):
        """Load and process the document based on its file extension."""
        file_extension = file_name.split('.')[-1].lower()
        temp_file_path = self._save_temp_file(file_content, file_extension)

        try:
            if file_extension == 'pdf':
                loader = PyPDFLoader(temp_file_path)
            elif file_extension == 'txt':
                loader = TextLoader(temp_file_path)
            elif file_extension in ['docx', 'doc']:
                loader = Docx2txtLoader(temp_file_path)
            else:
                raise ValueError(f"Unsupported file format: {file_extension}")

            documents = loader.load()
            texts = self.text_splitter.split_documents(documents)

            if self.vector_store is None:
                self.vector_store = FAISS.from_documents(texts, self.embeddings)
            else:
                self.vector_store.add_documents(texts)

            return True, "Document processed successfully."

        except Exception as e:
            return False, f"Error processing document: {str(e)}"

        finally:
            if os.path.exists(temp_file_path):
                os.unlink(temp_file_path)

    def answer_question(self, question: str, k_documents: int = 3):
        """Answer the given question using the loaded documents."""
        if self.vector_store is None:
            return "Please load some documents first."

        try:
            qa_chain = RetrievalQA.from_chain_type(
                llm=self.llm,
                chain_type="stuff",
                retriever=self.vector_store.as_retriever(search_kwargs={"k": k_documents}),
                return_source_documents=True
            )

            result = qa_chain({"query": question})
            answer = result['result']
            source_docs = result['source_documents']

            sources = []
            for doc in source_docs:
                if 'page' in doc.metadata:
                    sources.append(f"Page {doc.metadata['page']}")
                if 'source' in doc.metadata:
                    sources.append(f"Source: {doc.metadata['source']}")

            return {
                'answer': answer,
                'sources': sources,
                'source_docs': source_docs
            }

        except Exception as e:
            return {
                'answer': f"Error generating answer: {str(e)}",
                'sources': [],
                'source_docs': []
            }

    def clear_documents(self):
        """Clear all loaded documents from the vector store."""
        self.vector_store = None
        return "Document store cleared."

# Example usage
if __name__ == "__main__":
    rag = RAGModel()

    # Example: Load a PDF file
    with open('example.pdf', 'rb') as file:
        success, message = rag.load_document(file.read(), 'example.pdf')
        print(message)

    # Example: Ask a question
    question = "What are the main points discussed in the document?"
    response = rag.answer_question(question)
    print("\nQuestion:", question)
    print("Answer:", response['answer'])
    print("\nSources:", response['sources'])
