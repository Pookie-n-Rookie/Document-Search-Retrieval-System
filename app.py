import streamlit as st
from model.rag_model import RAGModel


def main():
    st.set_page_config(page_title="Document Q&A System", layout="wide")

    # Initialize RAG model
    if 'rag_model' not in st.session_state:
        st.session_state.rag_model = RAGModel()

    st.title("📚 Document Q&A System")

    # Sidebar for file upload
    with st.sidebar:
        st.header("Document Management")
        uploaded_file = st.file_uploader(
            "Upload Document",
            type=['pdf', 'txt', 'docx'],
            help="Upload a document to analyze"
        )

        if uploaded_file:
            if st.button("Process Document"):
                with st.spinner("Processing document..."):
                    success, message = st.session_state.rag_model.load_document(
                        uploaded_file.read(),
                        uploaded_file.name
                    )
                    if success:
                        st.success(message)
                    else:
                        st.error(message)

        if st.button("Clear All Documents"):
            message = st.session_state.rag_model.clear_documents()
            st.info(message)

    # Main content area
    st.header("Ask Questions")
    question = st.text_input("Enter your question about the document(s)")

    if question:
        with st.spinner("Generating answer..."):
            response = st.session_state.rag_model.answer_question(question)

            st.markdown("### Answer")
            st.write(response['answer'])

            if response['sources']:
                st.markdown("### Sources")
                for source in response['sources']:
                    st.write(f"- {source}")


if __name__ == "__main__":
    main()