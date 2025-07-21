#!/usr/bin/env python3
"""
F1 Matching System wrapper for FastAPI tabbed interface
Based on 04_04_f1_match_email.py
"""

import os
import asyncio
from typing import List, Dict, Any, Optional
from dotenv import load_dotenv
from langchain_openai import AzureOpenAIEmbeddings, AzureChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.runnables import Runnable
from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings

class F1MatchingSystem:
    """F1 (Foundation One) gene matching system for precision oncology."""
    
    def __init__(self):
        self.llm = None
        self.combined_db = None
        self.f1_db = None
        self.octsu_db = None
        self.rag_chain = None
        self.retriever = None
        self._ready = False
        
        # Load environment variables
        try:
            load_dotenv('.env', override=True)
        except Exception as e:
            print(f"Warning: Unable to load .env file: {e}")
    
    async def initialize(self):
        """Initialize the F1 matching system."""
        try:
            # Initialize the LLM - Azure OpenAI GPT-4o
            self.llm = AzureChatOpenAI(
                openai_api_key=os.environ["OPENAI_API_KEY000"],
                azure_endpoint=os.environ["OPENAI_API_BASE000"],
                openai_api_version=os.environ["OPENAI_API_VERSION"],
                azure_deployment=os.environ["CHAT_DEPLOYMENT_NAME"],
                temperature=0.0
            )
            
            # Load vector databases
            print("Loading OCTSU vector database from: faiss2_index")
            octsu_embeddings = AzureOpenAIEmbeddings(
                openai_api_key=os.environ["OPENAI_API_KEY000"],
                azure_endpoint=os.environ["OPENAI_API_BASE000"],
                openai_api_version=os.environ["OPENAI_API_VERSION"],
                azure_deployment=os.environ["EMBEDDING_DEPLOYMENT_NAME"]
            )
            self.octsu_db = FAISS.load_local("faiss2_index", octsu_embeddings, allow_dangerous_deserialization=True)
            
            print("Loading F1 vector database from: f1_json_index")
            f1_embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
            self.f1_db = FAISS.load_local("f1_json_index", f1_embeddings, allow_dangerous_deserialization=True)
            
            # Create combined database for comprehensive search
            self.combined_db = self.octsu_db
            
            # Create retrievers for each database
            octsu_retriever = self.octsu_db.as_retriever(search_kwargs={"k": 4})
            f1_retriever = self.f1_db.as_retriever(search_kwargs={"k": 4})
            
            # Use the primary database retriever for now
            self.retriever = octsu_retriever
            
            # Create RAG chain
            self._create_rag_chain()
            
            self._ready = True
            print("✅ F1 matching system initialized successfully")
            
        except Exception as e:
            print(f"❌ Error initializing F1 system: {str(e)}")
            self._ready = False
            raise
    
    def _create_comprehensive_retriever(self):
        """Create a comprehensive retriever for dual database search."""
        
        class ComprehensiveRetriever(Runnable):
            def __init__(self, octsu_db, f1_db, k=8):
                self.octsu_db = octsu_db
                self.f1_db = f1_db
                self.k = k
            
            def invoke(self, input, config=None, **kwargs):
                if isinstance(input, str):
                    query = input
                else:
                    query = input.get("input", input.get("question", str(input)))
                return self.get_relevant_documents(query)
            
            def get_relevant_documents(self, query, run_manager=None):
                """Get documents from both databases."""
                try:
                    # Search OCTSU database (clinical trials)
                    octsu_docs = self.octsu_db.similarity_search(query, k=self.k//2)
                    
                    # Search F1 database (genomic reports)
                    f1_docs = self.f1_db.similarity_search(query, k=self.k//2)
                    
                    # Combine and return
                    all_docs = octsu_docs + f1_docs
                    return all_docs[:self.k]
                    
                except Exception as e:
                    print(f"Error in retrieval: {e}")
                    return []
        
        return ComprehensiveRetriever(self.octsu_db, self.f1_db)
    
    def _create_rag_chain(self):
        """Create the RAG chain for F1 matching."""
        
        # Contextualize question prompt
        contextualize_q_system_prompt = """Given a chat history and the latest user question \
        which might reference context in the chat history, formulate a standalone question \
        which can be understood without the chat history. Do NOT answer the question, \
        just reformulate it if needed and otherwise return it as is."""
        
        contextualize_q_prompt = ChatPromptTemplate.from_messages([
            ("system", contextualize_q_system_prompt),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ])
        
        # Create history-aware retriever
        history_aware_retriever = create_history_aware_retriever(
            self.llm, self.retriever, contextualize_q_prompt
        )
        
        # Answer question prompt
        qa_system_prompt = """You are an expert clinical researcher specializing in precision oncology and genomic medicine. \
        You help match Foundation One (F1) genomic test results to appropriate clinical trials based on specific genetic mutations and biomarkers.

        Your expertise includes:
        - Foundation One genomic testing panel and interpretation
        - Clinical trial eligibility criteria and matching
        - BRCA1/BRCA2, HRD (Homologous Recombination Deficiency), and related mutations
        - Precision oncology treatment selection
        - Biomarker-driven clinical trial enrollment

        Context from the knowledge base:
        {context}

        Guidelines for responses:
        1. **Specific Matching**: When asked to match F1 reports to studies, provide specific study identifiers and detailed eligibility criteria
        2. **Mutation Focus**: Highlight relevant mutations found in F1 reports that match clinical trial requirements
        3. **Patient Criteria**: Consider both genomic and clinical criteria for trial eligibility
        4. **Clear Structure**: Organize responses with clear sections for different matches or recommendations
        5. **Professional Tone**: Maintain clinical accuracy while being accessible

        Example response structure for matching queries:
        - **Matched Studies**: List specific study IDs and titles
        - **Relevant Mutations**: Highlight key genomic findings
        - **Eligibility Summary**: Brief overview of matching criteria
        - **Next Steps**: Recommendations for further evaluation

        Focus on F1-specific genomic data and ensure all recommendations are based on the provided knowledge base."""

        qa_prompt = ChatPromptTemplate.from_messages([
            ("system", qa_system_prompt),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ])
        
        # Create question-answer chain
        question_answer_chain = create_stuff_documents_chain(self.llm, qa_prompt)
        
        # Create final RAG chain
        self.rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)
    
    async def process_query(self, query: str, chat_history: List[Dict]) -> str:
        """Process a query using the F1 matching system."""
        try:
            if not self._ready:
                return "F1 system is not ready. Please try again later."
            
            # Convert chat history to LangChain format
            formatted_history = []
            for msg in chat_history:
                if msg["role"] == "user":
                    formatted_history.append(HumanMessage(content=msg["content"]))
                elif msg["role"] == "assistant":
                    formatted_history.append(AIMessage(content=msg["content"]))
            
            # Get response from RAG chain
            response = self.rag_chain.invoke({
                "input": query,
                "chat_history": formatted_history
            })
            
            return response["answer"]
            
        except Exception as e:
            print(f"❌ Error processing F1 query: {str(e)}")
            return f"Sorry, I encountered an error processing your request: {str(e)}"
    
    def is_ready(self) -> bool:
        """Check if the F1 system is ready."""
        return self._ready
