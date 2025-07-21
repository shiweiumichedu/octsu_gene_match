#!/usr/bin/env python3
"""
Tempus Matching System for FastAPI Web Application
"""

import os
import asyncio
from typing import List, Dict, Any, Optional
from langchain_openai import AzureOpenAIEmbeddings, AzureChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.chains import create_history_aware_retriever
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.messages import HumanMessage
from langchain_core.runnables import RunnableLambda

class TempusMatchingSystem:
    def __init__(self):
        self.llm = None
        self.tempus_db = None
        self.octsu_db = None
        self.rag_chain = None
        self.chat_history = []
        self._ready = False
        
    async def initialize(self):
        """Initialize the Tempus matching system"""
        try:
            # Initialize the LLM - Azure OpenAI GPT-4o
            self.llm = AzureChatOpenAI(
                openai_api_key=os.environ["OPENAI_API_KEY000"],
                azure_endpoint=os.environ["OPENAI_API_BASE000"],
                openai_api_version=os.environ["OPENAI_API_VERSION"],
                deployment_name=os.environ["CHAT_DEPLOYMENT_NAME"],
                temperature=0.0
            )
            
            # Create embeddings models - assuming Tempus uses similar embedding as F1
            tempus_embeddings_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")  # 384 dimensions
            octsu_embeddings_model = AzureOpenAIEmbeddings(
                azure_endpoint=os.environ["OPENAI_API_BASE000"],
                openai_api_key=os.environ["OPENAI_API_KEY000"],
                openai_api_version=os.environ["OPENAI_API_VERSION"],
                azure_deployment=os.environ["EMBEDDING_DEPLOYMENT_NAME"]
            )  # 1536 dimensions for text-embedding-3-small
            
            # Get index paths from environment variables
            octsu_index_path = os.getenv('OCTSU_INDEX_PATH')
            tempus_index_path = os.getenv('TEMPUS_INDEX_PATH')
            
            if not octsu_index_path or not tempus_index_path:
                raise ValueError("Missing index paths in environment variables")
            
            # Load both vector databases
            print(f"Loading OCTSU vector database from: {octsu_index_path}")
            self.octsu_db = FAISS.load_local(octsu_index_path, 
                                             octsu_embeddings_model, 
                                             allow_dangerous_deserialization=True)
            
            print(f"Loading Tempus vector database from: {tempus_index_path}")
            self.tempus_db = FAISS.load_local(tempus_index_path, 
                                              tempus_embeddings_model, 
                                              allow_dangerous_deserialization=True)
            
            # Create dual search retriever
            def dual_search(query):
                if isinstance(query, str):
                    query_str = query
                else:
                    query_str = query.get('input', '') if isinstance(query, dict) else str(query)
                
                # Search both databases and combine results  
                octsu_docs = self.octsu_db.similarity_search(query_str, k=2)
                tempus_docs = self.tempus_db.similarity_search(query_str, k=2)
                return octsu_docs + tempus_docs
            
            retriever = RunnableLambda(dual_search)
            
            # System prompt for question reformulation
            system_prompt = """Given the chat history and a recent user question \
generate a new standalone question \
that can be understood without the chat history. Do NOT answer the question, \
just reformulate it if needed or otherwise return it as is."""

            # Create prompt template for history-aware retrieval
            prompt = ChatPromptTemplate.from_messages(
                [
                    ("system", system_prompt),
                    MessagesPlaceholder("chat_history"),
                    ("human", "{input}"),
                ]
            )

            # Create history-aware retriever
            retriever_with_history = create_history_aware_retriever(
                self.llm, retriever, prompt
            )
            
            # System prompt for question-answering tasks (Tempus-specific)
            qa_system_prompt = """You are a precision oncology assistant specializing in matching patients to clinical trials based on genomic findings.

CONTEXT SOURCES:
1. OCTSU Clinical Studies: Open enrollment studies with specific "genes of interest" eligibility criteria
2. Tempus Genomic Reports: Patient test results with identified gene mutations, variants, and biomarkers

CRITICAL INSTRUCTION: ONLY use actual data from the retrieved context. Do NOT fabricate or invent any Tempus IDs, study numbers, or gene information.

MATCHING LOGIC:
When matching reports to studies, look for:
- Exact gene name matches (e.g., KRAS mutation in report matches KRAS in study genes of interest)
- Pathway matches (e.g., EGFR pathway genes, PI3K/AKT pathway genes)
- Biomarker matches (e.g., MSI-H, TMB-H status)
- Related gene families (e.g., RAS pathway: KRAS, NRAS, BRAF, etc.)

COMPREHENSIVE MATCHING INSTRUCTIONS:
For queries asking to "match all reports" or "comprehensive matching":
1. Analyze ALL Tempus reports in the retrieved context for their gene mutations and biomarkers
2. Analyze ALL OCTSU studies in the retrieved context for their gene eligibility criteria
3. Systematically match each report to potentially eligible studies based on gene/biomarker overlaps
4. Focus on actionable mutations like KRAS, EGFR, BRAF, PIK3CA, and biomarkers like MSI-H, TMB-H

RESPONSE FORMAT:
For matching queries, structure your answer as:
1. MATCHES FOUND: List ONLY the actual Tempus IDs and Study Numbers found in the retrieved context
2. GENE DETAILS: Explain which specific genes/mutations/biomarkers match the study criteria using ONLY information from the retrieved documents
3. ELIGIBILITY: Note any additional study requirements beyond genetics

IMPORTANT: If you don't find specific Tempus IDs or study numbers in the retrieved context, state "No specific matches found in the retrieved documents" rather than inventing examples.

For general queries:
- Provide relevant information from either or both sources
- Always cite specific study numbers and Tempus IDs when available in the context
- Be precise about gene names, mutation types, and biomarker status
- Do not make up or assume any data not present in the retrieved context

{context}"""

            # Create prompt template for Q&A
            qa_prompt = ChatPromptTemplate.from_messages(
                [
                    ("system", qa_system_prompt),
                    MessagesPlaceholder("chat_history"),
                    ("human", "{input}"),
                ]
            )

            # Create document processing chain
            question_answer_chain = create_stuff_documents_chain(self.llm, qa_prompt)

            # Create final RAG chain with history awareness
            self.rag_chain = create_retrieval_chain(retriever_with_history, question_answer_chain)
            
            self._ready = True
            print("✅ Tempus matching system initialized successfully")
            
        except Exception as e:
            print(f"❌ Error initializing Tempus matching system: {e}")
            raise
    
    def is_ready(self) -> bool:
        """Check if the system is ready"""
        return self._ready
    
    async def process_query(self, query: str) -> str:
        """Process a user query and return the response"""
        if not self._ready:
            raise RuntimeError("Tempus matching system not initialized")
        
        try:
            # Get response from RAG chain
            ai_response = self.rag_chain.invoke({
                "input": query, 
                "chat_history": self.chat_history
            })
            
            # Update chat history
            self.chat_history.extend([
                HumanMessage(content=query),
                HumanMessage(content=ai_response["answer"])
            ])
            
            # Keep only last 20 messages to prevent context overflow
            if len(self.chat_history) > 20:
                self.chat_history = self.chat_history[-20:]
            
            return ai_response["answer"]
            
        except Exception as e:
            print(f"❌ Error processing Tempus query: {e}")
            return f"Sorry, I encountered an error processing your request: {str(e)}"
    
    def clear_history(self):
        """Clear the chat history"""
        self.chat_history = []
    
    def get_stats(self) -> Dict[str, Any]:
        """Get system statistics"""
        if not self._ready:
            return {"status": "not ready"}
        
        return {
            "status": "ready",
            "tempus_documents": self.tempus_db.index.ntotal if self.tempus_db else 0,
            "octsu_documents": self.octsu_db.index.ntotal if self.octsu_db else 0,
            "chat_history_length": len(self.chat_history)
        }
