#!/usr/bin/env python3
"""
Enhanced gene matching system with comprehensive retrieval for precision oncology
Using F1 (Foundation One) genomic data instead of Tempus
"""

import os
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from langchain_openai import AzureOpenAIEmbeddings, AzureChatOpenAI
from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS
from langchain import hub
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain.chains import create_history_aware_retriever
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.messages import HumanMessage
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document

def print_output(docs):
    """Print document content and metadata in a formatted way."""
    for doc in docs:
        print(f"Source: {doc.metadata.get('source', 'Unknown')}")
        print(f"Content: {doc.page_content}")
        print("-" * 50)

# Load environment variables from .env file
try:
    load_dotenv('.env', override=True)
except TypeError:
    print('Unable to load .env file.')
    quit()

# Initialize the LLM - Azure OpenAI GPT-4o
llm = AzureChatOpenAI(
    openai_api_key=os.environ["OPENAI_API_KEY000"],
    azure_endpoint=os.environ["OPENAI_API_BASE000"],
    openai_api_version=os.environ["OPENAI_API_VERSION"],
    deployment_name=os.environ["CHAT_DEPLOYMENT_NAME"],
    temperature=0.0
)

# Test the model with no additional knowledge beyond pretraining 
print("=== Testing LLM without RAG ===")
response = llm.invoke("What clinical trials are available for BRCA1 mutations?")
print(response.content)
print("\n" + "="*50 + "\n") 

# Create embeddings models - different models for different databases
from langchain_openai import AzureOpenAIEmbeddings
f1_embeddings_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")  # 384 dimensions
octsu_embeddings_model = AzureOpenAIEmbeddings(
    azure_endpoint=os.environ["OPENAI_API_BASE000"],
    openai_api_key=os.environ["OPENAI_API_KEY000"],
    openai_api_version=os.environ["OPENAI_API_VERSION"],
    azure_deployment=os.environ["EMBEDDING_DEPLOYMENT_NAME"]
)  # 1536 dimensions for text-embedding-3-small

# Get index paths from environment variables
octsu_index_path = os.getenv('OCTSU_INDEX_PATH')
f1_index_path = os.getenv('F1_INDEX_PATH')

if not octsu_index_path:
    print("Error: OCTSU_INDEX_PATH not found in environment variables")
    print("Please add OCTSU_INDEX_PATH to your .env file")
    exit(1)

if not f1_index_path:
    print("Error: F1_INDEX_PATH not found in environment variables")
    print("Please add F1_INDEX_PATH to your .env file")
    exit(1)

# Load both vector databases from disk with their respective embedding models
print(f"Loading OCTSU vector database from: {octsu_index_path}")
octsu_db = FAISS.load_local(octsu_index_path, 
                            octsu_embeddings_model, 
                            allow_dangerous_deserialization=True)

print(f"Loading F1 vector database from: {f1_index_path}")
f1_db = FAISS.load_local(f1_index_path, 
                         f1_embeddings_model, 
                         allow_dangerous_deserialization=True)

# Check if databases have compatible dimensions for merging
try:
    # Merge the two databases into one
    print("Merging vector databases...")
    combined_db = octsu_db
    combined_db.merge_from(f1_db)
    print(f"Combined database contains {combined_db.index.ntotal} total documents")
    use_combined = True
except RuntimeError as e:
    print(f"Cannot merge databases (different dimensions): {e}")
    print("Will use databases separately in comprehensive retrieval")
    combined_db = octsu_db  # Use OCTSU as primary
    use_combined = False

# ===================================================================
# CUSTOM RETRIEVAL FOR COMPREHENSIVE MATCHING
# ===================================================================

def comprehensive_retrieval(query: str, combined_db, f1_db, octsu_db, use_combined=True, k: int = 10):
    """Custom retrieval that ensures both F1 and OCTSU documents are included for comprehensive queries."""
    
    # Check if this is a comprehensive matching query
    comprehensive_keywords = ['match all', 'all reports', 'comprehensive', 'all studies', 'all patients']
    is_comprehensive = any(keyword in query.lower() for keyword in comprehensive_keywords)
    
    if is_comprehensive:
        # For comprehensive queries, ensure we get documents that are likely to have matches
        # Get BRCA1 documents specifically since we know they should match
        brca_f1_docs = f1_db.similarity_search("BRCA1 double hit mutation genomic", k=2)  # Reduced from 3
        other_f1_docs = f1_db.similarity_search("gene mutation variant", k=1)  # Reduced from 2
        
        brca_octsu_docs = octsu_db.similarity_search("BRCA1 BRCA2 study clinical trial", k=2)  # Reduced from 3
        other_octsu_docs = octsu_db.similarity_search("gene therapy clinical", k=1)  # Reduced from 2
        
        # Combine, removing duplicates
        f1_docs = brca_f1_docs + [doc for doc in other_f1_docs 
                                 if doc.metadata.get('f1_id') not in 
                                 [d.metadata.get('f1_id') for d in brca_f1_docs]]
        
        octsu_docs = brca_octsu_docs + [doc for doc in other_octsu_docs 
                                       if doc.metadata.get('study_number') not in 
                                       [d.metadata.get('study_number') for d in brca_octsu_docs]]
        
        # Limit to reasonable number
        all_docs = f1_docs[:k//2] + octsu_docs[:k//2]
        return all_docs
    else:
        # Use regular similarity search for specific queries
        if use_combined:
            return combined_db.similarity_search(query, k=k)
        else:
            # Search both databases separately and combine results
            f1_docs = f1_db.similarity_search(query, k=k//2)
            octsu_docs = octsu_db.similarity_search(query, k=k//2)
            return f1_docs + octsu_docs

# Create a custom retriever class that's compatible with LangChain
class ComprehensiveRetriever:
    def __init__(self, combined_db, f1_db, octsu_db, use_combined=True):
        self.combined_db = combined_db
        self.f1_db = f1_db
        self.octsu_db = octsu_db
        self.use_combined = use_combined
    
    def invoke(self, query):
        if isinstance(query, str):
            return comprehensive_retrieval(query, self.combined_db, self.f1_db, self.octsu_db, self.use_combined, k=8)
        else:
            # For dict input (from LangChain chain)
            query_str = query.get('input', '') if isinstance(query, dict) else str(query)
            return comprehensive_retrieval(query_str, self.combined_db, self.f1_db, self.octsu_db, self.use_combined, k=8)
    
    def get_relevant_documents(self, query):
        return self.invoke(query)
    
    def __or__(self, other):
        # Make it compatible with LangChain's | operator
        from langchain_core.runnables import RunnableLambda
        return RunnableLambda(self.invoke) | other
    
    def __ror__(self, other):
        # Make it compatible with LangChain's | operator (reverse)
        from langchain_core.runnables import RunnableLambda
        return other | RunnableLambda(self.invoke)

# Configure retriever for combined database - use default retriever for now
if use_combined:
    retriever = combined_db.as_retriever(search_type="similarity", search_kwargs={"k": 4})  # Reduced from 8
else:
    # Create a simple retriever that searches both databases
    from langchain_core.runnables import RunnableLambda
    
    def dual_search(query):
        if isinstance(query, str):
            query_str = query
        else:
            query_str = query.get('input', '') if isinstance(query, dict) else str(query)
        
        # Search both databases and combine results  
        octsu_docs = octsu_db.similarity_search(query_str, k=2)  # Reduced from 4 to 2
        f1_docs = f1_db.similarity_search(query_str, k=2)  # Reduced from 4 to 2
        return octsu_docs + f1_docs
    
    retriever = RunnableLambda(dual_search)

# We'll implement comprehensive matching in the QA prompt instead

# ===================================================================
# CONVERSATION HISTORY SETUP
# ===================================================================

# Preserve conversation history - system prompt for question reformulation
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
    llm, retriever, prompt
)

# ===================================================================
# QUESTION-ANSWERING CHAIN SETUP
# ===================================================================

# System prompt for question-answering tasks
qa_system_prompt = """You are a precision oncology assistant specializing in matching patients to clinical trials based on genomic findings.

CONTEXT SOURCES:
1. OCTSU Clinical Studies: Open enrollment studies with specific "genes of interest" eligibility criteria
2. F1 (Foundation One) Genomic Reports: Patient test results with identified gene mutations and variants

CRITICAL INSTRUCTION: ONLY use actual data from the retrieved context. Do NOT fabricate or invent any F1 IDs, study numbers, or gene information.

MATCHING LOGIC:
When matching reports to studies, look for:
- Exact gene name matches (e.g., BRCA1 mutation in report matches BRCA1 in study genes of interest)
- Pathway matches (e.g., HRD-related genes, RAS pathway genes)
- Related gene families (e.g., DNA repair genes: BRCA1/2, PALB2, ATM, etc.)

COMPREHENSIVE MATCHING INSTRUCTIONS:
For queries asking to "match all reports" or "comprehensive matching":
1. Analyze ALL F1 reports in the retrieved context for their gene mutations
2. Analyze ALL OCTSU studies in the retrieved context for their gene eligibility criteria
3. Systematically match each report to potentially eligible studies based on gene overlaps
4. Focus especially on BRCA1/BRCA2 matches as these are common actionable mutations

RESPONSE FORMAT:
For matching queries, structure your answer as:
1. MATCHES FOUND: List ONLY the actual F1 IDs and Study Numbers found in the retrieved context
2. GENE DETAILS: Explain which specific genes/mutations match the study criteria using ONLY information from the retrieved documents
3. ELIGIBILITY: Note any additional study requirements beyond genetics

IMPORTANT: If you don't find specific F1 IDs or study numbers in the retrieved context, state "No specific matches found in the retrieved documents" rather than inventing examples.

For general queries:
- Provide relevant information from either or both sources
- Always cite specific study numbers and F1 IDs when available in the context
- Be precise about gene names and mutation types
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
question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)

# Create final RAG chain with history awareness
rag_chain = create_retrieval_chain(retriever_with_history, question_answer_chain)

# ===================================================================
# INTERACTIVE CONVERSATIONAL RAG APPLICATION
# ===================================================================

def interactive_rag_chat():
    """Interactive chat function for continuous Q&A with the combined RAG system."""
    print("="*60)
    print("🧬 PRECISION ONCOLOGY GENE MATCHING SYSTEM (F1)")
    print("="*60)
    print("Match Foundation One genomic test results to clinical trial eligibility!")
    print("\nExample queries:")
    print("  🎯 'Match F1 reports to BRCA studies'")
    print("  🔍 'Find studies for patients with HRD mutations'")
    print("  📊 'Which F1 reports have BRCA1 mutations?'")
    print("  🔄 'Match all reports to available studies'")
    print("  📋 'Show study 2021.070 gene requirements'")
    print("  🧪 'What mutations are in F1 report?'")
    print("\nCommands:")
    print("  Type 'quit', 'exit', or 'q' to end")
    print("  Type 'history' to see conversation history")
    print("  Type 'clear' to clear conversation history")
    print("  Type 'test-email' to test email functionality")
    print("-"*60)
    
    # Initialize conversation history
    chat_history = []
    question_count = 0
    stored_email = None  # Store email address for reuse
    
    while True:
        try:
            # Get user input
            user_question = input("\n💬 You: ").strip()
            
            # Handle special commands
            if user_question.lower() in ['quit', 'exit', 'q']:
                print("\n👋 Thank you for using the Gene Matching System! Goodbye!")
                break
                
            elif user_question.lower() == 'history':
                print("\n📜 Conversation History:")
                if not chat_history:
                    print("   No conversation history yet.")
                else:
                    for i in range(0, len(chat_history), 2):
                        human_msg = chat_history[i].content if i < len(chat_history) else ""
                        ai_msg = chat_history[i+1].content if i+1 < len(chat_history) else ""
                        print(f"   Q{(i//2)+1}: {human_msg}")
                        print(f"   A{(i//2)+1}: {ai_msg}")
                        print()
                continue
                
            elif user_question.lower() == 'clear':
                chat_history = []
                question_count = 0
                print("\n🧹 Conversation history cleared!")
                continue
                
            elif user_question.lower() == 'test-email':
                test_email = input("📧 Enter email address to test: ").strip()
                if test_email and '@' in test_email:
                    print("📤 Sending test email...")
                    if send_gene_match_email(test_email, "Email Test", "This is a test email from the F1 Gene Matching System."):
                        print(f"✅ Test email sent successfully to {test_email}")
                    else:
                        print("❌ Failed to send test email")
                else:
                    print("❌ Invalid email address")
                continue
                
            elif not user_question:
                print("   Please enter a question or type 'quit' to exit.")
                continue
            
            # Process the question with RAG
            question_count += 1
            print(f"\n🔍 Processing question {question_count}...")
            
            # Get response from RAG chain
            ai_response = rag_chain.invoke({
                "input": user_question, 
                "chat_history": chat_history
            })
            
            # Update chat history
            chat_history.extend([
                HumanMessage(content=user_question),
                HumanMessage(content=ai_response["answer"])
            ])
            
            # Display the response
            print(f"\n🤖 AI: {ai_response['answer']}")
            
            # Prompt to send email with results
            stored_email = prompt_for_email_sending(user_question, ai_response["answer"], stored_email)
            
        except KeyboardInterrupt:
            print("\n\n👋 Goodbye!")
            break
        except Exception as e:
            print(f"\n❌ Error: {e}")
            print("Please try again or type 'quit' to exit.")

# ===================================================================
# EMAIL FUNCTIONALITY
# ===================================================================

def send_gene_match_email(recipient_email: str, question: str, answer: str) -> bool:
    """Send gene matching results via email."""
    try:
        # Email configuration from .env
        mail_host = os.environ.get("MAIL_HOST")
        mail_port = int(os.environ.get("MAIL_PORT", 25))
        mail_from = os.environ.get("MAIL_FROM")
        
        if not all([mail_host, mail_from]):
            print("❌ Email configuration missing in .env file")
            return False
        
        # Create message
        msg = MIMEMultipart()
        msg['From'] = mail_from
        msg['To'] = recipient_email
        msg['Subject'] = "F1 Gene Matching Results - Precision Oncology System"
        
        # Email body
        body = f"""
F1 (Foundation One) Gene Matching Results
=========================================

Query: {question}

Results:
{answer}

---
This email was generated by the Precision Oncology Gene Matching System (F1 Edition).
For questions, please contact: {mail_from}
"""
        
        msg.attach(MIMEText(body, 'plain'))
        
        # Send email
        server = smtplib.SMTP(mail_host, mail_port)
        text = msg.as_string()
        server.sendmail(mail_from, recipient_email, text)
        server.quit()
        
        return True
        
    except Exception as e:
        print(f"❌ Failed to send email: {e}")
        return False

def prompt_for_email_sending(question: str, answer: str, stored_email: str = None) -> str:
    """Prompt user for email sending and return the email used (if any)."""
    
    # Ask if user wants to send email
    send_email = input("\n📧 Send results via email? (y/N): ").strip().lower()
    
    if send_email != 'y':
        return stored_email  # Return the stored email unchanged
    
    # Get email address
    if stored_email:
        use_stored = input(f"📧 Send to {stored_email}? (Y/n): ").strip().lower()
        if use_stored != 'n':
            target_email = stored_email
        else:
            target_email = input("📧 Enter email address: ").strip()
    else:
        target_email = input("📧 Enter email address: ").strip()
    
    if not target_email or '@' not in target_email:
        print("❌ Invalid email address")
        return stored_email
    
    # Send the email
    print("📤 Sending email...")
    if send_gene_match_email(target_email, question, answer):
        print(f"✅ Email sent successfully to {target_email}")
        return target_email
    else:
        print("❌ Failed to send email")
        return stored_email

# ===================================================================
# MAIN EXECUTION
# ===================================================================

if __name__ == "__main__":
    # Test with some predefined questions
    print("=== Testing F1 RAG System ===")
    
    # Initialize conversation history for testing
    chat_history = []
    
    # Test question 1: General BRCA query
    question = "What clinical trials are available for BRCA1 mutations?"
    print(f"\nQuestion: {question}")
    ai_msg_1 = rag_chain.invoke({"input": question, "chat_history": chat_history})
    print(f"Answer: {ai_msg_1['answer']}")
    
    # Update history and ask follow-up
    chat_history.extend([HumanMessage(content=question), HumanMessage(content=ai_msg_1["answer"])])
    
    # Test question 2: Comprehensive matching
    second_question = "Match all F1 reports to available studies"
    print(f"\n\nQuestion: {second_question}")
    ai_msg_2 = rag_chain.invoke({"input": second_question, "chat_history": chat_history})
    print(f"Answer: {ai_msg_2['answer']}")
    
    # Update history
    chat_history.extend([HumanMessage(content=second_question), HumanMessage(content=ai_msg_2["answer"])])
    
    # Test question 3: Specific F1 query
    third_question = "What studies match patients with F1 reports containing BRCA mutations?"
    print(f"\n\nQuestion: {third_question}")
    ai_msg_3 = rag_chain.invoke({"input": third_question, "chat_history": chat_history})
    print(f"Answer: {ai_msg_3['answer']}")
    
    print("\n" + "="*60)
    print("✅ F1 RAG System Test Complete!")
    print("="*60)
    
    # Start interactive chat
    interactive_rag_chat()
