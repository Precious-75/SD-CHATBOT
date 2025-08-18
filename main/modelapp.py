import streamlit as st
import csv
import random
import re
import json
import os
import pandas as pd
from typing import Dict, List, Tuple
from io import StringIO

class SentimentAnalyzer:
    def __init__(self):
        # Simple rule-based sentiment analysis
        self.positive_words = {
            'good', 'great', 'excellent', 'awesome', 'amazing', 'fantastic', 'wonderful',
            'perfect', 'love', 'like', 'happy', 'satisfied', 'pleased', 'thank', 'thanks',
            'helpful', 'useful', 'appreciate', 'brilliant', 'outstanding', 'superb'
        }
        
        self.negative_words = {
            'bad', 'terrible', 'awful', 'horrible', 'hate', 'angry', 'frustrated',
            'annoyed', 'upset', 'disappointed', 'useless', 'stupid', 'dumb', 'broken',
            'problem', 'issue', 'error', 'fail', 'crash', 'stuck', 'help', 'urgent',
            'emergency', 'critical', 'serious', 'wrong', 'not working', 'broken'
        }
        
        self.urgency_words = {
            'urgent', 'emergency', 'asap', 'immediately', 'critical', 'serious',
            'important', 'deadline', 'quick', 'fast', 'hurry', 'rush'
        }
        
        self.confusion_words = {
            'confused', 'lost', 'stuck', 'dont understand', "don't understand",
            'how do i', 'what is', 'explain', 'help me understand', 'unclear'
        }
    
    def analyze_sentiment(self, text: str) -> Dict[str, any]:
        """Analyze sentiment and emotional state of text"""
        text_lower = text.lower()
        words = re.findall(r'\b\w+\b', text_lower)
        
        positive_count = sum(1 for word in words if word in self.positive_words)
        negative_count = sum(1 for word in words if word in self.negative_words)
        urgency_count = sum(1 for word in words if word in self.urgency_words)
        confusion_count = sum(1 for phrase in self.confusion_words if phrase in text_lower)
        
        # Determine overall sentiment
        if positive_count > negative_count:
            sentiment = "positive"
        elif negative_count > positive_count:
            sentiment = "negative"
        else:
            sentiment = "neutral"
        
        # Determine emotional states
        is_urgent = urgency_count > 0
        is_confused = confusion_count > 0
        is_frustrated = negative_count >= 2 or any(word in text_lower for word in ['frustrated', 'angry', 'annoyed'])
        
        # Calculate intensity (0.0 to 1.0)
        total_emotional_words = positive_count + negative_count + urgency_count
        intensity = min(total_emotional_words / max(len(words), 1), 1.0)
        
        return {
            'sentiment': sentiment,
            'is_urgent': is_urgent,
            'is_confused': is_confused,
            'is_frustrated': is_frustrated,
            'intensity': intensity,
            'positive_score': positive_count,
            'negative_score': negative_count
        }

class StreamlitChatbot:
    def __init__(self):
        if 'knowledge_base' not in st.session_state:
            st.session_state.knowledge_base = {}
        
        if 'sentiment_analyzer' not in st.session_state:
            st.session_state.sentiment_analyzer = SentimentAnalyzer()
        
        if 'chat_history' not in st.session_state:
            st.session_state.chat_history = []
        
        self.responses = {
            'greeting': [
                "Hello! How can I help you with IT support today?",
                "Hi there! What IT issue can I help you with?",
                "Hey! What technical problem are you facing?"
            ],
            'goodbye': [
                "Goodbye! Have a great day!",
                "See you later! Hope I helped!",
                "Take care! Don't hesitate to ask if you need more IT help!"
            ],
            'default': [
                "I'm not sure about that specific issue. Can you rephrase your question?",
                "Could you provide more details about the problem you're experiencing?",
                "I don't have information about that in my IT knowledge base. Can you describe it differently?",
                "That's an interesting question! Can you ask it in a different way?"
            ],
            'frustrated': [
                "I understand this can be frustrating. Let me try to help you with this issue.",
                "I can see you're having a tough time with this. Let's work through it together.",
                "Technical problems can be really annoying. I'm here to help you solve this."
            ],
            'urgent': [
                "I understand this is urgent. Let me find the best solution for you quickly.",
                "This seems like a priority issue. I'll do my best to help you right away.",
                "I can see this needs immediate attention. Let me help you resolve this."
            ],
            'confused': [
                "No worries! Let me explain this step by step.",
                "I understand this can be confusing. Let me break it down for you.",
                "That's okay! Let me help clarify this for you."
            ],
            'positive': [
                "I'm glad you're having a good experience! How can I help you today?",
                "Great to hear! What can I assist you with?",
                "Wonderful! What IT question do you have for me?"
            ]
        }
        
        self.patterns = {
            'greeting': [r'hello', r'hi', r'hey', r'good morning', r'good afternoon'],
            'goodbye': [r'bye', r'goodbye', r'see you', r'farewell', r'thanks', r'thank you']
        }
    
    def load_csv_from_upload(self, uploaded_file):
        """Load Q&A pairs from uploaded CSV file"""
        try:
            # Read the uploaded file
            content = uploaded_file.read()
            if isinstance(content, bytes):
                content = content.decode('utf-8')
            
            # Parse CSV
            csv_reader = csv.DictReader(StringIO(content))
            fieldnames = csv_reader.fieldnames
            
            # Find question and answer columns
            question_col = None
            answer_col = None
            
            for col in fieldnames:
                col_lower = col.lower().strip()
                if col_lower in ['question', 'questions', 'q', 'query']:
                    question_col = col
                elif col_lower in ['answer', 'answers', 'a', 'response', 'solution']:
                    answer_col = col
            
            if not question_col or not answer_col:
                st.error(f"Could not find Question and Answer columns. Available columns: {fieldnames}")
                st.info("Please make sure your CSV has columns named 'Question' and 'Answer' (or similar variations)")
                return False
            
            # Clear existing knowledge base
            st.session_state.knowledge_base.clear()
            
            # Load the data
            count = 0
            for row in csv_reader:
                question = row.get(question_col, '').strip()
                answer = row.get(answer_col, '').strip()
                
                if question and answer:  # Skip empty rows
                    self.add_to_knowledge_base(question, answer)
                    count += 1
            
            st.success(f"Successfully loaded {count} Q&A pairs!")
            return count > 0
                
        except Exception as e:
            st.error(f"Error reading CSV file: {e}")
            return False
    
    def add_to_knowledge_base(self, question: str, answer: str):
        """Add a Q&A pair to the knowledge base"""
        # Normalize question for matching
        normalized_question = self.normalize_text(question)
        keywords = self.extract_keywords(question)
        
        st.session_state.knowledge_base[normalized_question] = {
            'original_question': question,
            'answer': answer,
            'keywords': keywords
        }
    
    def normalize_text(self, text: str) -> str:
        """Normalize text for better matching"""
        return re.sub(r'[^\w\s]', '', text.lower().strip())
    
    def extract_keywords(self, text: str) -> List[str]:
        """Extract keywords from text"""
        stop_words = {
            'the', 'is', 'are', 'was', 'were', 'a', 'an', 'and', 'or', 'but', 
            'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'from', 'up', 
            'about', 'into', 'through', 'during', 'before', 'after', 'above', 
            'below', 'between', 'among', 'until', 'while', 'how', 'what', 
            'where', 'when', 'why', 'can', 'could', 'should', 'would', 'will',
            'do', 'does', 'did', 'have', 'has', 'had', 'be', 'been', 'being',
            'my', 'your', 'his', 'her', 'its', 'our', 'their', 'me', 'you',
            'him', 'her', 'us', 'them', 'this', 'that', 'these', 'those'
        }
        
        words = self.normalize_text(text).split()
        return [word for word in words if word not in stop_words and len(word) > 2]
    
    def find_best_match(self, user_question: str) -> Tuple[str, float]:
        """Find the best matching answer for user question"""
        if not st.session_state.knowledge_base:
            return None, 0
        
        user_normalized = self.normalize_text(user_question)
        user_keywords = set(self.extract_keywords(user_question))
        
        best_match = None
        best_score = 0
        
        for stored_question, data in st.session_state.knowledge_base.items():
            # Calculate text similarity
            similarity_score = self.calculate_text_similarity(user_normalized, stored_question)
            
            # Calculate keyword similarity
            stored_keywords = set(data['keywords'])
            if user_keywords and stored_keywords:
                keyword_overlap = len(user_keywords.intersection(stored_keywords))
                max_keywords = max(len(user_keywords), len(stored_keywords))
                keyword_score = keyword_overlap / max_keywords if max_keywords > 0 else 0
            else:
                keyword_score = 0
            
            # Combined score (text similarity weighted more heavily)
            total_score = (similarity_score * 0.7) + (keyword_score * 0.3)
            
            if total_score > best_score:
                best_score = total_score
                best_match = data['answer']
        
        return best_match, best_score
    
    def calculate_text_similarity(self, text1: str, text2: str) -> float:
        """Calculate similarity between two texts using Jaccard similarity"""
        words1 = set(text1.split())
        words2 = set(text2.split())
        
        if not words1 and not words2:
            return 1.0
        if not words1 or not words2:
            return 0.0
            
        intersection = words1.intersection(words2)
        union = words1.union(words2)
        
        return len(intersection) / len(union) if union else 0
    
    def detect_intent(self, message: str) -> str:
        """Detect user intent"""
        message_lower = message.lower()
        for intent, patterns in self.patterns.items():
            for pattern in patterns:
                if re.search(pattern, message_lower):
                    return intent
        return 'question'
    
    def get_sentiment_response_prefix(self, sentiment_data: Dict) -> str:
        """Get appropriate response prefix based on sentiment"""
        if sentiment_data['is_frustrated']:
            return random.choice(self.responses['frustrated'])
        elif sentiment_data['is_urgent']:
            return random.choice(self.responses['urgent'])
        elif sentiment_data['is_confused']:
            return random.choice(self.responses['confused'])
        elif sentiment_data['sentiment'] == 'positive':
            return random.choice(self.responses['positive'])
        return ""
    
    def get_response(self, user_input: str) -> str:
        """Generate response to user input with sentiment awareness"""
        if not user_input.strip():
            return "Please ask me something!"
        
        # Analyze sentiment first
        sentiment_data = st.session_state.sentiment_analyzer.analyze_sentiment(user_input)
        
        intent = self.detect_intent(user_input)
        
        # Handle greetings and goodbyes
        if intent in self.responses and intent != 'question':
            base_response = random.choice(self.responses[intent])
            
            # Add sentiment-aware touch to greetings
            if intent == 'greeting' and sentiment_data['sentiment'] == 'negative':
                return "Hello! I can see you might be having some issues. How can I help you with IT support today?"
            elif intent == 'greeting' and sentiment_data['is_urgent']:
                return "Hi there! I understand you need urgent help. What IT issue can I assist you with right away?"
            
            return base_response
        
        # Search for answer in knowledge base
        answer, confidence = self.find_best_match(user_input)
        
        # Get sentiment-aware prefix
        sentiment_prefix = self.get_sentiment_response_prefix(sentiment_data)
        
        if answer and confidence > 0.15:
            # Combine sentiment response with actual answer
            if sentiment_prefix:
                return f"{sentiment_prefix}\n\n{answer}"
            return answer
        else:
            # Handle no-match cases with sentiment awareness
            if sentiment_data['is_frustrated']:
                return "I understand your frustration. I don't have a direct answer for that, but could you rephrase your question? I want to make sure I help you properly."
            elif sentiment_data['is_urgent']:
                return "I want to help you with this urgent issue, but I need more details. Could you describe the problem differently so I can find the best solution?"
            elif sentiment_data['is_confused']:
                return "No problem! I don't have that exact information, but let's try a different approach. Can you tell me more about what you're trying to do?"
            else:
                return random.choice(self.responses['default'])

def main():
    st.set_page_config(
        page_title="IT Support Chatbot",
        page_icon="🤖",
        layout="wide"
    )
    
    # Initialize chatbot
    if 'chatbot' not in st.session_state:
        st.session_state.chatbot = StreamlitChatbot()
    
    # Title and description
    st.title("🤖 IT Support Chatbot with Sentiment Analysis")
    st.markdown("Upload a CSV file with IT questions and answers, then chat with the AI assistant!")
    
    # Sidebar for file upload and stats
    with st.sidebar:
        st.header("📊 Configuration")
        
        # File upload
        uploaded_file = st.file_uploader(
            "Upload CSV File",
            type=['csv'],
            help="Upload a CSV file with 'Question' and 'Answer' columns"
        )
        
        if uploaded_file is not None:
            if st.button("Load CSV File"):
                with st.spinner("Loading CSV file..."):
                    if st.session_state.chatbot.load_csv_from_upload(uploaded_file):
                        st.rerun()
        
        # Knowledge base stats
        st.header("📈 Statistics")
        kb_size = len(st.session_state.knowledge_base)
        st.metric("Q&A Pairs Loaded", kb_size)
        
        if kb_size > 0:
            st.subheader("Sample Questions")
            sample_questions = list(st.session_state.knowledge_base.values())[:5]
            for i, data in enumerate(sample_questions, 1):
                question = data['original_question']
                if len(question) > 60:
                    question = question[:57] + "..."
                st.write(f"{i}. {question}")
            
            if kb_size > 5:
                st.write(f"... and {kb_size - 5} more questions")
        
        # Clear chat history button
        if st.button("🗑️ Clear Chat History"):
            st.session_state.chat_history = []
            st.rerun()
        
        # Sentiment analysis toggle
        st.header("🎭 Sentiment Analysis")
        show_sentiment = st.checkbox("Show sentiment analysis", value=False)
    
    # Main chat interface
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.header("💬 Chat")
        
        # Display chat history
        chat_container = st.container()
        with chat_container:
            for message in st.session_state.chat_history:
                with st.chat_message(message["role"]):
                    st.write(message["content"])
                    
                    # Show sentiment analysis if enabled
                    if message["role"] == "user" and show_sentiment and "sentiment" in message:
                        sentiment = message["sentiment"]
                        
                        # Create sentiment display
                        sentiment_cols = st.columns(4)
                        with sentiment_cols[0]:
                            color = "green" if sentiment['sentiment'] == "positive" else "red" if sentiment['sentiment'] == "negative" else "gray"
                            st.markdown(f"**Sentiment:** :{color}[{sentiment['sentiment'].title()}]")
                        
                        with sentiment_cols[1]:
                            if sentiment['is_urgent']:
                                st.markdown("🚨 **Urgent**")
                        
                        with sentiment_cols[2]:
                            if sentiment['is_frustrated']:
                                st.markdown("😤 **Frustrated**")
                        
                        with sentiment_cols[3]:
                            if sentiment['is_confused']:
                                st.markdown("😕 **Confused**")
        
        # Chat input
        if kb_size > 0:
            user_input = st.chat_input("Ask your IT question here...")
            
            if user_input:
                # Analyze sentiment
                sentiment_data = st.session_state.sentiment_analyzer.analyze_sentiment(user_input)
                
                # Add user message to history
                st.session_state.chat_history.append({
                    "role": "user", 
                    "content": user_input,
                    "sentiment": sentiment_data
                })
                
                # Get bot response
                response = st.session_state.chatbot.get_response(user_input)
                
                # Add bot response to history
                st.session_state.chat_history.append({
                    "role": "assistant", 
                    "content": response
                })
                
                st.rerun()
        else:
            st.info("👆 Please upload a CSV file in the sidebar to start chatting!")
    
    with col2:
        st.header("📋 Instructions")
        
        st.markdown("""
        **How to use:**
        1. Upload a CSV file with IT Q&A data
        2. Make sure it has 'Question' and 'Answer' columns
        3. Start chatting!
        
        **CSV Format Example:**
        ```
        Question,Answer
        How do I reset my password?,Go to settings and click 'Reset Password'
        My computer is slow,Try restarting your computer first
        ```
        
        **Features:**
        - 🎭 Sentiment-aware responses
        - 🔍 Smart question matching
        - 💬 Natural conversation flow
        - 📊 Real-time statistics
        """)
        
        if kb_size > 0:
            st.header("🔍 Test Sentiment Analysis")
            test_input = st.text_input("Enter text to analyze sentiment:")
            
            if test_input:
                sentiment = st.session_state.sentiment_analyzer.analyze_sentiment(test_input)
                
                # Display sentiment analysis results
                st.subheader("Analysis Results:")
                
                col_a, col_b = st.columns(2)
                with col_a:
                    st.metric("Sentiment", sentiment['sentiment'].title())
                    st.metric("Intensity", f"{sentiment['intensity']:.2f}")
                
                with col_b:
                    st.metric("Positive Score", sentiment['positive_score'])
                    st.metric("Negative Score", sentiment['negative_score'])
                
                # Emotional states
                states = []
                if sentiment['is_urgent']: states.append("🚨 Urgent")
                if sentiment['is_frustrated']: states.append("😤 Frustrated")
                if sentiment['is_confused']: states.append("😕 Confused")
                
                if states:
                    st.write("**Emotional States:**", " | ".join(states))
        
        # Sample CSV download
        st.header("📥 Sample CSV")
        sample_data = {
            'Question': [
                'How do I reset my password?',
                'My computer is running slowly',
                'I cannot connect to the wifi',
                'How do I install new software?',
                'My screen is flickering'
            ],
            'Answer': [
                'Go to Settings > Security > Reset Password and follow the prompts.',
                'Try restarting your computer first. If the problem persists, check for running programs in Task Manager.',
                'Check if wifi is enabled, verify the network name and password, and try restarting your network adapter.',
                'Contact your IT administrator for software installation requests and approval.',
                'Check cable connections, update graphics drivers, and adjust display refresh rate in settings.'
            ]
        }
        
        sample_df = pd.DataFrame(sample_data)
        csv_sample = sample_df.to_csv(index=False)
        
        st.download_button(
            label="Download Sample CSV",
            data=csv_sample,
            file_name="sample_it_qa.csv",
            mime="text/csv"
        )

if __name__ == "__main__":
    main()
    