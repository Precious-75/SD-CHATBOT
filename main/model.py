import csv
import random
import re
import json
import os
from typing import Dict, List, Tuple

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

class CSVChatbot:
    def __init__(self):
        self.knowledge_base = {}
        self.config_file = "chatbot_config.json"
        self.sentiment_analyzer = SentimentAnalyzer()
        
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
    
    def load_config(self):
        """Load saved configuration"""
        try:
            if os.path.exists(self.config_file):
                with open(self.config_file, 'r', encoding='utf-8') as f:
                    config = json.load(f)
                    return config.get('csv_file_path', None)
        except Exception as e:
            print(f" Error loading config: {e}")
        return None
    
    def save_config(self, csv_file_path):
        """Save configuration"""
        try:
            config = {'csv_file_path': csv_file_path}
            with open(self.config_file, 'w', encoding='utf-8') as f:
                json.dump(config, f, indent=2)
        except Exception as e:
            print(f" Error saving config: {e}")
    
    def load_csv_file(self, file_path: str, save_to_config=True):
        """Load Q&A pairs from CSV file"""
        try:
            print(f" Looking for file: {file_path}")
            
            if not os.path.exists(file_path):
                print(f" File {file_path} not found!")
                print(" Files in current directory:")
                csv_files = [f for f in os.listdir('.') if f.endswith('.csv')]
                if csv_files:
                    for f in csv_files:
                        print(f"   - {f}")
                else:
                    print("   No CSV files found")
                return False
            
            with open(file_path, 'r', encoding='utf-8') as file:
                # Try to detect delimiter
                sample = file.read(1024)
                file.seek(0)
                
                # Use csv.Sniffer to detect delimiter
                sniffer = csv.Sniffer()
                delimiter = sniffer.sniff(sample).delimiter
                
                reader = csv.DictReader(file, delimiter=delimiter)
                fieldnames = reader.fieldnames
                
                print(f" Available columns: {fieldnames}")
                
                # Find question and answer columns
                question_col = None
                answer_col = None
                
                # Check for common column names
                for col in fieldnames:
                    col_lower = col.lower().strip()
                    if col_lower in ['question', 'questions', 'q', 'query']:
                        question_col = col
                    elif col_lower in ['answer', 'answers', 'a', 'response', 'solution']:
                        answer_col = col
                
                if not question_col or not answer_col:
                    print(" Could not automatically detect Question and Answer columns.")
                    print("Available columns:", fieldnames)
                    print("Please make sure your CSV has 'Question' and 'Answer' columns")
                    return False
                
                print(f" Using columns: '{question_col}' and '{answer_col}'")
                
                # Clear existing knowledge base
                self.knowledge_base.clear()
                
                # Load the data
                count = 0
                for row_num, row in enumerate(reader, 2):  # Start from 2 since row 1 is headers
                    try:
                        if question_col in row and answer_col in row:
                            question = row[question_col].strip() if row[question_col] else ""
                            answer = row[answer_col].strip() if row[answer_col] else ""
                            
                            if question and answer:  # Skip empty rows
                                self.add_to_knowledge_base(question, answer)
                                count += 1
                            elif not question and not answer:
                                continue  # Skip completely empty rows
                            else:
                                print(f" Row {row_num}: Missing question or answer")
                    except Exception as e:
                        print(f" Error processing row {row_num}: {e}")
                        continue
                
                print(f" Successfully loaded {count} Q&A pairs from {file_path}")
                
                # Save the file path to config for next time
                if save_to_config:
                    self.save_config(file_path)
                
                return count > 0
                
        except Exception as e:
            print(f" Error reading CSV file: {e}")
            return False
    
    def add_to_knowledge_base(self, question: str, answer: str):
        """Add a Q&A pair to the knowledge base"""
        # Normalize question for matching
        normalized_question = self.normalize_text(question)
        keywords = self.extract_keywords(question)
        
        self.knowledge_base[normalized_question] = {
            'original_question': question,
            'answer': answer,
            'keywords': keywords
        }
    
    def normalize_text(self, text: str) -> str:
        """Normalize text for better matching"""
        # Remove punctuation and convert to lowercase
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
        if not self.knowledge_base:
            return None, 0
        
        user_normalized = self.normalize_text(user_question)
        user_keywords = set(self.extract_keywords(user_question))
        
        best_match = None
        best_score = 0
        
        for stored_question, data in self.knowledge_base.items():
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
        sentiment_data = self.sentiment_analyzer.analyze_sentiment(user_input)
        
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
    
    def show_stats(self):
        """Show knowledge base statistics"""
        print(f"\n Knowledge Base Statistics:")
        print(f"   Total Q&A pairs loaded: {len(self.knowledge_base)}")
        
        if self.knowledge_base:
            # Show sample questions
            print(f"\n Sample questions (showing first 5):")
            for i, (_, data) in enumerate(list(self.knowledge_base.items())[:5], 1):
                question = data['original_question']
                if len(question) > 80:
                    question = question[:77] + "..."
                print(f"   {i}. {question}")
            
            if len(self.knowledge_base) > 5:
                print(f"   ... and {len(self.knowledge_base) - 5} more questions")
    
    def change_csv_file(self):
        """Allow user to change CSV file"""
        print("\n Current CSV file change:")
        print("Enter new CSV file path (or press Enter to browse current directory):")
        
        file_path = input("New CSV file path: ").strip()
        
        if not file_path:
            # Show available CSV files
            csv_files = [f for f in os.listdir('.') if f.endswith('.csv')]
            if csv_files:
                print("\n Available CSV files:")
                for i, f in enumerate(csv_files, 1):
                    print(f"   {i}. {f}")
                
                while True:
                    try:
                        choice = input(f"\nSelect file (1-{len(csv_files)}): ").strip()
                        if choice:
                            idx = int(choice) - 1
                            if 0 <= idx < len(csv_files):
                                file_path = csv_files[idx]
                                break
                        print("Invalid choice. Please try again.")
                    except ValueError:
                        print("Please enter a number.")
            else:
                print("No CSV files found in current directory.")
                return False
        
        return self.load_csv_file(file_path, save_to_config=True)
    
    def analyze_user_sentiment(self, text: str) -> None:
        """Debug function to show sentiment analysis results"""
        sentiment_data = self.sentiment_analyzer.analyze_sentiment(text)
        print(f"\n--- Sentiment Analysis ---")
        print(f"Text: {text}")
        print(f"Sentiment: {sentiment_data['sentiment']}")
        print(f"Frustrated: {sentiment_data['is_frustrated']}")
        print(f"Urgent: {sentiment_data['is_urgent']}")
        print(f"Confused: {sentiment_data['is_confused']}")
        print(f"Intensity: {sentiment_data['intensity']:.2f}")
        print(f"Positive Score: {sentiment_data['positive_score']}")
        print(f"Negative Score: {sentiment_data['negative_score']}")
        print("-------------------------")

def main():
    chatbot = CSVChatbot()
    
    print(" IT Support Chatbot with Sentiment Analysis")
    print("=" * 50)
    
    # Try to load previously saved CSV file
    saved_csv_path = chatbot.load_config()
    loaded = False
    
    if saved_csv_path and os.path.exists(saved_csv_path):
        print(f" Found previously used CSV file: {saved_csv_path}")
        use_saved = input("Use this file? (y/n/c to change): ").lower()
        
        if use_saved == 'c':
            loaded = chatbot.change_csv_file()
        elif use_saved != 'n':
            loaded = chatbot.load_csv_file(saved_csv_path, save_to_config=False)
    
    if not loaded:
        # Auto-load common CSV file names
        common_files = ['school_it_qa.csv', 'it_qa.csv', 'questions.csv', 'data.csv']
        
        print(" Searching for common CSV files...")
        for filename in common_files:
            if os.path.exists(filename):
                print(f" Found {filename}, loading automatically...")
                if chatbot.load_csv_file(filename):
                    loaded = True
                    break
    
    if not loaded:
        # Ask user for file path
        while True:
            print("\n Please provide your CSV file:")
            file_path = input("Enter CSV file path: ").strip()
            
            if not file_path:
                print("Please enter a file path")
                continue
            
            if chatbot.load_csv_file(file_path):
                loaded = True
                break
            else:
                retry = input("\n Try again? (y/n): ").lower()
                if retry != 'y':
                    print(" Exiting...")
                    return
    
    if loaded:
        # Show statistics
        chatbot.show_stats()
        
        print("\n CSV file loaded successfully!")
        print(" You can now start chatting!")
        print(" Special commands:")
        print("   - Type 'quit', 'exit', or 'bye' to end")
        print("   - Type 'change csv' to use a different CSV file")
        print("   - Type 'stats' to see knowledge base statistics")
        print("   - Type 'sentiment test' to analyze sentiment of your next message")
        print("-" * 50)
        
        # Start chat loop
        while True:
            try:
                user_input = input("\n You: ").strip()
                
                if not user_input:
                    continue
                
                # Check for special commands
                if user_input.lower() in ['quit', 'exit', 'bye', 'goodbye']:
                    print(" Support:", random.choice(chatbot.responses['goodbye']))
                    break
                elif user_input.lower() in ['change csv', 'change file', 'load csv']:
                    if chatbot.change_csv_file():
                        chatbot.show_stats()
                        print(" CSV file changed successfully!")
                    continue
                elif user_input.lower() in ['stats', 'statistics', 'info']:
                    chatbot.show_stats()
                    continue
                elif user_input.lower() in ['sentiment test', 'test sentiment']:
                    print(" Enter a message to analyze its sentiment:")
                    test_message = input(" Test message: ").strip()
                    if test_message:
                        chatbot.analyze_user_sentiment(test_message)
                    continue
                
                # Get and display response
                response = chatbot.get_response(user_input)
                print(" Support:", response)
                
            except KeyboardInterrupt:
                print("\n\n Goodbye!")
                break
            except Exception as e:
                print(f" Error: {e}")
                print("Please try again...")

if __name__ == "__main__":
    main()