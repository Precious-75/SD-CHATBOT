# csv_chatbot.py
# PURPOSE: Implements CSVChatbot which loads Q&A pairs from a CSV, detects intent,
# matches user questions using Jaccard + keyword overlap, and integrates sentiment
# prefixes. This file depends on SentimentAnalyzer from sentiment_analyzer.py.

from __future__ import annotations
import csv
import json
import random
import re
import os
from typing import Dict, List, Tuple

from sentiment_analyzer import SentimentAnalyzer  # local import


class CSVChatbot:
    def __init__(self):
        self.knowledge_base: Dict[str, Dict] = {}
        self.config_file = "chatbot_config.json"
        self.sentiment_analyzer = SentimentAnalyzer()

        # ---- Responses & intent patterns (restored/defaults) ----
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
            'capabilities': [
                "I can help with school IT issues like Wi-Fi, email/login, printers & projectors, LMS/gradebook, classroom devices (iPads, smartboards), software installs/updates, and basic troubleshooting. Ask me anything in those areas!"
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
            'greeting': [r'(^|\b)hello\b', r'(^|\b)hi\b', r'(^|\b)hey\b', r'good morning', r'good afternoon'],
            'goodbye': [r'(^|\b)bye\b', r'goodbye', r'see you', r'farewell', r'\bthanks?\b', r'thank you'],
            'capabilities': [r'what can you do',r'how (can|do) you help',r'what (are|is) your (skills|capabilities|functions?)',r'what can you (answer|handle)']

        }

    # ---------- Config persistence ----------
    def load_config(self) -> str | None:
        """Load saved CSV path from config file."""
        try:
            if os.path.exists(self.config_file):
                with open(self.config_file, 'r', encoding='utf-8') as f:
                    config = json.load(f)
                    return config.get('csv_file_path', None)
        except Exception as e:
            print(f" Error loading config: {e}")
        return None

    def save_config(self, csv_file_path: str) -> None:
        """Save CSV path to config file."""
        try:
            with open(self.config_file, 'w', encoding='utf-8') as f:
                json.dump({'csv_file_path': csv_file_path}, f, indent=2)
        except Exception as e:
            print(f" Error saving config: {e}")

    # ---------- CSV loading ----------
    def load_csv_file(self, file_path: str, save_to_config: bool = True) -> bool:
        """Load Q&A pairs from CSV file (supports optional Category/School_Level)."""
        try:
            print(f" Looking for file: {file_path}")

            if not os.path.exists(file_path):
                print(f" File {file_path} not found!")
                return False

            with open(file_path, 'r', encoding='utf-8') as file:
                sample = file.read(1024)
                file.seek(0)
                # Robust delimiter detection with fallback to comma
                try:
                    sniffer = csv.Sniffer()
                    dialect = sniffer.sniff(sample)
                    delimiter = dialect.delimiter
                except Exception:
                    delimiter = ','

                reader = csv.DictReader(file, delimiter=delimiter)
                fieldnames = [fn.strip() for fn in (reader.fieldnames or [])]

                print(f" Available columns: {fieldnames}")

                # Core columns
                question_col = None
                answer_col = None

                # Optional metadata
                category_col = None
                level_col = None

                for col in fieldnames:
                    low = col.lower().strip()
                    if low in ['question', 'questions', 'q', 'query']:
                        question_col = col
                    elif low in ['answer', 'answers', 'a', 'response', 'solution']:
                        answer_col = col
                    elif low in ['category', 'topic', 'tag']:
                        category_col = col
                    elif low in ['school_level', 'school level', 'level', 'grade_level', 'grade level']:
                        level_col = col

                if not question_col or not answer_col:
                    print(" Could not automatically detect Question and Answer columns.")
                    print("Available columns:", fieldnames)
                    print("Please make sure your CSV has 'Question' and 'Answer' columns")
                    return False

                print(
                    f" Using columns: '{question_col}' and '{answer_col}', "
                    f"meta: {category_col or 'Category? none'}, {level_col or 'School_Level? none'}"
                )

                self.knowledge_base.clear()
                count = 0

                for row_num, row in enumerate(reader, 2):
                    try:
                        q = (row.get(question_col) or "").strip()
                        a = (row.get(answer_col) or "").strip()
                        if not q and not a:
                            continue
                        if q and a:
                            meta = {
                                'category': (row.get(category_col) or "Uncategorized").strip() if category_col else "Uncategorized",
                                'school_level': (row.get(level_col) or "All").strip() if level_col else "All",
                            }
                            self.add_to_knowledge_base(q, a, meta)
                            count += 1
                        else:
                            print(f" Row {row_num}: Missing question or answer")
                    except Exception as e:
                        print(f" Error processing row {row_num}: {e}")
                        continue

                print(f" Successfully loaded {count} Q&A pairs from {file_path}")
                if count > 0 and save_to_config:
                    self.save_config(file_path)
                return count > 0

        except Exception as e:
            print(f" Error reading CSV file: {e}")
            return False

    # ---------- KB ops ----------
    def add_to_knowledge_base(self, question: str, answer: str, meta: Dict | None = None) -> None:
        """Add a Q&A pair to the knowledge base, with optional metadata."""
        normalized_question = self.normalize_text(question)
        keywords = self.extract_keywords(question)
        self.knowledge_base[normalized_question] = {
            'original_question': question,
            'answer': answer,
            'keywords': keywords,
            'meta': meta or {'category': 'Uncategorized', 'school_level': 'All'}
        }

    # ---------- Text utils ----------
    def normalize_text(self, text: str) -> str:
        """Normalize text for better matching (strip, lower, remove punctuation)."""
        return re.sub(r'[^\w\s]', '', text.lower().strip())

    def extract_keywords(self, text: str) -> List[str]:
        """Extract keywords from text (stopword-filtered tokens)."""
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
        return [w for w in words if w not in stop_words and len(w) > 2]

    # ---------- Matching ----------
    def find_best_match(self, user_question: str) -> Tuple[str | None, float]:
        """Find the best matching answer for user question via Jaccard + keyword overlap."""
        if not self.knowledge_base:
            return None, 0.0

        user_normalized = self.normalize_text(user_question)
        user_keywords = set(self.extract_keywords(user_question))

        best_match = None
        best_score = 0.0

        for stored_question, data in self.knowledge_base.items():
            # Text similarity (Jaccard)
            similarity_score = self.calculate_text_similarity(user_normalized, stored_question)

            # Keyword similarity (overlap ratio)
            stored_keywords = set(data['keywords'])
            if user_keywords and stored_keywords:
                keyword_overlap = len(user_keywords.intersection(stored_keywords))
                max_keywords = max(len(user_keywords), len(stored_keywords))
                keyword_score = (keyword_overlap / max_keywords) if max_keywords > 0 else 0.0
            else:
                keyword_score = 0.0

            # Combined score (text similarity weighted more heavily)
            total_score = (similarity_score * 0.7) + (keyword_score * 0.3)

            if total_score > best_score:
                best_score = total_score
                best_match = data['answer']

        return best_match, float(best_score)

    def calculate_text_similarity(self, text1: str, text2: str) -> float:
        """Calculate similarity between two texts using Jaccard similarity."""
        words1 = set(text1.split())
        words2 = set(text2.split())

        if not words1 and not words2:
            return 1.0
        if not words1 or not words2:
            return 0.0

        intersection = words1.intersection(words2)
        union = words1.union(words2)
        return (len(intersection) / len(union)) if union else 0.0

    # ---------- Intent & sentiment ----------
    def detect_intent(self, message: str) -> str:
        message_lower = message.lower()
        for intent, patterns in getattr(self, "patterns", {}).items():
            for pattern in patterns:
                if re.search(pattern, message_lower):
                    return intent
        return 'question'


    def get_sentiment_response_prefix(self, sentiment_data: Dict) -> str:
        """Get appropriate response prefix based on sentiment."""
        if sentiment_data.get('is_frustrated'):
            return random.choice(self.responses['frustrated'])
        elif sentiment_data.get('is_urgent'):
            return random.choice(self.responses['urgent'])
        elif sentiment_data.get('is_confused'):
            return random.choice(self.responses['confused'])
        elif sentiment_data.get('sentiment') == 'positive':
            return random.choice(self.responses['positive'])
        return ""

    def get_response(self, user_input: str) -> str:
        """Generate response to user input with sentiment awareness."""
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

    # ---------- CLI helpers ----------
    def show_stats(self) -> None:
        """Show knowledge base statistics (console)."""
        print(f"\n Knowledge Base Statistics:")
        print(f"   Total Q&A pairs loaded: {len(self.knowledge_base)}")

        if self.knowledge_base:
            print(f"\n Sample questions (showing first 5):")
            for i, (_, data) in enumerate(list(self.knowledge_base.items())[:5], 1):
                question = data['original_question']
                if len(question) > 80:
                    question = question[:77] + "..."
                print(f"   {i}. {question}")

            if len(self.knowledge_base) > 5:
                print(f"   ... and {len(self.knowledge_base) - 5} more questions")

    def change_csv_file(self) -> bool:
        """Allow user to change CSV file (console)."""
        print("\n Current CSV file change:")
        print("Enter new CSV file path (or press Enter to browse current directory):")

        file_path = input("New CSV file path: ").strip()

        if not file_path:
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
        """Debug function to show sentiment analysis results (console)."""
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
