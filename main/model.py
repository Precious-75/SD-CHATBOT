# cli.py
# PURPOSE: Command-line entrypoint for the CSV chatbot. Loads a CSV (auto-detects
# delimiter/columns), shows stats, and starts an interactive terminal chat loop.

import os
import random
from csv_chatbot import CSVChatbot

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
