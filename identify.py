import os
from llama_index.core import load_index_from_storage, StorageContext
import genai



# Initialize environment variables and storage context for the vector database
# Here should a openai api key be provided

PERSIST_DIR = "./storage"
storage_context = StorageContext.from_defaults(persist_dir=PERSIST_DIR)
index = load_index_from_storage(storage_context)


class Chatbot:
    def __init__(self):
        self.query_engine = index.as_query_engine()
        self.welcome_message = "Hello! I'm your healthcare assistant. How can I help you today?"
        self.goodbye_message = "Goodbye! If you have more questions later, just ask."
        self.error_message = "I'm sorry, I didn't understand that. Could you rephrase or ask something else?"

    def get_response(self, user_input):
        # Query the vector database for a response
        try:
            response = self.query_engine.query(user_input)
            return response if response else self.error_message
        except Exception as e:
            print(f"Error querying the vector database: {e}")
            return self.error_message

    def start(self):
        print(self.welcome_message)
        while True:
            user_input = input("You: ").strip()
            if user_input.lower() in ["exit", "quit", "bye"]:
                print(self.goodbye_message)
                break
            possibleDis = self.get_response("please provide all relevent diseases' name" + user_input)
            response = self.get_response(user_input)
            print("Bot:", response)
            # print(possibleDis)
            genai.detection(str(possibleDis).split(), "/Users/snows/Programming/hackthon/genai2024/genai/" + "4.png")
            # Please set the path above the absolute path in your own device


if __name__ == "__main__":
    chatbot = Chatbot()
    chatbot.start()
    # To test the program, please enter "my eyes are not feeling good", this is because a part of the
    # program relay on image to prediect the disease
