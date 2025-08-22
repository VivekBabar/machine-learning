from model import jarvis_model

# Test the model
test_questions = [
    "hi",
    "hello", 
    "how are you",
    "what can you do",
    "who are you",
    "test question"
]

print("Testing Jarvis model:")
print("=" * 50)

for question in test_questions:
    response = jarvis_model.predict(question)
    print(f"Q: {question}")
    print(f"A: {response}")
    print("-" * 30)