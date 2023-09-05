import openai

openai.api_key = "OPENAI_API_KEY"

messages = [{"role": "system", "content": "You are chatting with an AI assistant."}]

print("Your new assistant is ready!")
while True:
    message = input()
    messages.append({"role": "user", "content": message})

    if message.lower() == "exit":
        break

    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=messages)
    reply = response["choices"][0]["message"]["content"]
    messages.append({"role": "assistant", "content": reply})
    print("\n" + reply + "\n")