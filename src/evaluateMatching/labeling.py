import os
import openai
openai.api_key = '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'

response = openai.ChatCompletion.create(
    model='gpt-3.5-turbo',
    max_tokens=50,
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Could you generate a Python function to calculate the factorial of a number?"},
    ])
 
message = response.choices[0]['message']
print("{}: {}".format(message['role'], message['content']))