import openai
openai.api_key = "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"

model_engine = "text-chat-davinci-002-20221122"
while(True):
    print("Enter prompt")
    prompt = input()
    
    completion = openai.Completion.create(
        model=model_engine,
        prompt=prompt,
        max_tokens=250,
        temperature=0.5,
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0,
    )
    
    message = completion["choices"][0]["text"]
    print(message.replace("<|im_end|>", ""))
    print()