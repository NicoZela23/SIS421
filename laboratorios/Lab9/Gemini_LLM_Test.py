import google.generativeai as genai
from load_creds import load_creds

creds = load_creds()

genai.configure(credentials=creds)

# Set up the model
generation_config = {
  "temperature": 0.15,
  "top_p": 1,
  "top_k": 0,
  "max_output_tokens": 8192,
}

safety_settings = [
]

model = genai.GenerativeModel(model_name="tunedModels/mykeyboardqa-kalkoww5l19h",
                              generation_config=generation_config,
                              safety_settings=safety_settings)

prompt_parts = [
  "input: Is my keyboard RGB?",
  "output: ",
]

response = model.generate_content(prompt_parts)
print(response.text)