import pprint
import google.generativeai as genai
from load_creds import load_creds

creds = load_creds()

genai.configure(credentials=creds)

print()
print('Available base models:', [m.name for m in genai.list_models()])