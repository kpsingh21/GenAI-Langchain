from langchain_openai import ChatOpenAI
from dotenv import load_dotenv

load_dotenv()

llm = ChatOpenAI(model="gpt-5.2",temparature=1.5 , max_completion_tokens=100)
result = llm.invoke("What is the capital of India?")
print(result)



# Temperature is a crucial LLM parameter, typically ranging from 0 to 2, that controls the randomness and creativity of generated text by scaling the probability distribution of the next word. Lower values (0.0–0.4) make outputs deterministic, precise, and repetitive, while higher values (1.0–2.0) increase creativity and randomness, often used in API setting.
# max_completion_tokens =~ no of words