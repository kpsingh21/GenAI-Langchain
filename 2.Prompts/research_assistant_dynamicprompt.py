from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv
import os
import streamlit as st
from langchain_core.prompts import PromptTemplate,load_prompt

load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id="mistralai/Mistral-7B-Instruct-v0.2",
    task="text-generation",
    # huggingfacehub_api_token=os.getenv("HUGGINGFACEHUB_API_TOKEN"),
)
chat_model = ChatHuggingFace(llm=llm)
st.header("I know Everything!")


paper_input = st.selectbox( "Select Research Paper Name", ["Attention Is All You Need", "BERT: Pre-training of Deep Bidirectional Transformers", "GPT-3: Language Models are Few-Shot Learners", "Diffusion Models Beat GANs on Image Synthesis"] )

style_input = st.selectbox( "Select Explanation Style", ["Beginner-Friendly", "Technical", "Code-Oriented", "Mathematical"] ) 

length_input = st.selectbox( "Select Explanation Length", ["Short (1-2 paragraphs)", "Medium (3-5 paragraphs)", "Long (detailed explanation)"] )


# template = PromptTemplate(template= "\nPlease summarize the research paper titled \"{paper_input}\" with the following specifications:\nExplanation Style: {style_input}  \nExplanation Length: {length_input}  \n1. Mathematical Details:  \n   - Include relevant mathematical equations if present in the paper.  \n   - Explain the mathematical concepts using simple, intuitive code snippets where applicable.  \n2. Analogies:  \n   - Use relatable analogies to simplify complex ideas.  \nIf certain information is not available in the paper, respond with: \"Insufficient information available\" instead of guessing.  \nEnsure the summary is clear, accurate, and aligned with the provided style and length.\n",
# input_variables=['paper_input','style_input','length_input'],
# # validate_template=True
# )

# now we can load that template here from template.json file

template = load_prompt('template.json')

# dic: { "key":"value"}
# list: [ '','', '']

# prompt=template.invoke({
#     'paper_input':paper_input,
#     'style_input': style_input,
#     'length_input': length_input
# })

# here we are using invoke two times one during template and other during prompt
#  we can make chain

if st.button("Can I Answer!!!"):
    # result = chat_model.invoke(prompt)
    # st.write(result.content)

    chain = template | chat_model
    result= chain.invoke({
    'paper_input':paper_input,
    'style_input': style_input,
    'length_input': length_input
})
    st.write(result.content)


       
