import streamlit as st
from langchain_ollama import ChatOllama  # Corrected import
from langchain.globals import set_debug
from langchain.prompts import PromptTemplate

set_debug(True)

# Corrected prompt template
prompt = PromptTemplate(
    input_variables=['position', 'company', 'strengths', 'weaknesses'],  # Fixed `input_variable` -> `input_variables`
    template="""You are a career coach. Provide tailored interview tips for the 
    position of {position} at {company}. 
    Highlight strengths in {strengths} and prepare for questions 
    about weaknesses such as {weaknesses}.
    """
)

llm = ChatOllama(model="llama3:latest")  # Ensure Llama 3 is installed in Ollama

# Streamlit UI
st.title("Interview Tip's Generator")

position = st.text_input("Enter your position for the company:")
company = st.text_input("Enter the company name:")
strengths = st.text_area("Enter your strengths:")
weaknesses = st.text_area("Enter your weaknesses:")

if position and company and strengths and weaknesses:
    # Generate formatted prompt
    formatted_prompt = prompt.format_prompt(
        position=position,
        company=company,
        strengths=strengths,
        weaknesses=weaknesses
    ).to_messages()

    # Invoke LLM
    response = llm.invoke(formatted_prompt)

    # Extract response content
    if hasattr(response, 'content'):
        st.write(response.content)
    else:
        st.write(response)  # Fallback in case response structure changes
