import os
from apikey import apikey

import streamlit as st
from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain, SequentialChain

os.environ['OPENAI_API_KEY'] = apikey

#App framework
st.title('🦜🔗 UX Research Assistance')
prompt = st.text_input("Tell me a little bit about your app and what problem is your app trying to solve...")

#Prompt template
user_template = PromptTemplate(
    input_variables=["topic"],
    template= """You are an excellent UIUX designer. We are trying to build an app with description: "{topic}"". Please find some potential client groups that would like to use our app"""
)
value_template = PromptTemplate(
    input_variables=["topic", "user"],
    template= """Point out some values why users would like to use our application based on the app: "{topic}" and our target user: "{user}"."""
)
solution_template = PromptTemplate(
    input_variables=["topic","value"],
    template= """Based on the app description:"{topic}" and our values: "{value}". Write me some solution ideas that we can provide to the users."""
)
question_template = PromptTemplate(
    input_variables=["topic","solution"],
    template= """Based on the app description:"{topic}" and predict solution: "{solution}". Generate 10 questions for our future user interview."""
)
slogan_template = PromptTemplate(
    input_variables=["topic","solution"],
    template= """Based on the app description:"{topic}" and predict solution: "{solution}". Generate 10 call to action headers or slogan that align with our vision."""
)
aspect_template = PromptTemplate(
    input_variables=["topic", "solution"],
    template= """Based on the app description:"{topic}" and predict solution: "{solution}". Explore some aspects we can consider during the design and development process."""
)
ia_template = PromptTemplate(
    input_variables=["topic","solution"],
    template= """Based on the app description:"{topic}" and predict solutions: "{solution}". Define the information architecture we need to build this app."""
)
prioritize_template = PromptTemplate(
    input_variables=["topic","ia", "solution"],
    template= """Based on the app description:"{topic}", predict solution: "{solution}" and infromation Architecture: "{ia}". Prioritize the feature we need."""
)


#Config llm
llm = OpenAI(temperature=0.9)

#Chain
user_chain = LLMChain(llm=llm, prompt=user_template, verbose=True, output_key="user")
value_chain = LLMChain(llm=llm, prompt=value_template, verbose=True, output_key="value")
solution_chain = LLMChain(llm=llm, prompt=solution_template, verbose=True, output_key="solution")
question_chain = LLMChain(llm=llm, prompt=question_template, verbose=True, output_key="question")
slogan_chain = LLMChain(llm=llm, prompt=slogan_template, verbose=True, output_key="slogan")
aspect_chain = LLMChain(llm=llm, prompt=aspect_template, verbose=True, output_key="aspect")
ia_chain = LLMChain(llm=llm, prompt=ia_template, verbose=True, output_key="ia")
prioritize_chain = LLMChain(llm=llm, prompt=prioritize_template, verbose=True, output_key="prioritize")
sequential_chain = SequentialChain(
    chains=[
        user_chain,
        value_chain,
        solution_chain,
        question_chain,
        slogan_chain,
        aspect_chain,
        ia_chain,
        prioritize_chain,
    ],
    input_variables=[
        "topic",
    ],
    output_variables=[
        "user",
        "value",
        "solution",
        "question",
        "slogan",
        "aspect",
        "ia",
        'prioritize'
    ],
    verbose=True
)


#Action
if prompt:
    response = sequential_chain({"topic":prompt})

    st.header("Potential User Prediction:")
    st.write(response["user"])
    st.header("Values We Can Provide:")
    st.write(response["value"])
    st.header("Solution Ideas:")
    st.write(response["solution"])
    st.header("Interview Question for Future Users:")
    st.write(response["question"])
    st.header("Call-To-Action Headers Ideas:")
    st.write(response["slogan"])
    st.header("Aspects to Consider:")
    st.write(response["aspect"])
    st.header("Information Architecture:")
    st.write(response["ia"])
    st.header("Features Priority:")
    st.write(response["prioritize"])