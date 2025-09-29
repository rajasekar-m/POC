import re
import streamlit as st
from dotenv import load_dotenv
from youtube_transcript_api import YouTubeTranscriptApi
import time

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate


load_dotenv()

def extract_video_id(url):
    
    pattern = re.search(r'(?:v=|\/)([0-9A-Za-z_-]{11}).*', url)
    if pattern:
        return pattern.group(1)
    else:
        st.error("Invalid YouTube URL. Please enter a valid URL.")
        return None
    


def get_transcript(video_id, language):
    
    ytt_api = YouTubeTranscriptApi()
    try:
        transcript = ytt_api.fetch(video_id, languages=[language, 'en'])
        full_transcript = " ".join([i.text for i in transcript])
        time.sleep(10)
        return full_transcript
    except Exception as e:
        st.error(f"Error fetching transcript: {e}")
        return None
    
    
llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash", temperature=0.2)

def translate_transcript(transcript):
    try:
        
        prompt = ChatPromptTemplate.from_template("""
        You are an expert translator with deep cultural and linguistic knowledge.
        I will provide you with a transcript. Your task is to translate it into English with absolute accuracy, preserving:
        - Full meaning and context (no omissions, no additions).
        - Tone and style (formal/informal, emotional/neutral as in original).
        - Nuances, idioms, and cultural expressions (adapt appropriately while keeping intent).
        - Speaker's voice (same perspective, no rewriting into third-person).
        Do not summarize or simplify. The translation should read naturally in the target language but stay as close as possible to the original intent.

        Transcript:
        {transcript}
        """)
        
        chain = prompt | llm
        response = chain.invoke({"transcript": transcript})
        return response.content
    
    except Exception as e:
        st.error(f"Error translating transcript: {e}")
        return None
    
# function to get important topics

def get_important_topics(transcript):
    
    try:
        
        prompt = ChatPromptTemplate.from_template("""
               You are an assistant that extracts the 5 most important topics discussed in a video transcript or summary.

               Rules:
               - Summarize into exactly 5 major points.
               - Each point should represent a key topic or concept, not small details.
               - Keep wording concise and focused on the technical content.
               - Do not phrase them as questions or opinions.
               - Output should be a numbered list.
               - show only points that are discussed in the transcript.
               Here is the transcript:
               {transcript}
               """
        )
        
        chain = prompt | llm
        response = chain.invoke({"transcript": transcript})
        return response.content
    
    except Exception as e:
        st.error(f"Error extracting important topics: {e}")
        return None
    
    
# function to generate notes

def generate_notes(transcript):
    
    try:
        
        prompt = ChatPromptTemplate.from_template("""
                You are an AI note-taker. Your task is to read the following YouTube video transcript 
                and produce well-structured, concise notes.

                Requirements:
                - Present the output as **bulleted points**, grouped into clear sections.
                - Highlight key takeaways, important facts, and examples.
                - Use **short, clear sentences** (no long paragraphs).
                - If the transcript includes multiple themes, organize them under **subheadings**.
                - Do not add information that is not present in the transcript.

                Here is the transcript:
                {transcript}
                """
        )
        
        chain = prompt | llm
        response = chain.invoke({"transcript": transcript})
        return response.content
    
    except Exception as e:
        st.error(f"Error extracting important topics: {e}")
        return None



        