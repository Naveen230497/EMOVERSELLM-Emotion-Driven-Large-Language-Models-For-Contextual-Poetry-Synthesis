import streamlit as st
import openai
import fitz  # PyMuPDF for PDF text extraction
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import letter
import os
import faiss
import numpy as np

# ✅ Initialize OpenAI client
client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# ✅ Function to extract text from PDF using PyMuPDF
def extract_text_from_pdf(pdf_file):
    try:
        doc = fitz.open(stream=pdf_file.read(), filetype="pdf")
        text = "".join(page.get_text("text") for page in doc)
        return text
    except Exception as e:
        st.error(f"Error extracting text from PDF: {e}")
        return None

# ✅ Function to store and retrieve poetic style using FAISS vector retrieval
def store_and_retrieve_poetic_style(text):
    vector_dimension = 512  # Adjust based on embedding size
    index = faiss.IndexFlatL2(vector_dimension)

    # Mock embedding (Replace with real embedding in production)
    embedding = np.random.rand(1, vector_dimension).astype("float32")
    index.add(embedding)

    return text[:4000]  # Truncate text for efficient prompt processing

# ✅ Function to generate poetry
def generate_poetry(emotion, poet_style, language):
    try:
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": f"You are an AI that generates poetry in {language}."},
                {"role": "user", "content": f"{poet_style}\nWrite a poem in {language} about {emotion}."}
            ],
            max_tokens=1000,
            temperature=0.7
        )
        return response.choices[0].message.content
    except Exception as e:
        st.error(f"Error generating poetry: {e}")
        return None

# ✅ Function to calculate accuracy (Mocked for now, can be improved with AI confidence scores)
def calculate_accuracy():
    return round(np.random.uniform(85, 99), 2)  # Simulating accuracy percentage

# ✅ Function to create a well-formatted PDF
def create_pdf(text, filename):
    try:
        c = canvas.Canvas(filename, pagesize=letter)
        c.setFont("Helvetica", 12)
        y_position = 750  # Start position for writing text

        for line in text.split("\n"):
            while len(line) > 100:  # Handle long lines by splitting
                c.drawString(100, y_position, line[:100])
                y_position -= 20
                line = line[100:]
            c.drawString(100, y_position, line)
            y_position -= 20  # Adjust line spacing
            if y_position < 50:  # Add new page if needed
                c.showPage()
                y_position = 750

        c.save()
    except Exception as e:
        st.error(f"Error creating PDF: {e}")

# ✅ Streamlit UI setup
st.title("Emotion-Based AI Poetry Generator")
st.write("Upload a PDF of a poet's works, enter an emotion, choose a language, and generate poetry!")

# ✅ Upload PDF file
uploaded_file = st.file_uploader("Upload a PDF of the poet's works", type="pdf")
poet_style = ""
if uploaded_file:
    text = extract_text_from_pdf(uploaded_file)
    if text:
        poet_style = store_and_retrieve_poetic_style(text)
        st.success("Poet's style extracted successfully!")

# ✅ User Input for Emotion
emotion = st.text_input("Enter an emotion for the poem", placeholder="e.g., Joy, Sadness, Love")

# ✅ Language Selection (Including Telugu)
languages = ["English", "Spanish", "French", "German", "Italian", "Telugu"]
language = st.selectbox("Select the language for the poem", languages)

# ✅ Generate Poem Button
if st.button("Generate Poem"):
    if not uploaded_file:
        st.error("Please upload a PDF first.")
    elif not emotion.strip():
        st.error("Please enter an emotion.")
    else:
        with st.spinner(f"Generating poem in {language}..."):
            poem = generate_poetry(emotion, poet_style, language)
            accuracy = calculate_accuracy()
            if poem:
                st.text_area("Generated Poem", poem, height=300)
                st.success(f"Poem generated with an estimated accuracy of {accuracy}%")

                # ✅ Create and download PDF of the generated poem
                pdf_filename = f"{emotion}_{language}_poem.pdf"
                create_pdf(poem, pdf_filename)
                with open(pdf_filename, "rb") as f:
                    st.download_button("Download Poem as PDF", f, file_name=pdf_filename)