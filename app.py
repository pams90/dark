# app.py
import streamlit as st
from transformers import AutoTokenizer, pipeline
import torch
from chromadb import Client, Settings
import time

# ======================
# App Configuration
# ======================
st.set_page_config(
    page_title="Dark Romance Book Factory", 
    page_icon="🖤", 
    layout="wide",
    initial_sidebar_state="expanded"
)

# ======================
# NSFW Age Gate
# ======================
if "age_verified" not in st.session_state:
    st.title("Age Verification 🔞")
    st.markdown("This app contains mature content suitable for adults only.")
    
    col1, col2 = st.columns(2)
    with col1:
        if st.button("I am at least 18 years old", use_container_width=True):
            st.session_state.age_verified = True
            st.rerun()
    with col2:
        if st.button("Exit", use_container_width=True):
            st.stop()
    st.stop()

# ======================
# ChromaDB Memory Setup
# ======================
@st.cache_resource
def setup_memory():
    return Client(settings=Settings(
        persist_directory="./chroma_data",
        allow_reset=True,
        anonymized_telemetry=False
    ))

chroma_client = setup_memory()
collection = chroma_client.get_or_create_collection("dark_romance")

# ======================
# AI Model Loader (Optimized)
# ======================
@st.cache_resource
def load_model():
    model_name = "mistralai/Mistral-7B-v0.1"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    return pipeline(
        "text-generation",
        model=model_name,
        tokenizer=tokenizer,
        device_map="auto" if torch.cuda.is_available() else None,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        max_length=2000,
        do_sample=True,
        temperature=0.8,
        top_k=50,
        pad_token_id=tokenizer.eos_token_id
    )

generator = load_model()

# ======================
# Generation Utilities
# ======================
def style_randomizer(text: str) -> str:
    """Add human-like variations to generated text"""
    replacements = {
        " however": "; however",
        " therefore": ". Therefore",
        "The": "The" if torch.rand(1) > 0.5 else "The"
    }
    for k, v in replacements.items():
        text = text.replace(k, v, 1)
    return text

def build_prompt(user_inputs: dict, previous_chapters: list) -> str:
    return f"""
    [INST] Write a dark romance chapter with these specifications:
    Genre: {user_inputs['genre']}
    Main Characters: {', '.join(user_inputs['characters'])}
    Story Beat: {user_inputs['current_instruction']}
    Style: {user_inputs['style']}
    Taboos: {', '.join(user_inputs['taboos']) if user_inputs['taboos'] else 'None'}
    
    Previous Context: {previous_chapters[-2:] if previous_chapters else "None"}
    
    Include:
    - Atmospheric descriptions
    - Moral ambiguity
    - Forbidden desire
    - At least one plot twist
    [/INST]
    """

# ======================
# Streamlit UI
# ======================
st.title("🖤 Dark Romance Book Factory")
st.caption("Generate 100% Original Dark Romance Novels - Chapter by Chapter")

# ======================
# User Controls (Sidebar)
# ======================
with st.sidebar:
    st.header("⚙️ Configuration")
    
    # Genre Selection
    genre = st.selectbox("Book Type", [
        "Mafia Romance", 
        "Vampire Dark Romance", 
        "Gothic Horror Romance",
        "BDSM Power Dynamics"
    ], index=1)
    
    # Character Configuration
    num_protagonists = st.slider("Main Characters", 1, 3, 2)
    protagonists = []
    for i in range(num_protagonists):
        protagonists.append(st.text_input(
            f"Character {i+1} Description", 
            value=f"Morally gray {['mafia boss', 'vampire lord', 'tortured billionaire'][i%3]}",
            key=f"char_{i}"
        ))
    
    # Content Levers
    style = st.selectbox("Writing Style", [
        "Visceral First-Person POV",
        "Atmospheric Third-Person Limited",
        "Cinematic Present Tense"
    ], index=0)
    
    taboos = st.multiselect("Allowed Content", [
        "Non-Consensual Elements",
        "Psychological Manipulation",
        "Supernatural Violence",
        "Explicit Intimacy"
    ], default=["Supernatural Violence"])
    
    current_instruction = st.text_area(
        "📝 Chapter Instructions", 
        placeholder="e.g., 'Introduce betrayal scene with hidden supernatural powers'",
        height=100
    )

# ======================
# Main Generation Logic
# ======================
if st.button("✨ Generate Next Chapter", use_container_width=True):
    if not current_instruction:
        st.error("Please provide chapter instructions!")
        st.stop()
    
    user_inputs = {
        "genre": genre,
        "characters": protagonists,
        "style": style,
        "taboos": taboos,
        "current_instruction": current_instruction
    }
    
    # Retrieve context
    previous_chapters = collection.get()["documents"] if collection.count() > 0 else []
    
    # Generate chapter
    with st.spinner("Crafting your dark chapter..."):
        try:
            start_time = time.time()
            prompt = build_prompt(user_inputs, previous_chapters)
            response = generator(prompt, max_new_tokens=1500)[0]['generated_text']
            new_chapter = style_randomizer(response.split("[/INST]")[-1].strip())
            gen_time = time.time() - start_time
        except Exception as e:
            st.error(f"Generation failed: {str(e)}")
            st.stop()
    
    # Store in memory
    collection.add(
        documents=[new_chapter],
        ids=f"chapter_{collection.count() + 1}"
    )
    
    # Display results
    st.subheader(f"📖 Chapter {collection.count()}")
    st.write(new_chapter)
    st.caption(f"⏱️ Generated in {gen_time:.2f} seconds")

# ======================
# Book Management
# ======================
if collection.count() > 0:
    with st.expander("📚 Full Manuscript Preview", expanded=False):
        for idx, chapter in enumerate(collection.get()["documents"]):
            st.subheader(f"Chapter {idx + 1}")
            st.write(chapter)
            st.divider()

    st.download_button(
        label="💾 Download Manuscript",
        data="\n\n".join(collection.get()["documents"]),
        file_name="dark_romance_novel.txt",
        use_container_width=True
    )

    if st.button("🧹 Reset Book", type="secondary", use_container_width=True):
        collection.delete()
        st.rerun()