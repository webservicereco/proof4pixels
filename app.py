import docx
import pandas as pd
import numpy as np
import io
import os
import re
import pickle
import datetime
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import streamlit as st  # Streamlit for Chrome Web Interface

# PDF Generation Libraries
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak

# =====================================================================
# PHASE 1: OFFLINE TRAINING AND CLASSIFICATION MATRIX EXPORT
# =====================================================================

# =====================================================================
# EXPLICIT DIRECTORY PATH FIX (15Aug26)
# =====================================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__)) if '__file__' in globals() else os.getcwd()
os.chdir(BASE_DIR)

CLASSIFIER_PATH = os.path.join(BASE_DIR, 'moses_classifier.pkl')
VECTORIZER_PATH = os.path.join(BASE_DIR, 'moses_vectorizer.pkl')

def extract_stylistic_features(texts):
    """Extracts linguistic statistical features for MoSEs stylistic modeling."""
    feats = []
    for text in texts:
        words = str(text).split()
        num_words = max(len(words), 1)
        num_chars = max(len(str(text)), 1)
        avg_word_len = num_chars / num_words
        sentences = [s for s in re.split(r'[.!?]+', str(text)) if s.strip()]
        num_sents = max(len(sentences), 1)
        avg_sent_len = num_words / num_sents
        
        punct_count = sum(1 for c in str(text) if c in '.,;:!?"\'()-[]{}')
        punct_ratio = punct_count / num_chars
        upper_ratio = sum(1 for c in str(text) if c.isupper()) / num_chars
        unique_words = len(set(w.lower() for w in words))
        ttr = unique_words / num_words # Type-Token Ratio
        
        feats.append([num_words, avg_word_len, num_sents, avg_sent_len, punct_ratio, upper_ratio, ttr])
    return np.array(feats)

def build_and_serialize_matrix(docx_path):
    print(f"[*] Reading and parsing training document: {docx_path}...")
    doc = docx.Document(docx_path)
    lines = [p.text for p in doc.paragraphs if p.text.strip()]
    csv_raw = "\n".join(lines)
    df = pd.read_csv(io.StringIO(csv_raw))
    
    # Map dataset formatting: label=0 is machine (AI). Map target_ai=1 for AI, 0 for Human.
    df['target_ai'] = 1 - df['label']
    
    X_text = df['text'].astype(str).values
    y = df['target_ai'].values
    
    print("[*] Performing feature extraction and stylistic analysis...")
    style_feats = extract_stylistic_features(X_text)
    
    vectorizer = TfidfVectorizer(max_features=2000, stop_words='english', ngram_range=(1,2))
    X_tfidf = vectorizer.fit_transform(X_text).toarray()
    X_combined = np.hstack([X_tfidf, style_feats])
    
    print("[*] Fitting style-conditional classification matrix...")
    clf = LogisticRegression(max_iter=1000, C=1.2)
    clf.fit(X_combined, y)
    
    with open(CLASSIFIER_PATH, 'wb') as f:
        pickle.dump(clf, f)
    with open(VECTORIZER_PATH, 'wb') as f:
        pickle.dump(vectorizer, f)
        
    print(f"[+] Core classification matrix exported successfully to '{CLASSIFIER_PATH}'.\n")

# =====================================================================
# PHASE 2: ONLINE REAL-TIME CHROME BROWSER INTERFACE ENGINE
# =====================================================================

class RealTimeAIDetectorApp:
    def __init__(self, model_name="gpt2", classifier_path=CLASSIFIER_PATH, vectorizer_path=VECTORIZER_PATH):
        # 1. Initialize Causal LM for Zero-Shot Fast-DetectGPT Curvature Extraction
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"[*] Initializing deep curvature transformer on [{self.device}]...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name).to(self.device)
        self.model.eval()
        
        # 2. Deserializing Pre-Trained Weights Matrix into active Memory
        print(f"[*] Loading trained classification matrix weights from '{classifier_path}' into memory...")
        with open(classifier_path, 'rb') as f:
            self.clf = pickle.load(f)
        with open(vectorizer_path, 'rb') as f:
            self.vectorizer = pickle.load(f)
        print("[+] Standby initialization sequence complete.")

    def evaluate_input_paragraph(self, raw_text):
        """Processes a single pasted paragraph and returns P(AI) and Confidence."""
        if len(raw_text.strip().split()) < 5:
            return 0.0, 100.0, "Low", "Input string too short for analytical scoring."
            
        # A. Compute fast curvature metric via single token forward pass
        tokens = self.tokenizer(raw_text, return_tensors="pt").input_ids.to(self.device)
        with torch.no_grad():
            outputs = self.model(tokens)
            logits = outputs.logits[:, :-1, :]
            target_ids = tokens[:, 1:]
            
            log_probs = F.log_softmax(logits, dim=-1)
            actual_log_probs = torch.gather(log_probs, 2, target_ids.unsqueeze(-1)).squeeze(-1)
            probs = F.softmax(logits, dim=-1)
            expected_log_probs = torch.sum(probs * log_probs, dim=-1)
            
            curvature = (actual_log_probs - expected_log_probs).mean().item()
            
        # B. Extract structural stylistics and text vector mapping
        style_feats = extract_stylistic_features([raw_text])
        tfidf_feats = self.vectorizer.transform([raw_text]).toarray()
        combined_feats = np.hstack([tfidf_feats, style_feats])
        
        # C. Generate predictions via deserialized classification matrix
        matrix_prob = self.clf.predict_proba(combined_feats)[0, 1]
        k = 4.0
        tau_0 = 0.12
        curvature_prob = 1.0 / (1.0 + np.exp(-k * (curvature - tau_0)))
        
        # Combined dynamic blend calculation
        final_p_ai = 0.70 * matrix_prob + 0.30 * curvature_prob
        
        # D. Quantify Associated Confidence Level
        u_boundary = 1.0 - 2.0 * abs(final_p_ai - 0.50) # Proximity to threshold
        margin = abs(self.clf.decision_function(combined_feats)[0])
        u_domain = 1.0 / (1.0 + margin)                 # Distance from known style domains
        
        u_total = 0.40 * u_boundary + 0.60 * u_domain
        confidence_pct = (1.0 - u_total) * 100.0
        
        tier = "High" if confidence_pct >= 85 else ("Moderate" if confidence_pct >= 65 else "Low")
        return round(final_p_ai, 4), round(confidence_pct, 2), tier, None

# Cache detector instantiation so model doesn't reload on every webpage click
@st.cache_resource
def get_detector_app():
    #target_docx_name = "MoSEs_dataset_filtered_train_main_1000_15Jul26.docx"
    target_docx_name = "MoSEs_dataset_tiny_gpt4_200_15Jul26.docx"

    target_docx = os.path.join(BASE_DIR, target_docx_name)
    
    if not os.path.exists(CLASSIFIER_PATH):
        if os.path.exists(target_docx):
            build_and_serialize_matrix(target_docx)
        elif os.path.exists(target_docx_name):
            build_and_serialize_matrix(target_docx_name)
        else:
            st.error(f"Training document '{target_docx_name}' not found, and no pre-trained matrix ('{CLASSIFIER_PATH}') exists.")
            st.stop()
    return RealTimeAIDetectorApp(classifier_path=CLASSIFIER_PATH, vectorizer_path=VECTORIZER_PATH)


# =====================================================================
# PDF REPORT GENERATION UTILITY (MULTI-PAGE LOGGING)
# =====================================================================

def generate_cumulative_pdf(records):
    """Compiles all execution records into a multi-page PDF document buffer."""
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(
        buffer,
        pagesize=letter,
        leftMargin=36,
        rightMargin=36,
        topMargin=36,
        bottomMargin=36
    )
    
    styles = getSampleStyleSheet()
    title_style = ParagraphStyle(
        'DocTitle',
        parent=styles['Heading1'],
        fontSize=16,
        leading=20,
        textColor=colors.HexColor("#003366"),
        spaceAfter=10
    )
    h2_style = ParagraphStyle(
        'DocH2',
        parent=styles['Heading2'],
        fontSize=12,
        leading=15,
        textColor=colors.HexColor("#002244"),
        spaceBefore=8,
        spaceAfter=4
    )
    meta_style = ParagraphStyle(
        'MetaText',
        parent=styles['Normal'],
        fontSize=9,
        leading=12,
        textColor=colors.HexColor("#333333")
    )
    body_style = ParagraphStyle(
        'BodyTextCustom',
        parent=styles['Normal'],
        fontSize=9,
        leading=12,
        textColor=colors.black
    )
    
    story = []
    
    for idx, rec in enumerate(records):
        story.append(Paragraph(f"ProofPixel Authentication Analysis Report (Entry #{idx + 1})", title_style))
        story.append(Paragraph(f"<b>Timestamp:</b> {rec['timestamp']}", meta_style))
        story.append(Paragraph(f"<b>Input Source Mode:</b> {rec['input_mode']}", meta_style))
        story.append(Paragraph(f"<b>File / Source Location:</b> {rec['source_location']}", meta_style))
        story.append(Spacer(1, 10))
        
        # Results Table Box
        res_data = [
            [Paragraph("<b>Metric</b>", meta_style), Paragraph("<b>Analysis Output</b>", meta_style)],
            [Paragraph("<b>Predicted Classification</b>", meta_style), Paragraph(f"<b>{rec['classification']}</b>", meta_style)],
            [Paragraph("<b>AI-Generated Probability P(AI)</b>", meta_style), Paragraph(f"{rec['p_ai'] * 100:.2f}%", meta_style)],
            [Paragraph("<b>Confidence Tier</b>", meta_style), Paragraph(f"{rec['confidence']:.2f}% ({rec['tier']})", meta_style)]
        ]
        
        bg_color = colors.HexColor("#FFEEEE") if "AI" in rec['classification'] else colors.HexColor("#EEFFEE")
        res_table = Table(res_data, colWidths=[200, 340])
        res_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor("#003366")),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
            ('BACKGROUND', (0, 1), (-1, -1), bg_color),
            ('GRID', (0, 0), (-1, -1), 0.5, colors.grey),
            ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
            ('TOPPADDING', (0, 0), (-1, -1), 4),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 4),
        ]))
        story.append(res_table)
        story.append(Spacer(1, 12))
        
        # Input Text Content Box
        story.append(Paragraph("Evaluated Input Paragraph Content:", h2_style))
        escaped_input = rec['input_text'].replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;')
        input_table = Table([[Paragraph(escaped_input, body_style)]], colWidths=[540])
        input_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, -1), colors.HexColor("#F8F9FA")),
            ('BOX', (0, 0), (-1, -1), 0.5, colors.HexColor("#CCCCCC")),
            ('TOPPADDING', (0, 0), (-1, -1), 6),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 6),
            ('LEFTPADDING', (0, 0), (-1, -1), 6),
            ('RIGHTPADDING', (0, 0), (-1, -1), 6),
        ]))
        story.append(input_table)
        
        # Subsequent runs are saved on separate pages
        if idx < len(records) - 1:
            story.append(PageBreak())
            
    doc.build(story)
    buffer.seek(0)
    return buffer.getvalue()


def launch_chrome_web_interface():
    # Resolve absolute path to image relative to BASE_DIR
    crest_logo_filename = "Proof4Pixels_The Authentication Crest_logo_brand_tagline_TM_17Aug26.png"
    crest_logo_img = os.path.join(BASE_DIR, crest_logo_filename)
    if not os.path.exists(crest_logo_img):
        crest_logo_img = crest_logo_filename

    # Configure Streamlit page
    st.set_page_config(
        page_title="Real-Time AI Text Detection", 
        page_icon=crest_logo_img if os.path.exists(crest_logo_img) else "🤖", 
        layout="wide"
    )

    # Initialize session state for multi-page execution logging
    if 'history_records' not in st.session_state:
        st.session_state.history_records = []
    if 'last_result' not in st.session_state:
        st.session_state.last_result = None

    # Custom CSS for Dark Blue Styling (Buttons, Download Buttons, Tabs, Radio Buttons)
    st.markdown("""
        <style>
        .block-container {
            max-width: 96% !important;
            padding-left: 2rem !important;
            padding-right: 2rem !important;
            padding-top: 2rem !important;
        }
        .single-line-header {
            font-size: clamp(1.15rem, 1.75vw, 2.05rem);
            font-weight: 700;
            line-height: 1.25;
            white-space: nowrap;
            overflow: hidden;
            text-overflow: ellipsis;
            margin: 0;
            padding: 0;
        }
        div[role="radiogroup"] label div[role="radio"][aria-checked="true"] > div {
            background-color: #003366 !important;
            border-color: #003366 !important;
        }
        div[role="radiogroup"] label div[role="radio"]:hover {
            border-color: #003366 !important;
        }
        div[role="radiogroup"] label div[role="radio"] input:checked ~ div {
            background-color: #003366 !important;
            border-color: #003366 !important;
        }
        button[data-baseweb="tab"][aria-selected="true"] {
            color: #003366 !important;
            border-bottom-color: #003366 !important;
        }
        button[data-baseweb="tab"] p {
            font-weight: 600;
        }
        
        /* Analysis and Download Action Buttons styled explicitly to Dark Blue */
        div.stButton > button, 
        div.stDownloadButton > button,
        div.stDownloadButton > button[kind="primary"],
        div.stButton > button[kind="primary"] {
            background-color: #003366 !important;
            color: #FFFFFF !important;
            border-color: #002244 !important;
            font-weight: 600;
        }
        div.stButton > button:hover, 
        div.stDownloadButton > button:hover,
        div.stDownloadButton > button[kind="primary"]:hover,
        div.stButton > button[kind="primary"]:hover {
            background-color: #002244 !important;
            border-color: #001122 !important;
            color: #FFFFFF !important;
        }
        </style>
    """, unsafe_allow_html=True)

    # Header Row
    col_logo, col_title = st.columns([0.06, 0.94], vertical_alignment="center")
    with col_logo:
        if os.path.exists(crest_logo_img):
            st.image(crest_logo_img, width=70)
        else:
            st.markdown("<h2 style='margin:0;'>🤖</h2>", unsafe_allow_html=True)
    with col_title:
        st.markdown('<h1 class="single-line-header">Proof4Pixels: A real time platform for distinguishing AI-generated vs. Human-authored text, images, videos</h1>', unsafe_allow_html=True)

    st.markdown("Analyze text snippets or text document files to detect **AI vs. Human** authorship origin.")
    st.divider()

    # Load cached app model
    with st.spinner("Initializing Deep Curvature Models and Classification Matrix..."):
        app = get_detector_app()

    # Input method selection
    choice = st.radio(
        "Select Input Method:",
        ("Option A: Manually copy-paste paragraph", "Option B: Read from an uploaded file/ or text file path"),
        index=0
    )

    user_input = ""
    source_location = ""

    if "Option A" in choice:
        user_input = st.text_area("Paste Text Paragraph Here:", height=180, placeholder="Type or paste paragraph text to analyze...")
        source_location = "Direct Manual Input (Pasted Text Buffer)"
    else:
        tab1, tab2 = st.tabs(["📁 File Uploader", "🖥️ File System Path"])
        
        with tab1:
            uploaded_file = st.file_uploader("Upload a .txt file:", type=["txt"])
            if uploaded_file is not None:
                user_input = uploaded_file.read().decode("utf-8")
                source_location = f"Uploaded File: {uploaded_file.name} (Memory Buffer)"
                
        with tab2:
            file_path = st.text_input("Enter absolute file path to .txt file on your PC:", placeholder=r"C:\Users\username\Desktop\document.txt")
            if file_path:
                source_location = os.path.abspath(file_path.strip())
                try:
                    with open(file_path.strip(), 'r', encoding='utf-8') as f:
                        user_input = f.read()
                except FileNotFoundError:
                    st.error(f"File not found at path: `{file_path}`")
                except Exception as e:
                    st.error(f"Error reading file: {e}")

    # Analyze Action Button
    if st.button("Run Source of Origin Detection Analysis", type="primary"):
        if not user_input or not user_input.strip():
            st.warning("Empty input block. Please provide text content to analyze.")
        else:
            with st.spinner("Evaluating source of origin detection and calculating probabilities..."):
                p_ai, conf, tier, err = app.evaluate_input_paragraph(user_input)

            if err:
                st.error(f"Evaluation Error: {err}")
            else:
                is_ai = p_ai >= 0.5
                classification_label = "AI-Generated origin" if is_ai else "Human-Authored origin"
                
                # Store latest result in session state
                current_record = {
                    "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "input_mode": choice,
                    "source_location": source_location,
                    "input_text": user_input,
                    "classification": classification_label,
                    "p_ai": p_ai,
                    "confidence": conf,
                    "tier": tier
                }
                st.session_state.history_records.append(current_record)
                st.session_state.last_result = current_record

    # Render Current Result if available
    if st.session_state.last_result is not None:
        rec = st.session_state.last_result
        st.subheader("--- Source of Origin Detection Analysis Results ---")
        
        is_ai = rec['p_ai'] >= 0.5
        if is_ai:
            st.error(f"**Predicted Classification:** {rec['classification']}")
        else:
            st.success(f"**Predicted Classification:** {rec['classification']}")

        col1, col2 = st.columns(2)
        col1.metric(label="AI-Generated Probability P(AI)", value=f"{rec['p_ai'] * 100:.2f}%")
        col2.metric(label="Confidence Tier", value=f"{rec['confidence']:.2f}% ({rec['tier']})")

        st.write("**AI-Generated Probability Gauge:**")
        st.progress(float(rec['p_ai']))
        st.caption(f"Logged {len(st.session_state.history_records)} analysis session page(s) in PDF report.")

    # =====================================================================
    # DOWNLOAD RESULTS SECTION
    # =====================================================================
    if len(st.session_state.history_records) > 0:
        st.markdown("---")
        pdf_bytes = generate_cumulative_pdf(st.session_state.history_records)
        
        col_dl, col_spacer = st.columns([0.30, 0.70])
        with col_dl:
            # =================================================================
            # MODIFIED ON 15Aug26: Changed button label from "Close and download results"
            # to "Download results" and applied Dark Blue styling.
            # =================================================================
            # if st.download_button(
            #     label="📥 Close and download results",
            #     data=pdf_bytes,
            #     file_name=f"ProofPixel_Detection_Report_{datetime.datetime.now().strftime('%d%b%y_%H%M%S')}.pdf",
            #     mime="application/pdf",
            #     type="primary",
            #     key="close_download_btn"
            # ):
            #     st.markdown("""<script>window.parent.window.close(); window.close();</script>""", unsafe_allow_html=True)
            
            st.download_button(
                label="📥 Download results",
                data=pdf_bytes,
                file_name=f"ProofPixel_Detection_Report_{datetime.datetime.now().strftime('%d%b%y_%H%M%S')}.pdf",
                mime="application/pdf",
                type="primary",
                key="download_results_btn"
            )


# =====================================================================
# UNIFIED EXECUTION ENTRY POINT
# =====================================================================

if __name__ == '__main__':
    import sys
    import subprocess

    # Check if the script is being executed inside an active Streamlit server context
    if st.runtime.exists():
        launch_chrome_web_interface()
    else:
        print("[*] Launching Streamlit Web App in Google Chrome...")
        cmd = [sys.executable, "-m", "streamlit", "run", sys.argv[0]]
        subprocess.run(cmd)