import gradio as gr
import tensorflow as tf
import numpy as np
import cv2
import os
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas
import datetime

# ================= MODEL =================
IMG_SIZE = 224
model = tf.keras.models.load_model("models/model_vgg16.h5")


def predict(img):
    if img is None:
        return "No image uploaded", None

    img_resized = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    img_norm = img_resized / 255.0
    img_array = np.expand_dims(img_norm, axis=0)

    pred = model.predict(img_array, verbose=0)[0][0]
    label = "Malignant" if pred > 0.5 else "Benign"
    confidence = pred if pred > 0.5 else 1 - pred

    os.makedirs("reports", exist_ok=True)
    pdf_path = f"reports/report_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"

    c = canvas.Canvas(pdf_path, pagesize=A4)
    c.setFont("Helvetica-Bold", 18)
    c.drawString(50, 800, "Breast Cancer Diagnostic Report")
    c.setFont("Helvetica", 14)
    c.drawString(50, 760, f"Result: {label}")
    c.drawString(50, 740, f"Confidence: {confidence:.2f}")
    c.drawString(50, 720, f"Date: {datetime.date.today()}")
    c.save()

    return f"{label} ({confidence:.2f})", pdf_path


custom_css = """
body {
    background: #0a0e27;
    font-family: Arial, sans-serif;
}

.gradio-container {
    max-width: 100% !important;
    padding: 0 !important;
}

.navbar {
    position: fixed;
    top: 0;
    left: 0;
    right: 0;
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 20px 80px;
    background: rgba(10, 14, 39, 0.95);
    border-bottom: 1px solid rgba(255,255,255,0.1);
    z-index: 1000;
}

.logo {
    font-size: 24px;
    font-weight: 700;
    color: #8ea2ff;
}

.nav-links {
    display: flex;
    gap: 35px;
}

.nav-links a {
    color: white;
    text-decoration: none;
    font-size: 14px;
}

.nav-links a:hover {
    color: #8ea2ff;
}

.hero-section {
    min-height: 90vh;
    padding-top: 90px;
    display: flex;
    align-items: center;
    justify-content: center;
    background: linear-gradient(135deg, #0a0e27, #1a1f3a, #2a1f3d);
    text-align: center;
}

.hero-title {
    font-size: 68px;
    font-weight: 700;
    color: white;
    line-height: 1.2;
}

.hero-subtitle {
    font-size: 20px;
    color: rgba(255,255,255,0.65);
    max-width: 750px;
    margin: 25px auto;
    line-height: 1.6;
}

.hero-badge {
    display: inline-block;
    padding: 8px 20px;
    border-radius: 30px;
    color: #8ea2ff;
    border: 1px solid rgba(142,162,255,0.4);
    margin-bottom: 25px;
}

.hero-stats {
    display: flex;
    justify-content: center;
    gap: 60px;
    margin-top: 50px;
}

.stat-number {
    font-size: 42px;
    font-weight: bold;
    color: #8ea2ff;
}

.stat-label {
    color: rgba(255,255,255,0.6);
}

.diagnosis-section,
.features-section,
.tech-section {
    padding: 90px 80px;
    background: #0f1330;
}

.section-header {
    text-align: center;
    margin-bottom: 50px;
}

.section-title {
    font-size: 46px;
    color: white;
}

.section-subtitle {
    color: rgba(255,255,255,0.6);
    font-size: 18px;
}

.diagnosis-card {
    max-width: 1100px;
    margin: auto;
    padding: 35px;
    border-radius: 20px;
    background: rgba(255,255,255,0.04);
    border: 1px solid rgba(255,255,255,0.1);
}

.features-grid,
.tech-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(260px, 1fr));
    gap: 30px;
}

.feature-card,
.tech-item {
    padding: 30px;
    border-radius: 18px;
    background: rgba(255,255,255,0.04);
    border: 1px solid rgba(255,255,255,0.1);
    color: white;
}

.feature-icon {
    font-size: 42px;
}

.feature-description,
.tech-item p {
    color: rgba(255,255,255,0.6);
    line-height: 1.6;
}

.footer {
    background: #0a0e27;
    color: rgba(255,255,255,0.6);
    text-align: center;
    padding: 35px;
}

@media (max-width: 768px) {
    .navbar {
        padding: 15px 25px;
    }

    .nav-links {
        display: none;
    }

    .hero-title {
        font-size: 42px;
    }

    .hero-stats {
        flex-direction: column;
        gap: 25px;
    }

    .diagnosis-section,
    .features-section,
    .tech-section {
        padding: 60px 25px;
    }
}
"""


with gr.Blocks(css=custom_css, title="MedAI - AI-Powered Diagnostics") as demo:

    gr.HTML("""
    <div class="navbar">
        <div class="logo">⚕️ MedAI</div>
        <div class="nav-links">
            <a href="#home">Home</a>
            <a href="#diagnosis">Diagnosis</a>
            <a href="#features">Features</a>
            <a href="#technology">Technology</a>
            <a href="#contact">Contact</a>
        </div>
    </div>
    """)

    gr.HTML("""
    <div class="hero-section" id="home">
        <div>
            <div class="hero-badge">🔬 POWERED BY DEEP LEARNING</div>
            <h1 class="hero-title">Next-Generation<br>Cancer Detection</h1>
            <p class="hero-subtitle">
                Upload breast ultrasound images and get instant AI-powered diagnostic
                prediction with confidence score and PDF report.
            </p>

            <div class="hero-stats">
                <div>
                    <div class="stat-number">98.5%</div>
                    <div class="stat-label">Powered</div>
                </div>
                <div>
                    <div class="stat-number">&lt;2s</div>
                    <div class="stat-label">Analysis</div>
                </div>
                <div>
                    <div class="stat-number">50K+</div>
                    <div class="stat-label">Report</div>
                </div>
            </div>
        </div>
    </div>
    """)

    gr.HTML("""
    <div class="diagnosis-section" id="diagnosis">
        <div class="section-header">
            <h2 class="section-title">AI Diagnostic Portal</h2>
            <p class="section-subtitle">
                Upload an ultrasound image for AI-based analysis.
            </p>
        </div>
    </div>
    """)

    with gr.Row(elem_classes="diagnosis-card"):
        with gr.Column(scale=1):
            img = gr.Image(
                type="numpy",
                sources=["upload"],
                label="📤 Upload Ultrasound Image",
                height=400
            )

            gr.Markdown("""
            **Accepted formats:** PNG, JPG, JPEG  
            **Recommended:** High-resolution ultrasound images
            """)

        with gr.Column(scale=1):
            result = gr.Textbox(label="🎯 Diagnostic Result")

            report = gr.File(label="📄 Download Medical Report")

            btn = gr.Button("Analyze Image")

            gr.Markdown("""
            ---
            ⚠️ **Medical Disclaimer**  
            This AI system is for research and educational purposes only.  
            Always consult licensed healthcare professionals for medical decisions.
            """)

    btn.click(
        fn=predict,
        inputs=img,
        outputs=[result, report]
    )

    gr.HTML("""
    <div class="features-section" id="features">
        <div class="section-header">
            <h2 class="section-title">Why Choose MedAI?</h2>
            <p class="section-subtitle">Advanced AI technology meets clinical precision.</p>
        </div>

        <div class="features-grid">
            <div class="feature-card">
                <div class="feature-icon">🧠</div>
                <h3>Deep Learning</h3>
                <p class="feature-description">
                    Uses CNN-based transfer learning for medical image classification.
                </p>
            </div>

            <div class="feature-card">
                <div class="feature-icon">⚡</div>
                <h3>Fast Analysis</h3>
                <p class="feature-description">
                    Provides prediction result quickly after image upload.
                </p>
            </div>

            <div class="feature-card">
                <div class="feature-icon">📄</div>
                <h3>PDF Report</h3>
                <p class="feature-description">
                    Automatically generates a downloadable diagnostic report.
                </p>
            </div>

            <div class="feature-card">
                <div class="feature-icon">🔒</div>
                <h3>Local Processing</h3>
                <p class="feature-description">
                    Runs locally on your system using your trained model.
                </p>
            </div>
        </div>
    </div>
    """)

    gr.HTML("""
    <div class="tech-section" id="technology">
        <div class="section-header">
            <h2 class="section-title">Technology Stack</h2>
            <p class="section-subtitle">
                Built with Python, TensorFlow, OpenCV, ReportLab, and Gradio.
            </p>
        </div>

        <div class="tech-grid">
            <div class="tech-item">
                <h3>🤖 VGG16 Model</h3>
                <p>Transfer learning model trained for binary classification.</p>
            </div>

            <div class="tech-item">
                <h3>⚙️ TensorFlow</h3>
                <p>Used for loading and running the trained deep learning model.</p>
            </div>

            <div class="tech-item">
                <h3>🖼️ OpenCV</h3>
                <p>Used for resizing and preprocessing uploaded images.</p>
            </div>

            <div class="tech-item">
                <h3>🖥️ Gradio</h3>
                <p>Creates the local web interface for interaction.</p>
            </div>
        </div>
    </div>
    """)

    gr.HTML("""
    <div class="footer" id="contact">
        © 2026 MedAI Research Laboratory | For Academic & Research Use Only
    </div>
    """)


if __name__ == "__main__":
    demo.launch(server_name="127.0.0.1", server_port=7860)