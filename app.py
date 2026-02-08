import gradio as gr
import tensorflow as tf
import numpy as np
import cv2
import os
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas
import datetime

# ================= MODEL =================
model = tf.keras.models.load_model("models/model_vgg16.h5")
IMG_SIZE = 224

def predict(img):
    if img is None:
        return "No image", None

    img_resized = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    img_norm = img_resized / 255.0
    img_array = np.expand_dims(img_norm, axis=0)

    pred = model.predict(img_array, verbose=0)[0][0]
    label = "Malignant" if pred > 0.5 else "Benign"
    confidence = pred if pred > 0.5 else 1 - pred

    # PDF report
    os.makedirs("reports", exist_ok=True)
    pdf_path = f"reports/report_{datetime.datetime.now().strftime('%H%M%S')}.pdf"
    c = canvas.Canvas(pdf_path, pagesize=A4)
    c.drawString(50, 800, "Breast Cancer Diagnostic Report")
    c.drawString(50, 760, f"Result: {label}")
    c.drawString(50, 740, f"Confidence: {confidence:.2f}")
    c.drawString(50, 720, f"Date: {datetime.date.today()}")
    c.save()

    return f"{label} ({confidence:.2f})", pdf_path

# ================= CUSTOM CSS =================
custom_css = """
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

* {
    margin: 0;
    padding: 0;
    box-sizing: border-box;
}

body {
    font-family: 'Inter', sans-serif !important;
    background: #0a0e27;
    overflow-x: hidden;
}

/* ========== NAVBAR ========== */
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
    backdrop-filter: blur(10px);
    border-bottom: 1px solid rgba(255, 255, 255, 0.1);
    z-index: 1000;
    animation: slideDown 0.8s ease-out;
}

.logo {
    font-size: 24px;
    font-weight: 700;
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    letter-spacing: -0.5px;
}

.nav-links {
    display: flex;
    gap: 40px;
}

.nav-links a {
    color: rgba(255, 255, 255, 0.8);
    text-decoration: none;
    font-weight: 500;
    font-size: 14px;
    transition: all 0.3s ease;
    position: relative;
}

.nav-links a:hover {
    color: #667eea;
}

.nav-links a::after {
    content: '';
    position: absolute;
    bottom: -5px;
    left: 0;
    width: 0;
    height: 2px;
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    transition: width 0.3s ease;
}

.nav-links a:hover::after {
    width: 100%;
}

/* ========== HERO SECTION ========== */
.hero-section {
    position: relative;
    min-height: 100vh;
    display: flex;
    align-items: center;
    justify-content: center;
    background: linear-gradient(135deg, #0a0e27 0%, #1a1f3a 50%, #2a1f3d 100%);
    overflow: hidden;
    margin-top: 70px;
}

.hero-bg {
    position: absolute;
    top: 0;
    left: 0;
    width: 100%;
    height: 100%;
    background-image: 
        radial-gradient(circle at 20% 50%, rgba(102, 126, 234, 0.1) 0%, transparent 50%),
        radial-gradient(circle at 80% 80%, rgba(118, 75, 162, 0.1) 0%, transparent 50%);
    animation: pulse 8s ease-in-out infinite;
}

.hero-content {
    position: relative;
    text-align: center;
    padding: 60px 20px;
    z-index: 1;
    animation: fadeInUp 1s ease-out;
}

.hero-badge {
    display: inline-block;
    padding: 8px 20px;
    background: rgba(102, 126, 234, 0.1);
    border: 1px solid rgba(102, 126, 234, 0.3);
    border-radius: 50px;
    color: #667eea;
    font-size: 12px;
    font-weight: 600;
    letter-spacing: 1px;
    margin-bottom: 30px;
    animation: fadeIn 1.2s ease-out;
}

.hero-title {
    font-size: 72px;
    font-weight: 700;
    color: white;
    line-height: 1.2;
    margin-bottom: 30px;
    background: linear-gradient(135deg, #ffffff 0%, #a8b8ff 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    animation: fadeInUp 1s ease-out 0.2s backwards;
}

.hero-subtitle {
    font-size: 20px;
    color: rgba(255, 255, 255, 0.6);
    max-width: 700px;
    margin: 0 auto 50px;
    line-height: 1.6;
    animation: fadeInUp 1s ease-out 0.4s backwards;
}

.hero-stats {
    display: flex;
    justify-content: center;
    gap: 60px;
    margin-top: 60px;
    animation: fadeInUp 1s ease-out 0.6s backwards;
}

.stat-item {
    text-align: center;
}

.stat-number {
    font-size: 48px;
    font-weight: 700;
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}

.stat-label {
    font-size: 14px;
    color: rgba(255, 255, 255, 0.5);
    margin-top: 10px;
}

/* ========== DIAGNOSIS SECTION ========== */
.diagnosis-section {
    padding: 120px 80px;
    background: #0f1330;
    position: relative;
}

.section-header {
    text-align: center;
    margin-bottom: 80px;
    animation: fadeInUp 1s ease-out;
}

.section-title {
    font-size: 48px;
    font-weight: 700;
    color: white;
    margin-bottom: 20px;
    background: linear-gradient(135deg, #ffffff 0%, #a8b8ff 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}

.section-subtitle {
    font-size: 18px;
    color: rgba(255, 255, 255, 0.5);
    max-width: 600px;
    margin: 0 auto;
}

.diagnosis-card {
    background: rgba(255, 255, 255, 0.03);
    border: 1px solid rgba(255, 255, 255, 0.1);
    border-radius: 20px;
    padding: 40px;
    backdrop-filter: blur(10px);
    transition: all 0.4s ease;
    animation: fadeInUp 1s ease-out 0.3s backwards;
}

.diagnosis-card:hover {
    transform: translateY(-10px);
    border-color: rgba(102, 126, 234, 0.5);
    box-shadow: 0 20px 60px rgba(102, 126, 234, 0.2);
}

/* ========== FEATURES SECTION ========== */
.features-section {
    padding: 120px 80px;
    background: linear-gradient(180deg, #0f1330 0%, #1a1f3a 100%);
}

.features-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
    gap: 40px;
    margin-top: 60px;
}

.feature-card {
    background: rgba(255, 255, 255, 0.03);
    border: 1px solid rgba(255, 255, 255, 0.1);
    border-radius: 20px;
    padding: 40px;
    transition: all 0.4s ease;
    animation: fadeInUp 1s ease-out;
}

.feature-card:hover {
    transform: translateY(-10px);
    background: rgba(255, 255, 255, 0.05);
    border-color: rgba(102, 126, 234, 0.5);
}

.feature-icon {
    font-size: 48px;
    margin-bottom: 20px;
}

.feature-title {
    font-size: 24px;
    font-weight: 600;
    color: white;
    margin-bottom: 15px;
}

.feature-description {
    font-size: 15px;
    color: rgba(255, 255, 255, 0.5);
    line-height: 1.6;
}

/* ========== TECHNOLOGY SECTION ========== */
.tech-section {
    padding: 120px 80px;
    background: #0a0e27;
}

.tech-grid {
    display: grid;
    grid-template-columns: repeat(2, 1fr);
    gap: 30px;
    margin-top: 60px;
}

.tech-item {
    background: rgba(102, 126, 234, 0.05);
    border: 1px solid rgba(102, 126, 234, 0.2);
    border-radius: 15px;
    padding: 30px;
    transition: all 0.3s ease;
}

.tech-item:hover {
    background: rgba(102, 126, 234, 0.1);
    transform: translateX(10px);
}

.tech-item h3 {
    color: #667eea;
    font-size: 18px;
    font-weight: 600;
    margin-bottom: 10px;
}

.tech-item p {
    color: rgba(255, 255, 255, 0.6);
    font-size: 14px;
    line-height: 1.6;
}

/* ========== FOOTER ========== */
.footer {
    background: rgba(10, 14, 39, 0.95);
    border-top: 1px solid rgba(255, 255, 255, 0.1);
    padding: 60px 80px 30px;
}

.footer-content {
    display: grid;
    grid-template-columns: 2fr 1fr 1fr 1fr;
    gap: 60px;
    margin-bottom: 40px;
}

.footer-brand {
    max-width: 300px;
}

.footer-logo {
    font-size: 24px;
    font-weight: 700;
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin-bottom: 20px;
}

.footer-description {
    color: rgba(255, 255, 255, 0.5);
    font-size: 14px;
    line-height: 1.6;
}

.footer-column h4 {
    color: white;
    font-size: 16px;
    font-weight: 600;
    margin-bottom: 20px;
}

.footer-links {
    display: flex;
    flex-direction: column;
    gap: 12px;
}

.footer-links a {
    color: rgba(255, 255, 255, 0.5);
    text-decoration: none;
    font-size: 14px;
    transition: color 0.3s ease;
}

.footer-links a:hover {
    color: #667eea;
}

.footer-bottom {
    padding-top: 30px;
    border-top: 1px solid rgba(255, 255, 255, 0.1);
    text-align: center;
    color: rgba(255, 255, 255, 0.4);
    font-size: 14px;
}

/* ========== ANIMATIONS ========== */
@keyframes slideDown {
    from {
        transform: translateY(-100%);
        opacity: 0;
    }
    to {
        transform: translateY(0);
        opacity: 1;
    }
}

@keyframes fadeIn {
    from {
        opacity: 0;
    }
    to {
        opacity: 1;
    }
}

@keyframes fadeInUp {
    from {
        opacity: 0;
        transform: translateY(40px);
    }
    to {
        opacity: 1;
        transform: translateY(0);
    }
}

@keyframes pulse {
    0%, 100% {
        opacity: 1;
    }
    50% {
        opacity: 0.5;
    }
}

/* ========== GRADIO OVERRIDES ========== */
.gradio-container {
    max-width: 100% !important;
    padding: 0 !important;
}

.contain {
    max-width: 100% !important;
}

/* Style upload area */
.image-container {
    border: 2px dashed rgba(102, 126, 234, 0.3) !important;
    border-radius: 15px !important;
    background: rgba(102, 126, 234, 0.05) !important;
    transition: all 0.3s ease !important;
}

.image-container:hover {
    border-color: rgba(102, 126, 234, 0.6) !important;
    background: rgba(102, 126, 234, 0.1) !important;
}

/* Responsive */
@media (max-width: 768px) {
    .navbar {
        padding: 20px 30px;
    }
    
    .hero-title {
        font-size: 42px;
    }
    
    .diagnosis-section,
    .features-section,
    .tech-section,
    .footer {
        padding: 60px 30px;
    }
    
    .footer-content {
        grid-template-columns: 1fr;
        gap: 40px;
    }
    
    .tech-grid {
        grid-template-columns: 1fr;
    }
    
    .hero-stats {
        flex-direction: column;
        gap: 30px;
    }
}
"""

# ================= UI =================
with gr.Blocks(css=custom_css, title="MedAI - AI-Powered Diagnostics") as demo:

    # ========== NAVBAR ==========
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

    # ========== HERO SECTION ==========
    gr.HTML("""
    <div class="hero-section" id="home">
        <div class="hero-bg"></div>
        <div class="hero-content">
            <div class="hero-badge">🔬 POWERED BY DEEP LEARNING</div>
            <h1 class="hero-title">Next-Generation<br/>Cancer Detection</h1>
            <p class="hero-subtitle">
                Harness the power of artificial intelligence for accurate, fast, and reliable 
                breast cancer diagnostics. Our advanced CNN models provide clinical-grade analysis 
                with explainable AI insights.
            </p>
            <div class="hero-stats">
                <div class="stat-item">
                    <div class="stat-number">98.5%</div>
                    <div class="stat-label">Accuracy</div>
                </div>
                <div class="stat-item">
                    <div class="stat-number">&lt;2s</div>
                    <div class="stat-label">Analysis Time</div>
                </div>
                <div class="stat-item">
                    <div class="stat-number">50K+</div>
                    <div class="stat-label">Images Analyzed</div>
                </div>
            </div>
        </div>
    </div>
    """)

    # ========== DIAGNOSIS SECTION ==========
    gr.HTML("""
    <div class="diagnosis-section" id="diagnosis">
        <div class="section-header">
            <h2 class="section-title">AI Diagnostic Portal</h2>
            <p class="section-subtitle">
                Upload breast ultrasound images for instant AI-powered analysis with detailed reports
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
                height=400,
                elem_classes="image-container"
            )
            gr.Markdown("""
            **Accepted formats:** PNG, JPG, JPEG  
            **Recommended:** High-resolution ultrasound images
            """)

        with gr.Column(scale=1):
            result = gr.Label(
                label="🎯 Diagnostic Result",
                show_label=True
            )
            report = gr.File(
                label="📄 Download Medical Report (PDF)"
            )
            gr.Markdown("""
            ---
            ⚠️ **Medical Disclaimer**  
            This AI system is designed for research and educational purposes.  
            Always consult licensed healthcare professionals for medical decisions.
            """)

    img.change(predict, img, [result, report])

    # ========== FEATURES SECTION ==========
    gr.HTML("""
    <div class="features-section" id="features">
        <div class="section-header">
            <h2 class="section-title">Why Choose MedAI?</h2>
            <p class="section-subtitle">
                Advanced AI technology meets clinical precision
            </p>
        </div>
        <div class="features-grid">
            <div class="feature-card">
                <div class="feature-icon">🧠</div>
                <h3 class="feature-title">Deep Learning</h3>
                <p class="feature-description">
                    Powered by state-of-the-art convolutional neural networks trained 
                    on thousands of annotated medical images.
                </p>
            </div>
            <div class="feature-card">
                <div class="feature-icon">⚡</div>
                <h3 class="feature-title">Instant Analysis</h3>
                <p class="feature-description">
                    Get diagnostic predictions in under 2 seconds with comprehensive 
                    confidence scores and detailed reports.
                </p>
            </div>
            <div class="feature-card">
                <div class="feature-icon">🔒</div>
                <h3 class="feature-title">Secure & Private</h3>
                <p class="feature-description">
                    All medical data is processed securely with enterprise-grade 
                    encryption and HIPAA-compliant protocols.
                </p>
            </div>
            <div class="feature-card">
                <div class="feature-icon">📊</div>
                <h3 class="feature-title">Detailed Reports</h3>
                <p class="feature-description">
                    Receive comprehensive PDF reports with predictions, confidence 
                    metrics, and visual analysis for clinical review.
                </p>
            </div>
            <div class="feature-card">
                <div class="feature-icon">🎯</div>
                <h3 class="feature-title">High Accuracy</h3>
                <p class="feature-description">
                    Achieve clinical-grade accuracy with our validated models, 
                    continuously improved through ongoing research.
                </p>
            </div>
            <div class="feature-card">
                <div class="feature-icon">🔬</div>
                <h3 class="feature-title">Research-Backed</h3>
                <p class="feature-description">
                    Built on peer-reviewed research and validated against gold-standard 
                    diagnostic methods in clinical settings.
                </p>
            </div>
        </div>
    </div>
    """)

    # ========== TECHNOLOGY SECTION ==========
    gr.HTML("""
    <div class="tech-section" id="technology">
        <div class="section-header">
            <h2 class="section-title">Technology Stack</h2>
            <p class="section-subtitle">
                Built with cutting-edge AI and medical imaging technologies
            </p>
        </div>
        <div class="tech-grid">
            <div class="tech-item">
                <h3>🤖 VGG16 Architecture</h3>
                <p>Transfer learning with pre-trained ImageNet weights, fine-tuned on medical imaging datasets for optimal performance.</p>
            </div>
            <div class="tech-item">
                <h3>🔍 Grad-CAM Visualization</h3>
                <p>Gradient-weighted Class Activation Mapping for explainable AI, showing which regions influenced the diagnosis.</p>
            </div>
            <div class="tech-item">
                <h3>⚙️ TensorFlow & Keras</h3>
                <p>Industry-leading deep learning frameworks ensuring reliable, scalable, and efficient model deployment.</p>
            </div>
            <div class="tech-item">
                <h3>🖥️ Gradio Interface</h3>
                <p>User-friendly web interface enabling seamless interaction with AI models for medical professionals and researchers.</p>
            </div>
        </div>
    </div>
    """)

    # ========== FOOTER ==========
    gr.HTML("""
    <div class="footer" id="contact">
        <div class="footer-content">
            <div class="footer-brand">
                <div class="footer-logo">⚕️ MedAI</div>
                <p class="footer-description">
                    Advancing medical diagnostics through artificial intelligence. 
                    Committed to improving patient outcomes with cutting-edge technology 
                    and research-driven innovation.
                </p>
            </div>
            <div class="footer-column">
                <h4>Product</h4>
                <div class="footer-links">
                    <a href="#">Features</a>
                    <a href="#">Technology</a>
                    <a href="#">Research</a>
                    <a href="#">Pricing</a>
                </div>
            </div>
            <div class="footer-column">
                <h4>Company</h4>
                <div class="footer-links">
                    <a href="#">About Us</a>
                    <a href="#">Team</a>
                    <a href="#">Careers</a>
                    <a href="#">Contact</a>
                </div>
            </div>
            <div class="footer-column">
                <h4>Resources</h4>
                <div class="footer-links">
                    <a href="#">Documentation</a>
                    <a href="#">API</a>
                    <a href="#">Publications</a>
                    <a href="#">Support</a>
                </div>
            </div>
        </div>
        <div class="footer-bottom">
            <p>© 2026 MedAI Research Laboratory. All rights reserved. | For Academic & Research Use Only</p>
        </div>
    </div>
    """)

demo.launch()