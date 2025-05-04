# Deployment code will go here
import cv2
import torch
import torchvision.transforms as transforms
from torchvision import models
from PIL import Image
import os
import numpy as np
import torch.nn.functional as F
import streamlit as st
from huggingface_hub import login
from langchain_huggingface import HuggingFaceEndpoint

# -------------------------------
# UI Styling
# -------------------------------
st.set_page_config(page_title="IVF Embryo Grading", layout="wide")
st.markdown("""
    <style>
        body {
            background-color: #f5f9ff;
        }
        .main {
            background-color: #ffffff;
            padding: 2rem;
            border-radius: 10px;
            box-shadow: 0 4px 12px rgba(0, 0, 0, 0.1);
        }
        h1, h4 {
            color: #145DA0;
        }
        .stButton>button {
            background-color: #1f77b4;
            color: white;
            font-weight: bold;
            padding: 0.5rem 1rem;
            border-radius: 8px;
            margin-top: 0.5rem;
        }
        .stButton>button:hover {
            background-color: #125582;
        }
        .stImage>img {
            border-radius: 10px;
            box-shadow: 0 0 10px rgba(0,0,0,0.1);
            max-width: 150px;
            max-height: 150px;
            margin-bottom: 8px;
        }
        .uploadedFileName, .uploadedFileSize, .stFileUploader label + div {
            display: none !important;
        }
        .error-message {
            color: red;
            font-weight: bold;
            text-align: center;
            font-size: 14px;
            margin-top: 10px;
        }
        .implantation-recommendation {
            background-color: #e0f7fa;
            border-left: 5px solid #00796b;
            padding: 10px;
            margin-top: 20px;
            font-weight: bold;
        }
    </style>
""", unsafe_allow_html=True)




st.markdown("<h1 style='text-align: center; color: #145DA0;'>IVF Embryo Grading Dashboard</h1>", unsafe_allow_html=True)

# -------------------------------
# Patient Login Section (Left Sidebar)
# -------------------------------
import datetime
from sqlalchemy import create_engine, Column, String, Date, Integer, TIMESTAMP, VARCHAR
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
import streamlit as st

# SQLAlchemy setup
DB_URI = "mysql+pymysql://user3:user3@localhost/embryo_db"  # Replace with actual values
engine = create_engine(DB_URI)
Base = declarative_base()

class Patient(Base):
    __tablename__ = 'patients'
    patient_id = Column(VARCHAR(50), primary_key=True)
    name = Column(String(100))
    dob = Column(Date)
    age = Column(Integer)
    grade_day3 = Column(String(10))
    grade_day4 = Column(String(10))
    grade_day5 = Column(String(10))
    timestamp = Column(TIMESTAMP, server_default="CURRENT_TIMESTAMP")

Base.metadata.create_all(engine)
Session = sessionmaker(bind=engine)
session = Session()

# Sidebar Login
st.sidebar.header("🔐 Patient Login")
patient_id = st.sidebar.text_input("Patient ID")
patient_name = st.sidebar.text_input("Patient Name")
dob = st.sidebar.date_input("Date of Birth", min_value=datetime.date(1990, 1, 1), max_value=datetime.date.today())

logged_in = False
if st.sidebar.button("Login"):
    if patient_id and patient_name and dob:
        logged_in = True
        st.session_state['patient_logged_in'] = True
        st.session_state['patient_info'] = {
            'id': patient_id,
            'name': patient_name,
            'dob': dob,
            'age': (datetime.date.today() - dob).days // 365
        }
        st.sidebar.success("Logged in successfully.")
    else:
        st.sidebar.error("Please fill in all login fields.")

# Check login before allowing uploads
if not st.session_state.get('patient_logged_in'):
    st.warning("⚠️ Please log in from the sidebar to upload embryo images.")
    st.stop()

# -------------------------------
# Constants and Paths
# -------------------------------
class_names = ['Grade A', 'Grade B', 'Grade C']
MODEL_DIR = r'C:\Users\gopia\Downloads\Embryo_Quality_Prediction_project\models'  # Update as needed

# -------------------------------
# Load Models
# -------------------------------
def load_model_for_day(day):
    model_path = os.path.join(MODEL_DIR, f'Day_{day}_resnet_model.pth')
    model = models.resnet18(pretrained=False)
    model.fc = torch.nn.Linear(model.fc.in_features, 3)
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    model.eval()
    return model

def load_binary_model():
    model_path = os.path.join(MODEL_DIR, 'error_resnet_model.pth')
    model = models.resnet18(pretrained=False)
    model.fc = torch.nn.Linear(model.fc.in_features, 2)
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    model.eval()
    return model

# -------------------------------
# Image Preprocessing
# -------------------------------
def preprocess_image(image):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    return transform(image).unsqueeze(0)

def is_embryo_image(image):
    tensor = preprocess_image(image)
    with torch.no_grad():
        output = binary_model(tensor)
        pred = torch.argmax(output, 1).item()
    return pred == 0

# -------------------------------
# Grad-CAM Generation
# -------------------------------
def generate_cam(model, input_tensor, target_class):
    activations = []
    gradients = []

    def forward_hook(module, input, output):
        activations.append(output)
        output.register_hook(lambda grad: gradients.append(grad))

    final_conv = model.layer4[1].conv2
    final_conv.register_forward_hook(forward_hook)

    output = model(input_tensor)
    one_hot = torch.zeros_like(output)
    one_hot[0][target_class] = 1
    model.zero_grad()
    output.backward(gradient=one_hot)

    grad = gradients[0]
    act = activations[0]
    weights = torch.mean(grad, dim=(2, 3), keepdim=True)
    cam = torch.sum(weights * act, dim=1)
    cam = F.relu(cam).squeeze().detach().cpu().numpy()
    cam = (cam - cam.min()) / (cam.max() + 1e-8)
    return cam

def overlay_cam_on_image(cam, image):
    image_resized = np.array(image.resize((224, 224)).convert("RGB"))
    heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
    heatmap_resized = cv2.resize(heatmap, (224, 224))
    heatmap_resized = np.float32(heatmap_resized) / 255
    image_resized = np.float32(image_resized) / 255
    cam_img = heatmap_resized + image_resized
    cam_img = cam_img / np.max(cam_img)
    return np.uint8(255 * cam_img)

# -------------------------------
# Prediction + Explainability
# -------------------------------
def predict_grade_with_explainability(image, model, day):
    input_tensor = preprocess_image(image)
    with torch.no_grad():
        outputs = model(input_tensor)
        probs = F.softmax(outputs, dim=1)[0]
        pred_idx = torch.argmax(probs).item()

    grade = class_names[pred_idx]
    conf = probs[pred_idx].item()
    cam = generate_cam(model, input_tensor, pred_idx)
    cam_overlay = overlay_cam_on_image(cam, image)
    cam_overlay_img = Image.fromarray(cam_overlay).resize((300, 300))

    reason = generate_reason_from_cam(cam, day, grade)

    return grade, conf, cam_overlay_img, reason

# -------------------------------
# Reason Generation from Grad-CAM
# -------------------------------
def generate_reason_from_cam(cam, day, model_grade):
    cam_threshold = 0.4
    cam_score = np.mean(cam)
    high_activation = cam > 0.5
    activation_count = np.sum(high_activation)
    total_area = cam.shape[0] * cam.shape[1]
    activation_ratio = activation_count / total_area

    _, binary_map = cv2.threshold(np.uint8(high_activation * 255), 127, 255, cv2.THRESH_BINARY)
    blob_count, _ = cv2.connectedComponents(binary_map)
    blob_count -= 1

    missing_features = []

    if day == "Day 3":
        if blob_count < 8:
            missing_features.append("Low cell count")
        if activation_ratio < 0.15:
            missing_features.append("Fragmentation or unclear cells")
        if cam_score < cam_threshold:
            missing_features.append("Poor cell symmetry or clarity")
    elif day == "Day 4":
        if activation_ratio < 0.2:
            missing_features.append("Incomplete compaction")
        if cam_score < cam_threshold:
            missing_features.append("Weak central cohesion")
    elif day == "Day 5":
        if activation_ratio < 0.25:
            missing_features.append("Limited expansion")
        if cam_score < cam_threshold:
            missing_features.append("Poor ICM or TE visibility")

    if model_grade == "Grade A":
        return "All key features are present, indicating a high-quality embryo."
    else:
        return f"{model_grade} due to: {', '.join(missing_features)}"

# -------------------------------
# Implantation Recommendation Based on Grade
# -------------------------------
def generate_implantation_recommendation(day3_grade, day4_grade, day5_grade):
    grades = [day3_grade, day4_grade, day5_grade]
    if "Grade C" in grades:
        return "🚨 Implantation is not recommended at this time. Please consult your doctor for further guidance and potential next steps."
    elif "Grade B" in grades:
        return "🌟 Implantation may be considered, but we recommend consulting your doctor for a more thorough evaluation and tailored advice."
    else:
        return "✅ Implantation is strongly recommended. The embryos are of high quality, showing excellent potential."

# -------------------------------
# Load All Models
# -------------------------------
model_day3 = load_model_for_day(3)
model_day4 = load_model_for_day(4)
model_day5 = load_model_for_day(5)
binary_model = load_binary_model()

# -------------------------------
# Hugging Face Mistral LLM for Chatbot
# -------------------------------
hf_token = "your Token"
login(token=hf_token)

llm = HuggingFaceEndpoint(
    repo_id="mistralai/Mistral-7B-Instruct-v0.3",
    huggingfacehub_api_token=hf_token,
    task="text-generation",
    temperature=0.7,
    max_new_tokens=200
)

def get_medical_response(user_input):
    return llm.invoke(user_input)

# -------------------------------
# UI: Upload Images
# -------------------------------
col1, col2, col3, col4 = st.columns([1, 1, 1, 0.5])
images = {}

with col1:
    st.markdown("**Day 3 - 8-cell**")
    d3_file = st.file_uploader("Upload", type=["jpg", "png", "jpeg"], label_visibility="collapsed", key="day3")
    if d3_file:
        images["Day 3"] = Image.open(d3_file).convert('RGB')

with col2:
    st.markdown("**Day 4 - Morula**")
    d4_file = st.file_uploader("Upload", type=["jpg", "png", "jpeg"], label_visibility="collapsed", key="day4")
    if d4_file:
        images["Day 4"] = Image.open(d4_file).convert('RGB')

with col3:
    st.markdown("**Day 5 - Blastocyst**")
    d5_file = st.file_uploader("Upload", type=["jpg", "png", "jpeg"], label_visibility="collapsed", key="day5")
    if d5_file:
        images["Day 5"] = Image.open(d5_file).convert('RGB')

with col4:
    st.markdown("###")
    predict = st.button("🔍 Predict", use_container_width=True)

# -------------------------------
# Show Uploaded Images
# -------------------------------
if len(images) == 3:
    st.markdown("### Uploaded Images")
    col1, col2, col3 = st.columns([1, 1, 1])
    for idx, (day, img) in enumerate(images.items()):
        with [col1, col2, col3][idx]:
            st.image(img, caption=f"{day} Image", width=250)

# -------------------------------
# Perform Prediction
# -------------------------------
if predict and len(images) == 3:
    st.markdown("## 🔍 Embryo Grading Results")

    col1, col2, col3 = st.columns([1, 1, 1])
    grades = {}
    for idx, (day, img) in enumerate(images.items()):
        model = model_day3 if day == "Day 3" else model_day4 if day == "Day 4" else model_day5

        if not is_embryo_image(img):
            st.markdown(f"<p class='error-message'>🚫 The image for {day} is not a valid embryo image. Please upload a valid embryo image.</p>", unsafe_allow_html=True)
            continue

        grade, conf, cam_overlay, reason = predict_grade_with_explainability(img, model, day)
        grades[day] = grade

        with [col1, col2, col3][idx]:
            st.image(cam_overlay, caption="Grad-CAM",width=250)
            st.markdown(f"**Grade: {grade}**")
            st.markdown(f"**Confidence: {conf:.1%}**")
            st.markdown(f"**Reason:** {reason}")

    # -------------------------------
    # Implantation Recommendation Based on All Grades
    # -------------------------------
    if len(grades) == 3:
        implantation_recommendation = generate_implantation_recommendation(grades["Day 3"], grades["Day 4"], grades["Day 5"])
        st.markdown(f"<div class='implantation-recommendation'>{implantation_recommendation}</div>", unsafe_allow_html=True)
    # -------------------------------
    # Save to Database
    # -------------------------------
    if len(grades) == 3 and st.session_state.get('patient_logged_in'):
        patient_info = st.session_state['patient_info']
        patient_entry = Patient(
            patient_id=patient_info['id'],
            name=patient_info['name'],
            dob=patient_info['dob'],
            age=patient_info['age'],
            grade_day3=grades.get("Day 3", ""),
            grade_day4=grades.get("Day 4", ""),
            grade_day5=grades.get("Day 5", "")
        )
        try:
            session.merge(patient_entry)  # Insert or update if exists
            session.commit()
            st.success("✅ Patient data and grades saved to the database.")
        except Exception as e:
            session.rollback()
            st.error(f"❌ Failed to save data: {e}")

with st.expander("👩‍⚕️ Doctor's Advice"):
    st.markdown("""
    - **Grade A:** Embryos with the best morphology, high implantation potential.
    - **Grade B:** Average embryos, often still viable.
    - **Grade C:** Poor quality, typically low implantation chance. Further medical consultation recommended.
    - **Tip:** Maintain a healthy lifestyle, and follow embryologist or fertility specialist guidance closely.
    """)

# -------------------------------
# Chatbot Section
# -------------------------------
st.markdown("## 🤖 IVF Information Chatbot")
user_input = st.text_input("Ask me anything about IVF (e.g., What is IVF?)")
if user_input:
    response = get_medical_response(user_input)
    st.write(response)
