import streamlit as st
import torch
import pandas as pd
from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification

st.set_page_config(page_title="Bloom's Taxonomy Classifier", page_icon="🎓", layout="centered")

st.title("🎓 Bloom's Taxonomy Question Classifier")
st.write("An NLP-powered tool that classifies academic questions into cognitive complexity levels using fine-tuned DistilBERT.")

@st.cache_resource
def load_model():
    model_path = "./saved_blooms_model"
    tokenizer = DistilBertTokenizerFast.from_pretrained(model_path)
    model = DistilBertForSequenceClassification.from_pretrained(model_path)
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    model.to(device)
    model.eval()
    return tokenizer, model, device

try:
    tokenizer, model, device = load_model()
    st.success("✅ Model loaded successfully on Apple Silicon GPU!")
except Exception as e:
    st.error(f"❌ Could not load saved model. Ensure 'python train.py' completed successfully. Error: {e}")
    st.stop()

# Tab layout for Single Question vs. Batch Processing
tab1, tab2 = st.tabs(["Single Question", "Batch CSV Upload"])

with tab1:
    st.subheader("Real-time Question Classification")
    question = st.text_area(
        "Enter an exam question or assessment prompt:",
        "Compare and contrast supervised and unsupervised learning algorithms."
    )

    if st.button("Classify Question Level"):
        if question.strip() == "":
            st.warning("Please enter a question.")
        else:
            inputs = tokenizer(question, return_tensors="pt", truncation=True, max_length=128).to(device)
            with torch.no_grad():
                outputs = model(**inputs)
                probs = torch.nn.functional.softmax(outputs.logits, dim=-1)[0]

            top_pred = torch.argmax(probs).item()
            labels = model.config.id2label

            st.subheader("Classification Result")
            st.metric(label="Predicted Cognitive Level", value=labels[top_pred])

            st.subheader("Confidence Scores Across All Levels")
            for idx, prob in enumerate(probs):
                st.progress(float(prob.item()), text=f"{labels[idx]}: {prob.item()*100:.1f}%")

with tab2:
    st.subheader("Batch Question Classification")
    st.write("Upload a CSV file containing a column named `question` to classify multiple questions at once.")
    
    uploaded_file = st.file_uploader("Choose a CSV file", type=["csv"])

    if uploaded_file is not None:
        batch_df = pd.read_csv(uploaded_file)
        if "question" in batch_df.columns:
            results = []
            confidences = []
            
            with st.spinner("Classifying questions..."):
                for q in batch_df["question"]:
                    inputs = tokenizer(str(q), return_tensors="pt", truncation=True, max_length=128).to(device)
                    with torch.no_grad():
                        outputs = model(**inputs)
                        probs = torch.nn.functional.softmax(outputs.logits, dim=-1)[0]
                        top_pred = torch.argmax(probs).item()
                        
                    results.append(model.config.id2label[top_pred])
                    confidences.append(f"{probs[top_pred].item()*100:.1f}%")

            batch_df["Predicted_Bloom_Level"] = results
            batch_df["Confidence"] = confidences

            st.dataframe(batch_df)

            st.download_button(
                label="📥 Download Classified CSV",
                data=batch_df.to_csv(index=False),
                file_name="classified_blooms_questions.csv",
                mime="text/csv"
            )
        else:
            st.error("The uploaded CSV file must contain a column named 'question'.")
