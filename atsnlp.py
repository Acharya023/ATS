import tkinter as tk
from tkinter import filedialog, messagebox
import pdfplumber
from transformers import BertTokenizer, BertModel
import torch
from sklearn.metrics.pairwise import cosine_similarity

# Function to extract text from PDF
def extract_text_from_pdf(pdf_path):
    with pdfplumber.open(pdf_path) as pdf:
        text = ""
        for page in pdf.pages:
            text += page.extract_text()
    return text


# Load pre-trained BERT model and tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased')

# Function to encode text using BERT
def encode_text(text):
    # Tokenize and convert to tensor
    inputs = tokenizer(text, return_tensors='pt', truncation=True, padding=True, max_length=512)
    with torch.no_grad():
        outputs = model(**inputs)
    # Use the output embedding for the [CLS] token as the sentence embedding
    return outputs.last_hidden_state.mean(dim=1).squeeze().numpy()

# Function to handle file selection for Job Descriptions
def upload_job_descriptions():
    job_description_files = filedialog.askopenfilenames(title="Select Job Description PDFs", filetypes=[("PDF Files", "*.pdf")])
    if job_description_files:
        job_descriptions.clear()
        for file in job_description_files:
            job_description = extract_text_from_pdf(file)
            job_descriptions.append(job_description)
        job_description_list.delete(1.0, tk.END)
        job_description_list.insert(tk.END, "\n\n".join([f"Job {i+1}:\n{desc[:300]}..." for i, desc in enumerate(job_descriptions)]))  # Show only first 300 chars


# Function to handle file selection for Resumes
def upload_resumes():
    resume_files = filedialog.askopenfilenames(title="Select Resume PDFs", filetypes=[("PDF Files", "*.pdf")])
    if resume_files:
        resumes.clear()
        for file in resume_files:
            resume_text = extract_text_from_pdf(file)
            resumes.append(resume_text)
        resume_list.delete(1.0, tk.END)
        resume_list.insert(tk.END, "\n\n".join([f"Resume {i+1}:\n{res[:300]}..." for i, res in enumerate(resumes)]))  # Show only first 300 chars


# Function to find the best matching job description for each resume using BERT embeddings
def find_best_resumes():
    if not job_descriptions or not resumes:
        messagebox.showerror("Error", "Please upload at least one job description and one resume.")
        return

    result_text.delete(1.0, tk.END)

    # Encode job descriptions and resumes using BERT
    job_description_embeddings = [encode_text(desc) for desc in job_descriptions]
    resume_embeddings = [encode_text(res) for res in resumes]

    # Find the best matching job description for each resume
    for resume_idx, resume_emb in enumerate(resume_embeddings):
        similarities = [cosine_similarity([resume_emb], [job_emb])[0][0] for job_emb in job_description_embeddings]
        best_match_idx = similarities.index(max(similarities))
        best_match_score = max(similarities)

        result_text.insert(tk.END, f"Resume {resume_idx + 1} (Similarity: {best_match_score:.4f}) is best matched with Job Description {best_match_idx + 1}.\n\n")


# Set up the Tkinter window
root = tk.Tk()
root.title("Resume Ranking System with BERT")
root.geometry("900x700")

# Initialize lists to hold the job descriptions and resumes
job_descriptions = []
resumes = []

# Upload Files Section
upload_job_description_button = tk.Button(root, text="Upload Job Descriptions", command=upload_job_descriptions)
upload_job_description_button.pack(pady=10)

job_description_list_label = tk.Label(root, text="Selected Job Descriptions:")
job_description_list_label.pack(anchor="w", padx=10)

job_description_list = tk.Text(root, height=6, width=80)
job_description_list.pack(pady=5)

upload_resume_button = tk.Button(root, text="Upload Resumes", command=upload_resumes)
upload_resume_button.pack(pady=10)

resume_list_label = tk.Label(root, text="Selected Resumes:")
resume_list_label.pack(anchor="w", padx=10)

resume_list = tk.Text(root, height=6, width=80)
resume_list.pack(pady=5)

# Find Best Resumes Section
find_button = tk.Button(root, text="Find Best Resume Matches", command=find_best_resumes)
find_button.pack(pady=10)

result_label = tk.Label(root, text="Best Matching Job Description for Each Resume:")
result_label.pack(anchor="w", padx=10)

result_text = tk.Text(root, height=15, width=80)
result_text.pack(pady=5)

# Start the Tkinter event loop
root.mainloop()
