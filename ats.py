import tkinter as tk
from tkinter import filedialog, messagebox
import pdfplumber
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import os


# Function to extract text from a PDF file
def extract_text_from_pdf(pdf_path):
    with pdfplumber.open(pdf_path) as pdf:
        text = ""
        for page in pdf.pages:
            text += page.extract_text()
    return text


# Function to handle file selection for Job Descriptions
def upload_job_descriptions():
    # Ask user to select job description PDFs
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
    # Ask user to select resume PDFs
    resume_files = filedialog.askopenfilenames(title="Select Resume PDFs", filetypes=[("PDF Files", "*.pdf")])
    if resume_files:
        resumes.clear()
        for file in resume_files:
            resume_text = extract_text_from_pdf(file)
            resumes.append(resume_text)
        resume_list.delete(1.0, tk.END)
        resume_list.insert(tk.END, "\n\n".join([f"Resume {i+1}:\n{res[:300]}..." for i, res in enumerate(resumes)]))  # Show only first 300 chars


# Function to find the highest matching job description for each resume
def find_best_resumes():
    if not job_descriptions or not resumes:
        messagebox.showerror("Error", "Please upload at least one job description and one resume.")
        return

    tfidf_vectorizer = TfidfVectorizer(stop_words='english')

    result_text.delete(1.0, tk.END)

    for resume_idx, resume in enumerate(resumes):
        # Combine the current resume with all job descriptions
        documents = [resume] + job_descriptions

        # Fit and transform the documents to get the TF-IDF matrix
        tfidf_matrix = tfidf_vectorizer.fit_transform(documents)

        # Calculate Cosine Similarity between the resume and each job description
        cosine_similarities = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:]).flatten()

        # Find the job description with the highest similarity
        max_similarity_idx = cosine_similarities.argmax()
        max_similarity_score = cosine_similarities[max_similarity_idx]

        # Display the result (only the highest matching job description for this resume)
        result_text.insert(tk.END, f"Resume {resume_idx + 1} (Similarity: {max_similarity_score:.4f}) is best matched with Job Description {max_similarity_idx + 1}.\n\n")


# Set up the Tkinter window
root = tk.Tk()
root.title("Resume Ranking System")
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
