# from transformers import AutoTokenizer, AutoModelForCausalLM

# tokenizer = AutoTokenizer.from_pretrained("TinyLlama/TinyLlama-1.1B-Chat-v1.0")
# model = AutoModelForCausalLM.from_pretrained("TinyLlama/TinyLlama-1.1B-Chat-v1.0")

# import os
# import fitz  # PyMuPDF
# import json
# import time
# from sentence_transformers import SentenceTransformer, util
# from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
# import torch
# import re

# # ---- Config ----
# INPUT_DIR = "./pdf_inputs"
# OUTPUT_FILE = "final_output.json"
# EMBED_MODEL = 'sentence-transformers/all-MiniLM-L6-v2'
# SUMM_MODEL = 'Falconsai/text_summarization'  # CPU-friendly

# # ---- Load models ----
# embedder = SentenceTransformer(EMBED_MODEL)
# tokenizer = AutoTokenizer.from_pretrained(SUMM_MODEL)
# summarizer = AutoModelForSeq2SeqLM.from_pretrained(SUMM_MODEL)

# # ---- Helper Functions ----
# def clean_text(text):
#     text = re.sub(r'\s+', ' ', text)
#     return text.strip()

# def extract_text_blocks(pdf_path):
#     doc = fitz.open(pdf_path)
#     chunks = []
#     for page_num, page in enumerate(doc):
#         blocks = page.get_text("blocks")
#         for block in blocks:
#             text = clean_text(block[4])
#             if len(text.split()) > 10:
#                 chunks.append({
#                     "text": text,
#                     "page": page_num + 1
#                 })
#     return chunks

# def rank_by_relevance(chunks, query):
#     texts = [c["text"] for c in chunks]
#     doc_embeds = embedder.encode(texts, convert_to_tensor=True)
#     query_embed = embedder.encode([query], convert_to_tensor=True)
#     scores = util.cos_sim(query_embed, doc_embeds)[0]

#     ranked = sorted(
#         zip(chunks, scores),
#         key=lambda x: x[1],
#         reverse=True
#     )[:3]
#     return [r[0] for r in ranked]

# def summarize(text):
#     inputs = tokenizer.encode(text, return_tensors="pt", max_length=512, truncation=True)
#     summary_ids = summarizer.generate(inputs, max_length=100, min_length=20, num_beams=4)
#     return tokenizer.decode(summary_ids[0], skip_special_tokens=True)

# # ---- Main Pipeline ----
# def process_documents(persona, job):
#     start = time.time()
#     result = {
#         "metadata": {
#             "persona": persona,
#             "job_to_be_done": job
#         },
#         "documents": []
#     }

#     for file in os.listdir(INPUT_DIR):
#         if not file.endswith(".pdf"):
#             continue
#         path = os.path.join(INPUT_DIR, file)
#         chunks = extract_text_blocks(path)
#         top_chunks = rank_by_relevance(chunks, job)
#         summarized = [summarize(c["text"]) for c in top_chunks]
#         result["documents"].append({
#             "filename": file,
#             "summaries": summarized
#         })

#     result["processing_time_sec"] = round(time.time() - start, 2)

#     with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
#         json.dump(result, f, indent=2)

#     print(f"✅ Done. Output saved to {OUTPUT_FILE}")

# # ---- Run ----
# if __name__ == "__main__":
#     persona = "Government Grant Officer"
#     job = "Find training and educational opportunities for rural youth"
#     process_documents(persona, job)


# import os
# import json
# import fitz  # PyMuPDF
# import torch
# from datetime import datetime
# from sentence_transformers import SentenceTransformer, util
# from transformers import AutoTokenizer, AutoModelForCausalLM

# # === Load Input Config ===
# with open("input.json", "r", encoding="utf-8") as f:
#     config = json.load(f)

# documents = config["documents"]
# persona = config["persona"]["role"]
# job = config["job_to_be_done"]["task"]
# input_folder = "pdf_inputs"

# # === Load Embedding Model (MiniLM) ===
# embedder = SentenceTransformer("all-MiniLM-L6-v2")

# # === Load TinyLlama (offline, CPU) ===
# tokenizer = AutoTokenizer.from_pretrained("TinyLlama/TinyLlama-1.1B-Chat-v1.0")
# model = AutoModelForCausalLM.from_pretrained(
#     "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
#     torch_dtype=torch.float32,
#     device_map="cpu"
# )

# # === Helper Functions ===
# def extract_sections(pdf_path):
#     doc = fitz.open(pdf_path)
#     sections = []
#     for page_num in range(len(doc)):
#         blocks = doc[page_num].get_text("dict")["blocks"]
#         for block in blocks:
#             if "lines" not in block:
#                 continue
#             for line in block["lines"]:
#                 text = " ".join([span["text"] for span in line["spans"]]).strip()
#                 size = max([span["size"] for span in line["spans"]])
#                 if len(text) > 30 and size >= 12:
#                     sections.append({
#                         "text": text,
#                         "page": page_num + 1
#                     })
#     return sections

# def relevance_score(text):
#     context = f"{persona}. Task: {job}"
#     e1 = embedder.encode(context, convert_to_tensor=True)
#     e2 = embedder.encode(text, convert_to_tensor=True)
#     return util.pytorch_cos_sim(e1, e2).item()

# def summarize_text(text):
#     prompt = f"Summarize this for persona '{persona}' doing task '{job}':\n{text}\nSummary:"
#     inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512).to("cpu")
#     output_ids = model.generate(inputs["input_ids"], max_new_tokens=100)
#     return tokenizer.decode(output_ids[0], skip_special_tokens=True)

# # === Process PDFs ===
# ranked_sections = []
# for doc in documents:
#     pdf_path = os.path.join(input_folder, doc["filename"])
#     if not os.path.exists(pdf_path):
#         continue
#     print(f"Processing: {doc['filename']}")
#     sections = extract_sections(pdf_path)
#     for sec in sections:
#         score = relevance_score(sec["text"])
#         ranked_sections.append({
#             "document": doc["filename"],
#             "section_title": sec["text"][:100],
#             "importance_score": score,
#             "page_number": sec["page"]
#         })

# # === Sort & Summarize ===
# ranked_sections.sort(key=lambda x: x["importance_score"], reverse=True)
# top5 = ranked_sections[:5]

# subsection_analysis = []
# for sec in top5:
#     summary = summarize_text(sec["section_title"])
#     subsection_analysis.append({
#         "document": sec["document"],
#         "refined_text": summary,
#         "page_number": sec["page_number"]
#     })

# # === Generate Output ===
# output = {
#     "metadata": {
#         "input_documents": [doc["filename"] for doc in documents],
#         "persona": persona,
#         "job_to_be_done": job,
#         "processing_timestamp": str(datetime.now())
#     },
#     "extracted_sections": [
#         {
#             "document": sec["document"],
#             "section_title": sec["section_title"],
#             "importance_rank": i + 1,
#             "page_number": sec["page_number"]
#         } for i, sec in enumerate(top5)
#     ],
#     "subsection_analysis": subsection_analysis
# }

# with open("output.json", "w", encoding="utf-8") as f:
#     json.dump(output, f, indent=2, ensure_ascii=False)

# print("✅ Output saved to output.json")

import os
import json
import fitz  # PyMuPDF
import torch
from datetime import datetime
from sentence_transformers import SentenceTransformer, util
from transformers import AutoTokenizer, AutoModelForCausalLM

# Load input config
with open("input.json", "r", encoding="utf-8") as f:
    config = json.load(f)

documents = config["documents"]
persona = config["persona"]["role"]
job = config["job_to_be_done"]["task"]
input_folder = "pdf_inputs"

# Load sentence embedding model
embedder = SentenceTransformer("all-MiniLM-L6-v2")

# Load TinyLlama LLM for summarization
tokenizer = AutoTokenizer.from_pretrained("TinyLlama/TinyLlama-1.1B-Chat-v1.0")
model = AutoModelForCausalLM.from_pretrained(
    "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    torch_dtype=torch.float32,
    device_map=None
).cpu()

# Helper: Extract headings from PDFs
def extract_sections(pdf_path):
    doc = fitz.open(pdf_path)
    sections = []
    for page_num in range(len(doc)):
        page = doc[page_num]
        blocks = page.get_text("dict")["blocks"]
        for b in blocks:
            if "lines" not in b:
                continue
            for line in b["lines"]:
                spans = line["spans"]
                text = " ".join([span["text"].strip() for span in spans])
                font_sizes = [span["size"] for span in spans]
                if len(text) > 20 and max(font_sizes) > 11:
                    sections.append({
                        "text": text,
                        "page": page_num + 1
                    })
    return sections

# Helper: Compute relevance score using embeddings
def relevance_score(section_text):
    context = f"{persona}: {job}"
    emb1 = embedder.encode(context, convert_to_tensor=True)
    emb2 = embedder.encode(section_text, convert_to_tensor=True)
    return util.pytorch_cos_sim(emb1, emb2).item()

# Helper: Summarize using TinyLlama
def summarize_with_llm(text):
    prompt = f"Summarize the following for a {persona} tasked to: {job}:\n{text}\nSummary:"
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512).to("cpu")
    output_ids = model.generate(**inputs, max_new_tokens=120, do_sample=False)
    return tokenizer.decode(output_ids[0], skip_special_tokens=True)

# Step 1: Extract all headings/sections
all_sections = []
for doc in documents:
    path = os.path.join(input_folder, doc["filename"])
    if not os.path.exists(path):
        continue
    print(f"Processing {doc['filename']}...")
    sections = extract_sections(path)
    for s in sections:
        score = relevance_score(s["text"])
        all_sections.append({
            "document": doc["filename"],
            "section_title": s["text"][:80],
            "page_number": s["page"],
            "importance_score": score
        })

# Step 2: Rank and summarize top 5
all_sections.sort(key=lambda x: x["importance_score"], reverse=True)
top_sections = all_sections[:5]
subsection_analysis = []

for s in top_sections:
    summary = summarize_with_llm(s["section_title"])
    subsection_analysis.append({
        "document": s["document"],
        "refined_text": summary,
        "page_number": s["page_number"]
    })

# Step 3: Create final output
output = {
    "metadata": {
        "input_documents": [d["filename"] for d in documents],
        "persona": persona,
        "job_to_be_done": job,
        "processing_timestamp": datetime.now().isoformat()
    },
    "extracted_sections": [
        {
            "document": s["document"],
            "section_title": s["section_title"],
            "importance_rank": idx + 1,
            "page_number": s["page_number"]
        }
        for idx, s in enumerate(top_sections)
    ],
    "subsection_analysis": subsection_analysis
}

with open("output.json", "w", encoding="utf-8") as f:
    json.dump(output, f, indent=2, ensure_ascii=False)

print("✅ Completed! Output written to output.json")
