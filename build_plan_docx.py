import os
from pathlib import Path
import docx
from docx.shared import Inches, Pt, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_TABLE_ALIGNMENT, WD_ALIGN_VERTICAL
from docx.oxml import OxmlElement
from docx.oxml.ns import qn

def set_cell_background(cell, fill_hex):
    tcPr = cell._tc.get_or_add_tcPr()
    shd = OxmlElement('w:shd')
    shd.set(qn('w:val'), 'clear')
    shd.set(qn('w:color'), 'auto')
    shd.set(qn('w:fill'), fill_hex)
    tcPr.append(shd)

def set_cell_margins(cell, top=140, bottom=140, left=180, right=180):
    tcPr = cell._tc.get_or_add_tcPr()
    tcMar = OxmlElement('w:tcMar')
    for m, val in [('w:top', top), ('w:bottom', bottom), ('w:left', left), ('w:right', right)]:
        node = OxmlElement(m)
        node.set(qn('w:w'), str(val))
        node.set(qn('w:type'), 'dxa')
        tcMar.append(node)
    tcPr.append(tcMar)

def add_callout(doc, title, text, bg_hex="F0F4F8", border_hex="1D4ED8"):
    tbl = doc.add_table(rows=1, cols=1)
    tbl.alignment = WD_TABLE_ALIGNMENT.CENTER
    tbl.autofit = False
    tbl.columns[0].width = Inches(6.5)
    cell = tbl.cell(0, 0)
    set_cell_background(cell, bg_hex)
    set_cell_margins(cell, top=160, bottom=160, left=220, right=200)

    tcPr = cell._tc.get_or_add_tcPr()
    tcBorders = OxmlElement('w:tcBorders')
    left = OxmlElement('w:left')
    left.set(qn('w:val'), 'single')
    left.set(qn('w:sz'), '32')
    left.set(qn('w:space'), '0')
    left.set(qn('w:color'), border_hex)
    tcBorders.append(left)

    for side in ['top', 'bottom', 'right']:
        b = OxmlElement(f'w:{side}')
        b.set(qn('w:val'), 'none')
        tcBorders.append(b)
    tcPr.append(tcBorders)

    p = cell.paragraphs[0]
    p.paragraph_format.space_before = Pt(2)
    p.paragraph_format.space_after = Pt(4)
    run_t = p.add_run(f"📌 {title}\n")
    run_t.font.name = "Calibri"
    run_t.font.size = Pt(11)
    run_t.font.bold = True
    run_t.font.color.rgb = RGBColor(0x1E, 0x29, 0x3B)

    run_body = p.add_run(text)
    run_body.font.name = "Calibri"
    run_body.font.size = Pt(10)
    run_body.font.color.rgb = RGBColor(0x33, 0x41, 0x55)
    doc.add_paragraph().paragraph_format.space_after = Pt(4)

def build_document():
    doc = docx.Document()
    for section in doc.sections:
        section.top_margin = Inches(1.0)
        section.bottom_margin = Inches(1.0)
        section.left_margin = Inches(1.0)
        section.right_margin = Inches(1.0)

    # Title
    t_p = doc.add_paragraph()
    t_p.paragraph_format.space_before = Pt(0)
    t_p.paragraph_format.space_after = Pt(4)
    r = t_p.add_run("Existing Foundation LLM Fine-Tuning")
    r.font.name = "Calibri"
    r.font.size = Pt(26)
    r.font.bold = True
    r.font.color.rgb = RGBColor(0x0F, 0x17, 0x2A)

    sub_p = doc.add_paragraph()
    sub_p.paragraph_format.space_after = Pt(14)
    r_sub = sub_p.add_run("Comprehensive Engineering Master Plan: LoRA & QLoRA Fine-Tuning for Qwen 2.5, TinyLlama & LLaMA Models")
    r_sub.font.name = "Calibri"
    r_sub.font.size = Pt(13)
    r_sub.font.color.rgb = RGBColor(0x47, 0x55, 0x69)

    add_callout(doc, "Project Purpose & Paradigm",
                "While the Custom_LLM_Project built a Decoder-Only model from absolute scratch (zero initial weights), "
                "the Existing_LLM_Project leverages world-class pre-trained foundation models (Qwen 2.5, TinyLlama, Llama 3) "
                "and adapts them to proprietary WhatsApp & citizen identity datasets using Parameter-Efficient Fine-Tuning (PEFT/LoRA). "
                "This achieves world-class reasoning, rich vocabulary, and instant conversational fluency with minimal compute.")

    # Section 1
    h1 = doc.add_heading("1. Strategic Comparison: Scratch Model vs. Existing LLM Fine-Tuning", level=1)
    h1.paragraph_format.space_before = Pt(14)

    comp_table = doc.add_table(rows=6, cols=3)
    comp_table.alignment = WD_TABLE_ALIGNMENT.CENTER
    headers = ["Evaluation Metric", "Custom LLM (From Zero Scratch)", "Existing LLM (LoRA Fine-Tuning)"]
    for i, h in enumerate(headers):
        cell = comp_table.cell(0, i)
        cell.text = h
        set_cell_background(cell, "0F172A")
        set_cell_margins(cell, top=140, bottom=140, left=160, right=160)
        for p in cell.paragraphs:
            for r in p.runs:
                r.font.name = "Calibri"
                r.font.bold = True
                r.font.color.rgb = RGBColor(0xFF, 0xFF, 0xFF)
                r.font.size = Pt(10)

    rows_data = [
        ("Base Knowledge", "Zero initially; knows only training corpus.", "Billions of tokens (world knowledge, logic, grammar)."),
        ("Compute Required", "Requires full pre-training (all layers active).", "Trains only <1% adapter parameters (super fast)."),
        ("Out-of-Domain Generalization", "Sensitive to template/slang drift at small sizes.", "Highly robust; understands synonyms, slang & typos."),
        ("Hardware Needs", "Laptop RAM/MPS runs 28M-35M smoothly.", "Runs 0.5B-1.1B on Mac MPS; 7B-8B with 4-bit QLoRA."),
        ("Primary Use Case", "Data sovereignty, total architectural control.", "Production-grade domain adaptation & highest accuracy.")
    ]

    for row_idx, data in enumerate(rows_data, start=1):
        bg = "F8FAFC" if row_idx % 2 == 1 else "FFFFFF"
        for col_idx, text in enumerate(data):
            cell = comp_table.cell(row_idx, col_idx)
            cell.text = text
            set_cell_background(cell, bg)
            set_cell_margins(cell, top=100, bottom=100, left=140, right=140)
            for p in cell.paragraphs:
                for r in p.runs:
                    r.font.name = "Calibri"
                    r.font.size = Pt(9.5)
                    if col_idx == 0:
                        r.font.bold = True

    doc.add_paragraph().paragraph_format.space_after = Pt(8)

    # Section 2
    h2 = doc.add_heading("2. Recommended Foundation Models for Local & Cloud Execution", level=1)
    h2.paragraph_format.space_before = Pt(12)

    p = doc.add_paragraph()
    p.add_run("Choosing the correct foundation model is crucial for balancing inference latency, memory consumption, and reasoning capacity:")

    bullets = [
        ("Qwen/Qwen2.5-0.5B-Instruct (Recommended for Mac Laptop): ", "0.49 Billion parameters. State-of-the-art benchmark scores for sub-billion models. Native ChatML format, blazingly fast on Apple Silicon unified memory (uses ~1.2 GB RAM)."),
        ("TinyLlama/TinyLlama-1.1B-Chat-v1.0: ", "1.1 Billion parameters trained on 3 Trillion tokens. Compact LLaMA architecture, ideal for low-latency edge deployment."),
        ("Qwen/Qwen2.5-7B-Instruct (Recommended for GPU / Colab): ", "7.6 Billion parameters. World-leading reasoning, multi-lingual proficiency, and conversational depth. Fine-tunable on free Colab T4 GPU via 4-bit QLoRA.")
    ]
    for b_title, b_desc in bullets:
        bp = doc.add_paragraph(style='List Bullet')
        r1 = bp.add_run(b_title)
        r1.font.bold = True
        bp.add_run(b_desc)

    # Section 3
    h3 = doc.add_heading("3. The Mathematical Principle of LoRA (Low-Rank Adaptation)", level=1)
    h3.paragraph_format.space_before = Pt(12)

    p3 = doc.add_paragraph()
    p3.add_run("Instead of updating all pre-trained weight matrices W_0 in the Transformer (which would require immense VRAM and gradient storage), "
               "LoRA freezes W_0 and injects trainable rank-decomposition matrices:\n\n"
               "    W = W_0 + ΔW = W_0 + (B · A) · (α / r)\n\n"
               "Where:\n"
               "• W_0 is frozen (d × k matrix).\n"
               "• B is initialized to 0 (d × r matrix).\n"
               "• A is initialized with Gaussian noise (r × k matrix).\n"
               "• r is the rank (e.g., r = 16), which reduces trainable parameters by >98%.\n"
               "• α is the scaling constant (e.g., α = 32).")

    add_callout(doc, "Zero-Overhead Inference via Weight Merging",
                "At deployment time, the adapter weights can be permanently fused back into the base model: "
                "W_merged = W_0 + (B · A). The resulting model has exactly the same latency and architecture "
                "as the original base model, requiring zero special adapter runtimes.")

    # Section 4
    h4 = doc.add_heading("4. Project Directory Structure & Workflow Pipeline", level=1)
    h4.paragraph_format.space_before = Pt(12)

    p4 = doc.add_paragraph()
    p4.add_run("The Existing_LLM_Project workspace is organized cleanly into modular, production-tested components:\n\n"
               "• 01_data/ - Contains prepare_finetune_data.py, formatted ChatML train (23,750 samples) and validation splits.\n"
               "• 02_config/ - Central config.py managing model identifiers, LoRA rank (r=16, alpha=32), learning rate, and target modules.\n"
               "• 03_finetune/ - train_lora_local.py for native Mac MPS/CPU execution and train_qlora_colab.py for 4-bit GPU training.\n"
               "• 04_merge/ - merge_adapters.py to fuse LoRA adapters directly into standalone safetensors weights.\n"
               "• 05_inference/ - Multi-channel serving via cli_chat.py (interactive terminal), app_streamlit.py (Web GUI), and server_fastapi.py (REST API).")

    # Section 5
    h5 = doc.add_heading("5. Operational Execution Steps (No Code Edits Required)", level=1)
    h5.paragraph_format.space_before = Pt(12)

    exec_steps = [
        ("Step 1: Setup Environment", "Run 'source venv/bin/activate' and install requirements ('pip install -r requirements.txt')."),
        ("Step 2: Prepare Training Data", "Run 'python3 01_data/prepare_finetune_data.py' to generate balanced ChatML datasets."),
        ("Step 3: Execute Local Fine-Tuning", "Run 'python3 03_finetune/train_lora_local.py'. The script automatically attaches LoRA to attention heads and trains on Apple Silicon MPS."),
        ("Step 4: Merge Adapters (Optional)", "Run 'python3 04_merge/merge_adapters.py' to fuse weights into a standalone safetensors model."),
        ("Step 5: Interactive Chat Testing", "Launch 'python3 05_inference/cli_chat.py' or 'streamlit run 05_inference/app_streamlit.py' to converse.")
    ]

    for st_title, st_desc in exec_steps:
        sp = doc.add_paragraph(style='List Bullet')
        r1 = sp.add_run(f"{st_title}: ")
        r1.font.bold = True
        sp.add_run(st_desc)

    out_file = "/Users/DF/Documents/llm/Existing_LLM_Project/Existing_LLM_Implementation_Plan.docx"
    doc.save(out_file)
    print(f"Implementation Plan successfully written to: {out_file}")

if __name__ == "__main__":
    build_document()

