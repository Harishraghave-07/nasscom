import gradio as gr
import requests
import os
import json

GATEWAY_URL = os.environ.get("GATEWAY_URL", "http://localhost:8000")


def detect_text(text, role, token):
    headers = {"Authorization": f"Bearer {token}"} if token else {}
    resp = requests.post(f"{GATEWAY_URL}/api/v1/process/text", data={"text": text}, headers=headers)
    if resp.status_code != 200:
        return {}, f"Error: {resp.status_code} {resp.text}"
    return resp.json(), ""


def anonymize_text(text, selections, role, token):
    # selections could be list of entities to keep or anonymize; pass through gateway anonymize endpoint
    headers = {"Authorization": f"Bearer {token}", "Content-Type": "application/json"} if token else {"Content-Type": "application/json"}
    payload = {"text": text, "keep": selections}
    resp = requests.post(f"{GATEWAY_URL}/api/v1/route/PHI_ANONYMIZER", json=payload, headers=headers)
    if resp.status_code != 200:
        return "", f"Error: {resp.status_code} {resp.text}"
    j = resp.json()
    return j.get("anonymized_text") or j.get("text") or json.dumps(j), ""


def upload_pdf(file, role, token):
    if file is None:
        return "No file provided"
    files = {"file": (file.name, file.read(), file.type or 'application/pdf')}
    headers = {"Authorization": f"Bearer {token}"} if token else {}
    resp = requests.post(f"{GATEWAY_URL}/api/v1/upload/pdf", files=files, headers=headers)
    if resp.status_code != 200:
        return f"Error: {resp.status_code} {resp.text}"
    return "Uploaded and processed. Check job status in logs or pipeline output."


with gr.Blocks(title="Clinical Image Masker (UI)", analytics_enabled=False) as demo:
    gr.Markdown("# Clinical Image Masker — PHI Detection & Anonymization")
    with gr.Row():
        role = gr.Radio(choices=["user", "service_account", "admin"], label="Role (demo)", value="user")
        token = gr.Textbox(label="JWT Token (demo)", placeholder="Paste JWT token for auth (optional)")

    with gr.Tabs():
        with gr.TabItem("Text Input"):
            text_in = gr.Textbox(lines=8, label="Input Text")
            detect_btn = gr.Button("Detect PHI")
            detect_out = gr.JSON(label="Detections")
            detect_err = gr.Textbox(label="Status", interactive=False)
            anonymize_btn = gr.Button("Anonymize Text")
            anonymized_out = gr.Textbox(lines=8, label="Anonymized Text")

            def do_detect(text, role, token):
                detections, err = detect_text(text, role, token)
                return detections, err

            detect_btn.click(do_detect, inputs=[text_in, role, token], outputs=[detect_out, detect_err])

            def do_anonymize(text, det_json, role, token):
                # simple selection protocol: anonymize all detected spans
                selections = []
                if isinstance(det_json, dict):
                    for d in det_json.get("detections", []):
                        selections.append({"start": d.get("start"), "end": d.get("end"), "type": d.get("phi_type") or d.get("entity_type")})
                anon, err = anonymize_text(text, selections, role, token)
                return anon or "", err

            anonymize_btn.click(do_anonymize, inputs=[text_in, detect_out, role, token], outputs=[anonymized_out, detect_err])

        with gr.TabItem("Upload PDF"):
            pdf_file = gr.File(label="Upload PDF", file_types=[".pdf"])
            upload_btn = gr.Button("Upload & Process")
            upload_status = gr.Textbox(label="Upload Status", interactive=False)
            upload_btn.click(upload_pdf, inputs=[pdf_file, role, token], outputs=[upload_status])

        with gr.TabItem("Export"):
            export_text = gr.Textbox(lines=8, label="Export Anonymized Text")
            export_btn = gr.Button("Export JSON")
            def export_json(text):
                return json.dumps({"anonymized_text": text})
            export_btn.click(export_json, inputs=[export_text], outputs=[gr.File(label="Download JSON")])

    demo.launch(server_port=7860, server_name="0.0.0.0")
