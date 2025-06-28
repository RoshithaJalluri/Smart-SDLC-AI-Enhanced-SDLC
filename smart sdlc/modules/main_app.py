# main_app.py (Smart SDLC - Full 8-Feature Integration for Google Colab + Gradio)

import os
import sys
import gradio as gr
from getpass import getpass
from huggingface_hub import login

# === Setup Hugging Face Access Token (Safe for Colab) ===
hf_token = getpass("Enter your Hugging Face Token:")
os.environ["HUGGINGFACE_TOKEN"] = hf_token
login(token=hf_token)

# === Add project module path for imports ===
sys.path.append("smart_sdlc/modules")

# === Import Smart SDLC Modules ===
from ai_story_generator import RequirementsProcessor
from code_generator import MultilingualCodeGenerator, ProgrammingLanguage, CodeComplexity, CodeGenerationRequest
from bug_resolver import BugResolver
from doc_generator import GraniteAIEngine
from feedback_routes import create_feedback_interface
from github_service import create_github_integration_interface
from main_chatbot import SmartSDLCChatbot

# === Initialize Core Components ===
ai_engine = GraniteAIEngine()
req_processor = RequirementsProcessor()
code_generator = MultilingualCodeGenerator()
bug_resolver = BugResolver()
feedback_ui = create_feedback_interface()
github_ui = create_github_integration_interface()
chatbot = SmartSDLCChatbot()

# === Gradio UI Logic ===
def process_pdf(pdf_file):
    try:
        return req_processor.process_pdf(pdf_file.name)["user_stories"]
    except Exception as e:
        return {"error": str(e)}

def generate_code(task_description, language, complexity):
    try:
        request = CodeGenerationRequest(
            task_description=task_description,
            language=ProgrammingLanguage[language.upper()],
            complexity=CodeComplexity[complexity.upper()]
        )
        result = code_generator.generate_complete_solution(request)
        return result.main_code, result.test_code, result.documentation
    except Exception as e:
        return f"Error: {e}", "", ""

def fix_bugs(code_input):
    result = bug_resolver.resolve_bugs(code_input)
    return result.fixed_code, result.diff

def summarize_text(text_input):
    prompt = f"Summarize the following content:\n{text_input}"
    return ai_engine.generate_response(prompt)

def sync_to_github(repo_name, commit_msg):
    return github_ui.sync(repo_name, commit_msg)

# === Build Gradio Interface ===
with gr.Blocks() as demo:
    gr.Markdown("# 🤖 Smart SDLC - AI Enhanced Software Development Lifecycle")

    with gr.Tab("📄 Requirement Analysis"):
        pdf = gr.File()
        stories = gr.JSON()
        analyze_btn = gr.Button("Extract & Classify Requirements")
        analyze_btn.click(fn=process_pdf, inputs=pdf, outputs=stories)

    with gr.Tab("💻 Code Generation"):
        task = gr.Textbox(label="Task Description/User Story")
        lang = gr.Dropdown([lang.name for lang in ProgrammingLanguage], label="Language")
        complexity = gr.Dropdown([lvl.name for lvl in CodeComplexity], label="Complexity")
        code_out = gr.Code(label="Generated Code")
        test_out = gr.Code(label="Unit Tests")
        doc_out = gr.Textbox(label="Generated Documentation")
        gr.Button("Generate").click(generate_code, inputs=[task, lang, complexity], outputs=[code_out, test_out, doc_out])

    with gr.Tab("🐞 Bug Fixer"):
        buggy = gr.Code()
        fixed = gr.Code()
        diff = gr.Textbox()
        gr.Button("Fix").click(fix_bugs, inputs=buggy, outputs=[fixed, diff])

    with gr.Tab("📝 Code Summary"):
        txt_in = gr.Textbox(label="Paste any content or code")
        txt_out = gr.Textbox(label="Summarized Output")
        gr.Button("Summarize").click(fn=summarize_text, inputs=txt_in, outputs=txt_out)

    with gr.Tab("🧠 Chat Assistant"):
        chatbot.chat_interface.create_interface()

    with gr.Tab("📬 Feedback & Ratings"):
        feedback_ui.render()

    with gr.Tab("🐙 GitHub Integration"):
        github_ui.render()

    gr.Markdown("Created by Hema Sundar Sai Ratnala | Powered by IBM Granite & Gradio")

# Launch the app in Colab
if __name__ == '__main__':
    demo.launch(share=True)
