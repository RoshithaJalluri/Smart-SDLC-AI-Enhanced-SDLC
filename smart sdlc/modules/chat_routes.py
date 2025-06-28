# chat_routes.py
import gradio as gr
from typing import List, Tuple
import json

class ChatInterface:
    """
    Gradio interface for the SDLC chatbot
    """
    
    def __init__(self, conversation_handler: ConversationHandler):
        self.conversation_handler = conversation_handler
        self.chat_history = []
    
    def chat_response(self, message: str, history: List[Tuple[str, str]]) -> Tuple[str, List[Tuple[str, str]]]:
        """
        Process chat message and return response with updated history
        """
        if not message.strip():
            return "", history
        
        # Generate AI response
        ai_response = self.conversation_handler.generate_contextual_response(message)
        
        # Add to conversation handler history
        self.conversation_handler.add_to_history(message, ai_response)
        
        # Update chat history for Gradio
        history.append((message, ai_response))
        
        return "", history
    
    def clear_chat(self):
        """
        Clear chat history
        """
        self.conversation_handler.conversation_history = []
        return [], ""
    
    def get_phase_suggestions(self, phase: str) -> str:
        """
        Get suggestions for a specific SDLC phase
        """
        suggestions = self.conversation_handler.get_quick_suggestions(phase)
        return "\n".join([f"• {suggestion}" for suggestion in suggestions])
    
    def create_interface(self) -> gr.Interface:
        """
        Create the Gradio interface for the chatbot
        """
        with gr.Blocks(
            title="Smart SDLC Assistant",
            theme=gr.themes.Soft(),
            css="""
            .chatbot-container {
                height: 500px;
                overflow-y: auto;
            }
            .suggestion-box {
                background-color: #f0f8ff;
                padding: 10px;
                border-radius: 5px;
                margin: 10px 0;
            }
            .phase-buttons {
                display: flex;
                gap: 10px;
                flex-wrap: wrap;
                margin: 10px 0;
            }
            """
        ) as interface:
            
            gr.Markdown(
                """
                # 🤖 Smart SDLC Assistant
                
                Your AI-powered companion for Software Development Lifecycle tasks.
                Ask me anything about requirements, design, development, testing, deployment, or maintenance!
                """,
                elem_classes=["header"]
            )
            
            with gr.Row():
                # Main chat area
                with gr.Column(scale=3):
                    chatbot = gr.Chatbot(
                        label="SDLC Assistant Chat",
                        height=400,
                        elem_classes=["chatbot-container"],
                        avatar_images=("👤", "🤖")
                    )
                    
                    with gr.Row():
                        msg_input = gr.Textbox(
                            placeholder="Ask me about any SDLC task...",
                            label="Your Message",
                            scale=4
                        )
                        send_btn = gr.Button("Send", variant="primary", scale=1)
                    
                    with gr.Row():
                        clear_btn = gr.Button("Clear Chat", variant="secondary")
                        export_btn = gr.Button("Export History", variant="secondary")
                
                # Sidebar with quick actions
                with gr.Column(scale=1):
                    gr.Markdown("### 🚀 Quick Actions")
                    
                    # SDLC Phase buttons
                    gr.Markdown("**SDLC Phases:**")
                    with gr.Column():
                        req_btn = gr.Button("📋 Requirements", size="sm")
                        design_btn = gr.Button("🎨 Design", size="sm")
                        dev_btn = gr.Button("💻 Development", size="sm")
                        test_btn = gr.Button("🧪 Testing", size="sm")
                        deploy_btn = gr.Button("🚀 Deployment", size="sm")
                        maintain_btn = gr.Button("🔧 Maintenance", size="sm")
                    
                    # Suggestions area
                    suggestions_area = gr.Textbox(
                        label="💡 Suggestions",
                        value="Welcome! Select a phase above or ask me anything about SDLC.",
                        lines=8,
                        interactive=False,
                        elem_classes=["suggestion-box"]
                    )
            
            # Example questions
            gr.Markdown(
                """
                ### 🎯 Example Questions:
                - "Help me write user stories for a shopping cart feature"
                - "Generate Python code for user authentication"
                - "Create unit tests for my calculator function"
                - "What are the best practices for API design?"
                - "How do I set up a CI/CD pipeline?"
                """
            )
            
            # Event handlers
            def handle_send(message, history):
                return self.chat_response(message, history)
            
            def handle_clear():
                return self.clear_chat()
            
            def handle_phase_click(phase):
                return self.get_phase_suggestions(phase)
            
            def export_history():
                if self.conversation_handler.conversation_history:
                    return json.dumps(self.conversation_handler.conversation_history, indent=2)
                return "No conversation history to export."
            
            # Wire up events
            send_btn.click(
                handle_send,
                inputs=[msg_input, chatbot],
                outputs=[msg_input, chatbot]
            )
            
            msg_input.submit(
                handle_send,
                inputs=[msg_input, chatbot],
                outputs=[msg_input, chatbot]
            )
            
            clear_btn.click(
                handle_clear,
                outputs=[chatbot, msg_input]
            )
            
            # Phase button clicks
            req_btn.click(lambda: handle_phase_click("requirements"), outputs=suggestions_area)
            design_btn.click(lambda: handle_phase_click("design"), outputs=suggestions_area)
            dev_btn.click(lambda: handle_phase_click("development"), outputs=suggestions_area)
            test_btn.click(lambda: handle_phase_click("testing"), outputs=suggestions_area)
            deploy_btn.click(lambda: handle_phase_click("deployment"), outputs=suggestions_area)
            maintain_btn.click(lambda: handle_phase_click("maintenance"), outputs=suggestions_area)
            
            export_btn.click(
                export_history,
                outputs=gr.Textbox(label="Exported History", lines=10)
            )
        
        return interface