# main_chatbot.py - Integration with the main application
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
import torch

class SmartSDLCChatbot:
    """
    Main chatbot class that integrates with the Smart SDLC application
    """
    
    def __init__(self):
        self.model_name = "ibm-granite/granite-3.3-2b-instruct"
        self.model_pipeline = None
        self.conversation_handler = None
        self.chat_interface = None
        self.setup_model()
    
    def setup_model(self):
        """
        Initialize the IBM Granite model
        """
        try:
            print("Loading IBM Granite model...")
            
            # Load tokenizer and model
            tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                device_map="auto" if torch.cuda.is_available() else None
            )
            
            # Create pipeline
            self.model_pipeline = pipeline(
                "text-generation",
                model=model,
                tokenizer=tokenizer,
                device=0 if torch.cuda.is_available() else -1
            )
            
            # Initialize conversation handler
            self.conversation_handler = ConversationHandler(self.model_pipeline)
            
            # Initialize chat interface
            self.chat_interface = ChatInterface(self.conversation_handler)
            
            print("✅ Chatbot initialized successfully!")
            
        except Exception as e:
            print(f"❌ Error initializing chatbot: {e}")
            # Fallback to a simple response system
            self.setup_fallback()
    
    def setup_fallback(self):
        """
        Setup fallback system if model loading fails
        """
        print("Setting up fallback response system...")
        
        class FallbackPipeline:
            def __call__(self, prompt, **kwargs):
                # Simple rule-based responses
                responses = {
                    "requirements": "I can help you with requirements analysis. Try uploading a document or describing your user stories.",
                    "design": "For design assistance, I can help with architecture patterns, database design, and API specifications.",
                    "development": "I can help generate code, review existing code, and suggest improvements.",
                    "testing": "I can help create test cases, suggest testing strategies, and identify potential bugs.",
                    "deployment": "I can assist with deployment strategies, CI/CD setup, and production considerations.",
                    "maintenance": "I can help with code refactoring, performance optimization, and documentation."
                }
                
                prompt_lower = prompt.lower()
                for key, response in responses.items():
                    if key in prompt_lower:
                        return [{"generated_text": f"{prompt}\nAssistant: {response}"}]
                
                return [{"generated_text": f"{prompt}\nAssistant: I'm here to help with your SDLC tasks. What specific area would you like assistance with?"}]
        
        self.model_pipeline = FallbackPipeline()
        self.conversation_handler = ConversationHandler(self.model_pipeline)
        self.chat_interface = ChatInterface(self.conversation_handler)
    
    def launch_interface(self, share=True, debug=True):
        """
        Launch the Gradio interface
        """
        if self.chat_interface:
            interface = self.chat_interface.create_interface()
            return interface.launch(share=share, debug=debug)
        else:
            print("❌ Chat interface not initialized")
            return None
    
    def get_response(self, message: str) -> str:
        """
        Get a response from the chatbot (for programmatic use)
        """
        if self.conversation_handler:
            return self.conversation_handler.generate_contextual_response(message)
        return "Chatbot not initialized"