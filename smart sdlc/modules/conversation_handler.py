# conversation_handler.py
import json
import logging
from typing import Dict, List, Optional, Tuple
from datetime import datetime
import re

class ConversationHandler:
    """
    Handles conversation flow, context management, and SDLC-specific assistance
    """
    
    def __init__(self, model_pipeline):
        self.model_pipeline = model_pipeline
        self.conversation_history = []
        self.context = {
            "current_phase": None,
            "project_context": {},
            "user_preferences": {},
            "active_tasks": []
        }
        self.sdlc_phases = [
            "requirements", "design", "development", 
            "testing", "deployment", "maintenance"
        ]
        
        # SDLC-specific prompts and responses
        self.sdlc_prompts = {
            "requirements": {
                "keywords": ["requirement", "user story", "functional", "non-functional", "acceptance criteria"],
                "suggestions": [
                    "Would you like me to help analyze requirements from a document?",
                    "I can help generate user stories from your requirements.",
                    "Need assistance with acceptance criteria definition?"
                ]
            },
            "design": {
                "keywords": ["architecture", "design pattern", "UML", "database", "API"],
                "suggestions": [
                    "I can help with system architecture recommendations.",
                    "Would you like suggestions for design patterns?",
                    "Need help with database schema design?"
                ]
            },
            "development": {
                "keywords": ["code", "programming", "function", "class", "algorithm"],
                "suggestions": [
                    "I can generate code based on your requirements.",
                    "Need help with specific programming problems?",
                    "Would you like me to review your code structure?"
                ]
            },
            "testing": {
                "keywords": ["test", "unit test", "integration", "bug", "quality"],
                "suggestions": [
                    "I can generate test cases for your code.",
                    "Need help with test automation strategies?",
                    "Would you like assistance with bug analysis?"
                ]
            },
            "deployment": {
                "keywords": ["deploy", "CI/CD", "docker", "kubernetes", "production"],
                "suggestions": [
                    "I can help with deployment strategies.",
                    "Need assistance with CI/CD pipeline setup?",
                    "Would you like help with containerization?"
                ]
            },
            "maintenance": {
                "keywords": ["maintenance", "refactor", "optimization", "documentation"],
                "suggestions": [
                    "I can help with code refactoring suggestions.",
                    "Need assistance with performance optimization?",
                    "Would you like help generating documentation?"
                ]
            }
        }
    
    def detect_sdlc_phase(self, message: str) -> Optional[str]:
        """
        Detect which SDLC phase the user is asking about
        """
        message_lower = message.lower()
        phase_scores = {}
        
        for phase, data in self.sdlc_prompts.items():
            score = 0
            for keyword in data["keywords"]:
                if keyword in message_lower:
                    score += 1
            phase_scores[phase] = score
        
        # Return phase with highest score, or None if no matches
        max_score = max(phase_scores.values())
        if max_score > 0:
            return max(phase_scores, key=phase_scores.get)
        return None
    
    def generate_contextual_response(self, user_message: str) -> str:
        """
        Generate AI response with SDLC context awareness
        """
        # Detect SDLC phase
        detected_phase = self.detect_sdlc_phase(user_message)
        if detected_phase:
            self.context["current_phase"] = detected_phase
        
        # Build context-aware prompt
        system_prompt = self.build_system_prompt(detected_phase)
        full_prompt = f"{system_prompt}\n\nUser: {user_message}\nAssistant:"
        
        # Generate response using the model
        try:
            response = self.model_pipeline(
                full_prompt,
                max_length=512,
                temperature=0.7,
                do_sample=True,
                pad_token_id=self.model_pipeline.tokenizer.eos_token_id
            )[0]['generated_text']
            
            # Extract only the assistant's response
            response = response.split("Assistant:")[-1].strip()
            
            # Add contextual suggestions if appropriate
            if detected_phase and len(response) < 100:
                suggestions = self.sdlc_prompts[detected_phase]["suggestions"]
                response += f"\n\n💡 {suggestions[0]}"
            
            return response
            
        except Exception as e:
            logging.error(f"Error generating response: {e}")
            return "I apologize, but I encountered an error. Please try rephrasing your question."
    
    def build_system_prompt(self, phase: Optional[str] = None) -> str:
        """
        Build system prompt based on current context and SDLC phase
        """
        base_prompt = """You are an AI assistant specialized in Software Development Lifecycle (SDLC). 
You help developers with requirements analysis, design, coding, testing, deployment, and maintenance tasks.
Provide practical, actionable advice and be concise but helpful."""
        
        if phase:
            phase_context = f"\nCurrently focusing on: {phase.upper()} phase of SDLC."
            base_prompt += phase_context
        
        if self.context["project_context"]:
            project_info = f"\nProject context: {self.context['project_context']}"
            base_prompt += project_info
        
        return base_prompt
    
    def add_to_history(self, user_message: str, ai_response: str):
        """
        Add conversation turn to history
        """
        self.conversation_history.append({
            "timestamp": datetime.now().isoformat(),
            "user_message": user_message,
            "ai_response": ai_response,
            "context": self.context.copy()
        })
        
        # Keep only last 10 conversations to manage memory
        if len(self.conversation_history) > 10:
            self.conversation_history = self.conversation_history[-10:]
    
    def get_quick_suggestions(self, phase: str = None) -> List[str]:
        """
        Get quick suggestions based on current or specified phase
        """
        target_phase = phase or self.context.get("current_phase", "development")
        if target_phase in self.sdlc_prompts:
            return self.sdlc_prompts[target_phase]["suggestions"]
        return [
            "How can I help with your development task?",
            "Would you like code generation assistance?",
            "Need help with testing or documentation?"
        ]