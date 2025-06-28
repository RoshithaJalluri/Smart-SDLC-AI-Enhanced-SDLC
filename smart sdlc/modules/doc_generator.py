import os
import re
import json
import logging
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
from datetime import datetime

# Core libraries
import PyPDF2
import fitz  # PyMuPDF - alternative PDF reader
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import torch

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class UserStory:
    """Data class for structured user stories"""
    id: str
    title: str
    description: str
    acceptance_criteria: List[str]
    priority: str
    story_points: int
    epic: str
    phase: str
    original_text: str
    confidence_score: float

@dataclass
class SDLCClassification:
    """Data class for SDLC phase classification results"""
    sentence: str
    phase: str
    confidence: float
    reasoning: str

class GraniteAIEngine:
    """Core AI engine using IBM Granite 3.3-2B Instruct model"""
    
    def __init__(self, model_name: str = "ibm-granite/granite-3.3-2b-instruct"):
        self.model_name = model_name
        self.tokenizer = None
        self.model = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.load_model()
    
    def load_model(self):
        """Load the IBM Granite model and tokenizer"""
        try:
            logger.info(f"Loading {self.model_name} on {self.device}")
            
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_name,
                trust_remote_code=True
            )
            
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
                device_map="auto" if self.device == "cuda" else None,
                trust_remote_code=True
            )
            
            # Set pad token if not present
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            logger.info("Model loaded successfully")
            
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            raise
    
    def generate_response(self, prompt: str, max_length: int = 512, temperature: float = 0.7) -> str:
        """Generate response using Granite model"""
        try:
            inputs = self.tokenizer.encode(prompt, return_tensors="pt").to(self.device)
            
            with torch.no_grad():
                outputs = self.model.generate(
                    inputs,
                    max_length=max_length,
                    temperature=temperature,
                    do_sample=True,
                    pad_token_id=self.tokenizer.eos_token_id,
                    attention_mask=torch.ones_like(inputs)
                )
            
            response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            # Remove the original prompt from response
            response = response[len(prompt):].strip()
            
            return response
            
        except Exception as e:
            logger.error(f"Error generating response: {str(e)}")
            return ""

class PDFProcessor:
    """Handles PDF text extraction with multiple fallback methods"""
    
    @staticmethod
    def extract_text_pypdf2(pdf_path: str) -> str:
        """Extract text using PyPDF2"""
        try:
            with open(pdf_path, 'rb') as file:
                reader = PyPDF2.PdfReader(file)
                text = ""
                for page in reader.pages:
                    text += page.extract_text() + "\n"
                return text
        except Exception as e:
            logger.warning(f"PyPDF2 extraction failed: {str(e)}")
            return ""
    
    @staticmethod
    def extract_text_pymupdf(pdf_path: str) -> str:
        """Extract text using PyMuPDF (fitz)"""
        try:
            doc = fitz.open(pdf_path)
            text = ""
            for page in doc:
                text += page.get_text() + "\n"
            doc.close()
            return text
        except Exception as e:
            logger.warning(f"PyMuPDF extraction failed: {str(e)}")
            return ""
    
    def extract_text(self, pdf_path: str) -> str:
        """Extract text with fallback methods"""
        if not os.path.exists(pdf_path):
            raise FileNotFoundError(f"PDF file not found: {pdf_path}")
        
        # Try PyMuPDF first (generally more reliable)
        text = self.extract_text_pymupdf(pdf_path)
        
        # Fallback to PyPDF2
        if not text.strip():
            text = self.extract_text_pypdf2(pdf_path)
        
        if not text.strip():
            raise ValueError("Could not extract text from PDF")
        
        return text

class SDLCPhaseClassifier:
    """Classifies sentences into SDLC phases using Granite AI"""
    
    def __init__(self, ai_engine: GraniteAIEngine):
        self.ai_engine = ai_engine
        self.sdlc_phases = {
            "requirements": "Requirements gathering, analysis, and specification",
            "design": "System design, architecture, and technical specifications",
            "development": "Coding, implementation, and programming activities",
            "testing": "Quality assurance, testing, and validation activities",
            "deployment": "Release, deployment, and production activities",
            "maintenance": "Support, maintenance, and post-deployment activities",
            "planning": "Project planning, estimation, and resource allocation",
            "other": "General or non-specific SDLC content"
        }
    
    def create_classification_prompt(self, sentence: str) -> str:
        """Create prompt for SDLC phase classification"""
        phases_desc = "\n".join([f"- {phase}: {desc}" for phase, desc in self.sdlc_phases.items()])
        
        prompt = f"""You are an expert software development lifecycle (SDLC) analyst. Classify the following sentence into one of these SDLC phases:

{phases_desc}

Sentence to classify: "{sentence}"

Analyze the sentence and respond in this exact JSON format:
{{
    "phase": "phase_name",
    "confidence": confidence_score_0_to_1,
    "reasoning": "brief explanation of why this phase was chosen"
}}

Classification:"""
        
        return prompt
    
    def classify_sentence(self, sentence: str) -> SDLCClassification:
        """Classify a single sentence into SDLC phase"""
        try:
            prompt = self.create_classification_prompt(sentence)
            response = self.ai_engine.generate_response(prompt, max_length=256, temperature=0.3)
            
            # Parse JSON response
            try:
                # Extract JSON from response
                json_match = re.search(r'\{.*\}', response, re.DOTALL)
                if json_match:
                    result = json.loads(json_match.group())
                    
                    return SDLCClassification(
                        sentence=sentence,
                        phase=result.get("phase", "other"),
                        confidence=float(result.get("confidence", 0.5)),
                        reasoning=result.get("reasoning", "No reasoning provided")
                    )
            except (json.JSONDecodeError, ValueError) as e:
                logger.warning(f"Failed to parse JSON response: {str(e)}")
            
            # Fallback classification
            return SDLCClassification(
                sentence=sentence,
                phase="other",
                confidence=0.3,
                reasoning="Failed to parse AI response"
            )
            
        except Exception as e:
            logger.error(f"Error classifying sentence: {str(e)}")
            return SDLCClassification(
                sentence=sentence,
                phase="other",
                confidence=0.1,
                reasoning=f"Classification error: {str(e)}"
            )

class UserStoryGenerator:
    """Generates structured user stories from classified requirements"""
    
    def __init__(self, ai_engine: GraniteAIEngine):
        self.ai_engine = ai_engine
    
    def create_story_prompt(self, text: str, phase: str) -> str:
        """Create prompt for user story generation"""
        prompt = f"""You are a product owner creating user stories from requirements. Convert the following {phase} text into a structured user story.

Text: "{text}"

Generate a user story following this format:
{{
    "title": "Short descriptive title",
    "description": "As a [user type], I want [functionality] so that [benefit]",
    "acceptance_criteria": ["criterion 1", "criterion 2", "criterion 3"],
    "priority": "High|Medium|Low",
    "story_points": story_point_estimate_1_to_13,
    "epic": "related epic or feature area"
}}

User Story:"""
        
        return prompt
    
    def generate_user_story(self, classification: SDLCClassification, story_id: str) -> UserStory:
        """Generate a user story from classified text"""
        try:
            # Only generate stories for requirements and design phases
            if classification.phase not in ["requirements", "design", "planning"]:
                return None
            
            prompt = self.create_story_prompt(classification.sentence, classification.phase)
            response = self.ai_engine.generate_response(prompt, max_length=400, temperature=0.6)
            
            # Parse JSON response
            try:
                json_match = re.search(r'\{.*\}', response, re.DOTALL)
                if json_match:
                    result = json.loads(json_match.group())
                    
                    return UserStory(
                        id=story_id,
                        title=result.get("title", f"Story from {classification.phase}"),
                        description=result.get("description", classification.sentence),
                        acceptance_criteria=result.get("acceptance_criteria", []),
                        priority=result.get("priority", "Medium"),
                        story_points=int(result.get("story_points", 3)),
                        epic=result.get("epic", "General"),
                        phase=classification.phase,
                        original_text=classification.sentence,
                        confidence_score=classification.confidence
                    )
            except (json.JSONDecodeError, ValueError) as e:
                logger.warning(f"Failed to parse user story JSON: {str(e)}")
            
            # Fallback user story creation
            return UserStory(
                id=story_id,
                title=f"Generated Story {story_id}",
                description=f"As a user, I want {classification.sentence[:100]}...",
                acceptance_criteria=["Acceptance criteria to be defined"],
                priority="Medium",
                story_points=3,
                epic="General",
                phase=classification.phase,
                original_text=classification.sentence,
                confidence_score=classification.confidence
            )
            
        except Exception as e:
            logger.error(f"Error generating user story: {str(e)}")
            return None

class RequirementsProcessor:
    """Main class that orchestrates the entire requirements processing pipeline"""
    
    def __init__(self):
        self.ai_engine = GraniteAIEngine()
        self.pdf_processor = PDFProcessor()
        self.phase_classifier = SDLCPhaseClassifier(self.ai_engine)
        self.story_generator = UserStoryGenerator(self.ai_engine)
    
    def preprocess_text(self, text: str) -> List[str]:
        """Clean and split text into sentences"""
        # Clean text
        text = re.sub(r'\s+', ' ', text)  # Normalize whitespace
        text = re.sub(r'\n+', '\n', text)  # Normalize line breaks
        
        # Split into sentences
        sentences = re.split(r'[.!?]+', text)
        
        # Filter and clean sentences
        cleaned_sentences = []
        for sentence in sentences:
            sentence = sentence.strip()
            if len(sentence) > 10 and len(sentence) < 500:  # Filter by length
                cleaned_sentences.append(sentence)
        
        return cleaned_sentences
    
    def process_pdf(self, pdf_path: str) -> Dict:
        """Main processing pipeline for PDF requirements"""
        try:
            logger.info(f"Processing PDF: {pdf_path}")
            
            # Step 1: Extract text from PDF
            logger.info("Extracting text from PDF...")
            raw_text = self.pdf_processor.extract_text(pdf_path)
            
            # Step 2: Preprocess text
            logger.info("Preprocessing text...")
            sentences = self.preprocess_text(raw_text)
            
            # Step 3: Classify sentences
            logger.info("Classifying sentences into SDLC phases...")
            classifications = []
            for i, sentence in enumerate(sentences):
                logger.info(f"Classifying sentence {i+1}/{len(sentences)}")
                classification = self.phase_classifier.classify_sentence(sentence)
                classifications.append(classification)
            
            # Step 4: Generate user stories
            logger.info("Generating user stories...")
            user_stories = []
            story_counter = 1
            
            for classification in classifications:
                if classification.confidence > 0.6:  # Only high-confidence classifications
                    story = self.story_generator.generate_user_story(
                        classification, 
                        f"US-{story_counter:03d}"
                    )
                    if story:
                        user_stories.append(story)
                        story_counter += 1
            
            # Step 5: Compile results
            results = {
                "pdf_path": pdf_path,
                "processing_timestamp": datetime.now().isoformat(),
                "total_sentences": len(sentences),
                "classifications": [
                    {
                        "sentence": c.sentence,
                        "phase": c.phase,
                        "confidence": c.confidence,
                        "reasoning": c.reasoning
                    }
                    for c in classifications
                ],
                "user_stories": [
                    {
                        "id": story.id,
                        "title": story.title,
                        "description": story.description,
                        "acceptance_criteria": story.acceptance_criteria,
                        "priority": story.priority,
                        "story_points": story.story_points,
                        "epic": story.epic,
                        "phase": story.phase,
                        "original_text": story.original_text,
                        "confidence_score": story.confidence_score
                    }
                    for story in user_stories
                ],
                "phase_summary": self._get_phase_summary(classifications),
                "statistics": {
                    "total_user_stories": len(user_stories),
                    "avg_confidence": sum(c.confidence for c in classifications) / len(classifications),
                    "phase_distribution": self._get_phase_distribution(classifications)
                }
            }
            
            logger.info(f"Processing complete. Generated {len(user_stories)} user stories.")
            return results
            
        except Exception as e:
            logger.error(f"Error processing PDF: {str(e)}")
            raise
    
    def _get_phase_summary(self, classifications: List[SDLCClassification]) -> Dict[str, int]:
        """Get summary of phases identified"""
        phase_counts = {}
        for classification in classifications:
            phase = classification.phase
            phase_counts[phase] = phase_counts.get(phase, 0) + 1
        return phase_counts
    
    def _get_phase_distribution(self, classifications: List[SDLCClassification]) -> Dict[str, float]:
        """Get percentage distribution of phases"""
        total = len(classifications)
        phase_counts = self._get_phase_summary(classifications)
        return {phase: (count / total) * 100 for phase, count in phase_counts.items()}
    
    def save_results(self, results: Dict, output_path: str):
        """Save processing results to JSON file"""
        try:
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(results, f, indent=2, ensure_ascii=False)
            logger.info(f"Results saved to: {output_path}")
        except Exception as e:
            logger.error(f"Error saving results: {str(e)}")
            raise

# Example usage and testing
if __name__ == "__main__":
    # Initialize the requirements processor
    processor = RequirementsProcessor()
    
    # Example usage
    try:
        # Process a PDF file
        pdf_path = "sample_requirements.pdf"  # Replace with actual PDF path
        results = processor.process_pdf(pdf_path)
        
        # Save results
        output_path = "requirements_analysis_results.json"
        processor.save_results(results, output_path)
        
        # Print summary
        print(f"Processing complete!")
        print(f"Total sentences processed: {results['total_sentences']}")
        print(f"User stories generated: {results['statistics']['total_user_stories']}")
        print(f"Average confidence: {results['statistics']['avg_confidence']:.2f}")
        print(f"Phase distribution: {results['statistics']['phase_distribution']}")
        
    except Exception as e:
        print(f"Error: {str(e)}")