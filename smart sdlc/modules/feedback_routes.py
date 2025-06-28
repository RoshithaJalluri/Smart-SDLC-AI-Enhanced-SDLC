# feedback_routes.py
import gradio as gr
from feedback_model import FeedbackModel, FeedbackData
import uuid
from datetime import datetime
import json

class FeedbackHandler:
    """Handler class for feedback operations in Gradio interface"""
    
    def __init__(self):
        self.feedback_model = FeedbackModel()
        self.current_session = str(uuid.uuid4())
    
    def collect_feedback(self, 
                        user_id: str,
                        module_name: str, 
                        input_data: str, 
                        ai_output: str, 
                        rating: int, 
                        feedback_text: str, 
                        feedback_type: str,
                        improvement_suggestions: str = None) -> str:
        """Collect and save user feedback"""
        
        try:
            feedback = FeedbackData(
                feedback_id=str(uuid.uuid4()),
                user_id=user_id or "anonymous",
                module_name=module_name,
                input_data=input_data,
                ai_output=ai_output,
                rating=rating,
                feedback_text=feedback_text,
                feedback_type=feedback_type,
                timestamp=datetime.now().isoformat(),
                session_id=self.current_session,
                improvement_suggestions=improvement_suggestions
            )
            
            success = self.feedback_model.save_feedback(feedback)
            
            if success:
                return "✅ Thank you for your feedback! It will help us improve the AI models."
            else:
                return "❌ Error saving feedback. Please try again."
                
        except Exception as e:
            return f"❌ Error: {str(e)}"
    
    def get_module_analytics(self, module_name: str) -> str:
        """Get analytics for a specific module"""
        try:
            feedback_data = self.feedback_model.get_feedback_by_module(module_name)
            
            if not feedback_data:
                return f"No feedback data available for {module_name}"
            
            total_feedback = len(feedback_data)
            avg_rating = sum(item['rating'] for item in feedback_data) / total_feedback
            
            positive_count = sum(1 for item in feedback_data if item['feedback_type'] == 'positive')
            negative_count = sum(1 for item in feedback_data if item['feedback_type'] == 'negative')
            suggestion_count = sum(1 for item in feedback_data if item['feedback_type'] == 'suggestion')
            
            analytics = f"""
📊 **Feedback Analytics for {module_name}**

📈 **Overall Statistics:**
- Total Feedback: {total_feedback}
- Average Rating: {avg_rating:.2f}/5.0 ⭐
- Positive Feedback: {positive_count} ({positive_count/total_feedback*100:.1f}%)
- Negative Feedback: {negative_count} ({negative_count/total_feedback*100:.1f}%)
- Suggestions: {suggestion_count} ({suggestion_count/total_feedback*100:.1f}%)

📝 **Recent Feedback:**
"""
            
            # Add recent feedback (last 3)
            for item in feedback_data[:3]:
                analytics += f"\n- **Rating:** {item['rating']}/5 | **Type:** {item['feedback_type']}\n"
                analytics += f"  **Feedback:** {item['feedback_text'][:100]}...\n"
            
            return analytics
            
        except Exception as e:
            return f"Error retrieving analytics: {str(e)}"
    
    def export_feedback_data(self, module_name: str = None) -> str:
        """Export feedback data for analysis"""
        try:
            if module_name:
                feedback_data = self.feedback_model.get_feedback_by_module(module_name)
                filename = f"feedback_export_{module_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            else:
                # Export all feedback (you'd need to implement this in FeedbackModel)
                feedback_data = []
                filename = f"feedback_export_all_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            
            with open(filename, 'w') as f:
                json.dump(feedback_data, f, indent=2)
            
            return f"✅ Feedback data exported to {filename}"
            
        except Exception as e:
            return f"❌ Error exporting data: {str(e)}"

# Gradio Interface Components
def create_feedback_interface():
    """Create Gradio interface for feedback collection"""
    
    feedback_handler = FeedbackHandler()
    
    with gr.Blocks(title="Smart SDLC - Feedback System") as feedback_interface:
        
        gr.Markdown("# 📋 Smart SDLC Feedback System")
        gr.Markdown("Help us improve our AI models by providing feedback on generated outputs!")
        
        with gr.Tab("Submit Feedback"):
            with gr.Row():
                with gr.Column():
                    user_id_input = gr.Textbox(label="User ID (Optional)", placeholder="Enter your user ID")
                    module_select = gr.Dropdown(
                        choices=["code_generator", "bug_resolver", "doc_generator", "ai_story_generator", "test_generator"],
                        label="Module Name",
                        value="code_generator"
                    )
                    
                with gr.Column():
                    rating_slider = gr.Slider(minimum=1, maximum=5, step=1, label="Rating (1-5 stars)", value=3)
                    feedback_type_radio = gr.Radio(
                        choices=["positive", "negative", "suggestion"],
                        label="Feedback Type",
                        value="positive"
                    )
            
            input_data_text = gr.Textbox(
                label="Original Input",
                placeholder="Paste the original input you provided to the AI...",
                lines=3
            )
            
            ai_output_text = gr.Textbox(
                label="AI Generated Output",
                placeholder="Paste the AI-generated output...",
                lines=5
            )
            
            feedback_text_area = gr.Textbox(
                label="Your Feedback",
                placeholder="Please provide detailed feedback about the AI output...",
                lines=4
            )
            
            improvement_text = gr.Textbox(
                label="Improvement Suggestions (Optional)",
                placeholder="Any specific suggestions for improvement?",
                lines=2
            )
            
            submit_btn = gr.Button("Submit Feedback", variant="primary")
            feedback_result = gr.Textbox(label="Result", interactive=False)
            
            submit_btn.click(
                feedback_handler.collect_feedback,
                inputs=[user_id_input, module_select, input_data_text, ai_output_text, 
                       rating_slider, feedback_text_area, feedback_type_radio, improvement_text],
                outputs=feedback_result
            )
        
        with gr.Tab("Analytics Dashboard"):
            analytics_module_select = gr.Dropdown(
                choices=["code_generator", "bug_resolver", "doc_generator", "ai_story_generator", "test_generator"],
                label="Select Module for Analytics",
                value="code_generator"
            )
            
            analytics_btn = gr.Button("Get Analytics")
            analytics_output = gr.Markdown()
            
            analytics_btn.click(
                feedback_handler.get_module_analytics,
                inputs=analytics_module_select,
                outputs=analytics_output
            )
            
            export_btn = gr.Button("Export Feedback Data")
            export_result = gr.Textbox(label="Export Result", interactive=False)
            
            export_btn.click(
                feedback_handler.export_feedback_data,
                inputs=analytics_module_select,
                outputs=export_result
            )
    
    return feedback_interface

