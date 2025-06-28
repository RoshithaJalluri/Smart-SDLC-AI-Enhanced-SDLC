# feedback_model.py
import json
import sqlite3
from datetime import datetime
from typing import Dict, List, Optional, Union
import pandas as pd
from dataclasses import dataclass, asdict
import uuid

@dataclass
class FeedbackData:
    """Data class for storing feedback information"""
    feedback_id: str
    user_id: str
    module_name: str  # e.g., 'code_generator', 'bug_resolver', 'doc_generator'
    input_data: str   # Original user input
    ai_output: str    # AI-generated output
    rating: int       # 1-5 star rating
    feedback_text: str  # User's written feedback
    feedback_type: str  # 'positive', 'negative', 'suggestion'
    timestamp: str
    session_id: str
    improvement_suggestions: Optional[str] = None

class FeedbackModel:
    """Model class for handling feedback data operations"""
    
    def __init__(self, db_path: str = "smart_sdlc_feedback.db"):
        self.db_path = db_path
        self.init_database()
    
    def init_database(self):
        """Initialize SQLite database for feedback storage"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS feedback (
                feedback_id TEXT PRIMARY KEY,
                user_id TEXT NOT NULL,
                module_name TEXT NOT NULL,
                input_data TEXT NOT NULL,
                ai_output TEXT NOT NULL,
                rating INTEGER NOT NULL CHECK (rating >= 1 AND rating <= 5),
                feedback_text TEXT,
                feedback_type TEXT CHECK (feedback_type IN ('positive', 'negative', 'suggestion')),
                timestamp TEXT NOT NULL,
                session_id TEXT NOT NULL,
                improvement_suggestions TEXT
            )
        ''')
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS feedback_analytics (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                module_name TEXT NOT NULL,
                avg_rating REAL,
                total_feedback INTEGER,
                positive_count INTEGER,
                negative_count INTEGER,
                suggestion_count INTEGER,
                last_updated TEXT
            )
        ''')
        
        conn.commit()
        conn.close()
    
    def save_feedback(self, feedback: FeedbackData) -> bool:
        """Save feedback to database"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT INTO feedback VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                feedback.feedback_id,
                feedback.user_id,
                feedback.module_name,
                feedback.input_data,
                feedback.ai_output,
                feedback.rating,
                feedback.feedback_text,
                feedback.feedback_type,
                feedback.timestamp,
                feedback.session_id,
                feedback.improvement_suggestions
            ))
            
            conn.commit()
            conn.close()
            
            # Update analytics
            self.update_analytics(feedback.module_name)
            return True
            
        except Exception as e:
            print(f"Error saving feedback: {e}")
            return False
    
    def get_feedback_by_module(self, module_name: str) -> List[Dict]:
        """Retrieve feedback for a specific module"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT * FROM feedback WHERE module_name = ?
            ORDER BY timestamp DESC
        ''', (module_name,))
        
        results = cursor.fetchall()
        conn.close()
        
        columns = ['feedback_id', 'user_id', 'module_name', 'input_data', 
                  'ai_output', 'rating', 'feedback_text', 'feedback_type', 
                  'timestamp', 'session_id', 'improvement_suggestions']
        
        return [dict(zip(columns, row)) for row in results]
    
    def update_analytics(self, module_name: str):
        """Update analytics for a module"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Get statistics
        cursor.execute('''
            SELECT 
                AVG(rating) as avg_rating,
                COUNT(*) as total_feedback,
                SUM(CASE WHEN feedback_type = 'positive' THEN 1 ELSE 0 END) as positive_count,
                SUM(CASE WHEN feedback_type = 'negative' THEN 1 ELSE 0 END) as negative_count,
                SUM(CASE WHEN feedback_type = 'suggestion' THEN 1 ELSE 0 END) as suggestion_count
            FROM feedback WHERE module_name = ?
        ''', (module_name,))
        
        stats = cursor.fetchone()
        
        # Update or insert analytics
        cursor.execute('''
            INSERT OR REPLACE INTO feedback_analytics 
            (module_name, avg_rating, total_feedback, positive_count, negative_count, suggestion_count, last_updated)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        ''', (module_name, stats[0], stats[1], stats[2], stats[3], stats[4], datetime.now().isoformat()))
        
        conn.commit()
        conn.close()