# github_service.py
import os
import json
import base64
import requests
from datetime import datetime
from typing import Dict, List, Optional, Union
import gradio as gr
from dataclasses import dataclass
import zipfile
import tempfile

@dataclass
class GitHubConfig:
    """Configuration for GitHub integration"""
    token: str
    username: str
    repo_name: str
    base_url: str = "https://api.github.com"

class GitHubService:
    """Service class for GitHub API interactions"""
    
    def __init__(self, config: GitHubConfig):
        self.config = config
        self.headers = {
            "Authorization": f"token {config.token}",
            "Accept": "application/vnd.github.v3+json",
            "Content-Type": "application/json"
        }
    
    def test_connection(self) -> Dict[str, Union[bool, str]]:
        """Test GitHub API connection and token validity"""
        try:
            response = requests.get(
                f"{self.config.base_url}/user",
                headers=self.headers
            )
            
            if response.status_code == 200:
                user_data = response.json()
                return {
                    "success": True,
                    "message": f"✅ Connected as {user_data.get('login', 'Unknown')}",
                    "user_info": user_data
                }
            else:
                return {
                    "success": False,
                    "message": f"❌ Authentication failed: {response.status_code}"
                }
        except Exception as e:
            return {
                "success": False,
                "message": f"❌ Connection error: {str(e)}"
            }
    
    def create_repository(self, repo_name: str, description: str = "", private: bool = False) -> Dict:
        """Create a new GitHub repository"""
        try:
            data = {
                "name": repo_name,
                "description": description,
                "private": private,
                "auto_init": True,
                "license_template": "mit"
            }
            
            response = requests.post(
                f"{self.config.base_url}/user/repos",
                headers=self.headers,
                json=data
            )
            
            if response.status_code == 201:
                repo_data = response.json()
                return {
                    "success": True,
                    "message": f"✅ Repository '{repo_name}' created successfully!",
                    "repo_url": repo_data["html_url"],
                    "clone_url": repo_data["clone_url"]
                }
            else:
                return {
                    "success": False,
                    "message": f"❌ Failed to create repository: {response.json().get('message', 'Unknown error')}"
                }
        except Exception as e:
            return {
                "success": False,
                "message": f"❌ Error creating repository: {str(e)}"
            }
    
    def upload_file(self, file_path: str, content: str, commit_message: str, branch: str = "main") -> Dict:
        """Upload or update a file in the repository"""
        try:
            # Check if file exists to get SHA for update
            existing_file_response = requests.get(
                f"{self.config.base_url}/repos/{self.config.username}/{self.config.repo_name}/contents/{file_path}",
                headers=self.headers
            )
            
            # Encode content to base64
            encoded_content = base64.b64encode(content.encode('utf-8')).decode('utf-8')
            
            data = {
                "message": commit_message,
                "content": encoded_content,
                "branch": branch
            }
            
            # If file exists, include SHA for update
            if existing_file_response.status_code == 200:
                existing_data = existing_file_response.json()
                data["sha"] = existing_data["sha"]
                action = "updated"
            else:
                action = "created"
            
            response = requests.put(
                f"{self.config.base_url}/repos/{self.config.username}/{self.config.repo_name}/contents/{file_path}",
                headers=self.headers,
                json=data
            )
            
            if response.status_code in [200, 201]:
                return {
                    "success": True,
                    "message": f"✅ File '{file_path}' {action} successfully!",
                    "commit_url": response.json()["commit"]["html_url"]
                }
            else:
                return {
                    "success": False,
                    "message": f"❌ Failed to upload file: {response.json().get('message', 'Unknown error')}"
                }
        except Exception as e:
            return {
                "success": False,
                "message": f"❌ Error uploading file: {str(e)}"
            }
    
    def create_issue(self, title: str, body: str, labels: List[str] = None, assignees: List[str] = None) -> Dict:
        """Create a new GitHub issue"""
        try:
            data = {
                "title": title,
                "body": body
            }
            
            if labels:
                data["labels"] = labels
            if assignees:
                data["assignees"] = assignees
            
            response = requests.post(
                f"{self.config.base_url}/repos/{self.config.username}/{self.config.repo_name}/issues",
                headers=self.headers,
                json=data
            )
            
            if response.status_code == 201:
                issue_data = response.json()
                return {
                    "success": True,
                    "message": f"✅ Issue created successfully! #{issue_data['number']}",
                    "issue_url": issue_data["html_url"],
                    "issue_number": issue_data["number"]
                }
            else:
                return {
                    "success": False,
                    "message": f"❌ Failed to create issue: {response.json().get('message', 'Unknown error')}"
                }
        except Exception as e:
            return {
                "success": False,
                "message": f"❌ Error creating issue: {str(e)}"
            }
    
    def create_pull_request(self, title: str, body: str, head: str, base: str = "main") -> Dict:
        """Create a pull request"""
        try:
            data = {
                "title": title,
                "body": body,
                "head": head,
                "base": base
            }
            
            response = requests.post(
                f"{self.config.base_url}/repos/{self.config.username}/{self.config.repo_name}/pulls",
                headers=self.headers,
                json=data
            )
            
            if response.status_code == 201:
                pr_data = response.json()
                return {
                    "success": True,
                    "message": f"✅ Pull request created successfully! #{pr_data['number']}",
                    "pr_url": pr_data["html_url"],
                    "pr_number": pr_data["number"]
                }
            else:
                return {
                    "success": False,
                    "message": f"❌ Failed to create pull request: {response.json().get('message', 'Unknown error')}"
                }
        except Exception as e:
            return {
                "success": False,
                "message": f"❌ Error creating pull request: {str(e)}"
            }
    
    def create_branch(self, branch_name: str, source_branch: str = "main") -> Dict:
        """Create a new branch"""
        try:
            # Get the SHA of the source branch
            ref_response = requests.get(
                f"{self.config.base_url}/repos/{self.config.username}/{self.config.repo_name}/git/refs/heads/{source_branch}",
                headers=self.headers
            )
            
            if ref_response.status_code != 200:
                return {
                    "success": False,
                    "message": f"❌ Source branch '{source_branch}' not found"
                }
            
            source_sha = ref_response.json()["object"]["sha"]
            
            # Create new branch
            data = {
                "ref": f"refs/heads/{branch_name}",
                "sha": source_sha
            }
            
            response = requests.post(
                f"{self.config.base_url}/repos/{self.config.username}/{self.config.repo_name}/git/refs",
                headers=self.headers,
                json=data
            )
            
            if response.status_code == 201:
                return {
                    "success": True,
                    "message": f"✅ Branch '{branch_name}' created successfully!"
                }
            else:
                return {
                    "success": False,
                    "message": f"❌ Failed to create branch: {response.json().get('message', 'Unknown error')}"
                }
        except Exception as e:
            return {
                "success": False,
                "message": f"❌ Error creating branch: {str(e)}"
            }
    
    def upload_multiple_files(self, files_data: Dict[str, str], commit_message: str, branch: str = "main") -> Dict:
        """Upload multiple files in a single operation"""
        results = []
        successful_uploads = 0
        
        for file_path, content in files_data.items():
            result = self.upload_file(file_path, content, f"{commit_message} - {file_path}", branch)
            results.append(f"📁 {file_path}: {result['message']}")
            if result['success']:
                successful_uploads += 1
        
        return {
            "success": successful_uploads > 0,
            "message": f"✅ {successful_uploads}/{len(files_data)} files uploaded successfully!",
            "details": "\n".join(results)
        }

class SmartSDLCGitHubIntegration:
    """Main integration class for Smart SDLC GitHub workflows"""
    
    def __init__(self):
        self.github_service = None
        self.config = None
    
    def setup_github_connection(self, token: str, username: str, repo_name: str) -> str:
        """Setup GitHub connection with provided credentials"""
        try:
            self.config = GitHubConfig(token=token, username=username, repo_name=repo_name)
            self.github_service = GitHubService(self.config)
            
            # Test connection
            test_result = self.github_service.test_connection()
            return test_result["message"]
        except Exception as e:
            return f"❌ Setup failed: {str(e)}"
    
    def push_generated_code(self, code_content: str, filename: str, module_type: str, description: str = "") -> str:
        """Push AI-generated code to GitHub repository"""
        if not self.github_service:
            return "❌ Please setup GitHub connection first!"
        
        try:
            # Create appropriate folder structure
            folder_mapping = {
                "code_generator": "src/generated_code",
                "bug_resolver": "src/fixed_code", 
                "doc_generator": "docs",
                "ai_story_generator": "requirements",
                "test_generator": "tests"
            }
            
            folder = folder_mapping.get(module_type, "src/misc")
            file_path = f"{folder}/{filename}"
            
            # Add metadata comment
            metadata = f"""\"\"\"
Generated by Smart SDLC - {module_type}
Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
Description: {description}
\"\"\"

{code_content}
"""
            
            commit_message = f"Add {module_type} generated code: {filename}"
            result = self.github_service.upload_file(file_path, metadata, commit_message)
            
            return result["message"]
        except Exception as e:
            return f"❌ Error pushing code: {str(e)}"
    
    def create_project_structure(self, project_name: str) -> str:
        """Create complete project structure for Smart SDLC project"""
        if not self.github_service:
            return "❌ Please setup GitHub connection first!"
        
        try:
            # Define project structure
            project_files = {
                "README.md": f"""# {project_name}

## Smart SDLC Generated Project

This project was generated using Smart SDLC - AI Enhanced Software Development Lifecycle.

### Project Structure
```
├── src/
│   ├── generated_code/    # AI-generated code
│   ├── fixed_code/        # Bug-fixed code
│   └── main.py           # Main application
├── tests/                 # Generated test cases
├── docs/                  # Generated documentation
├── requirements/          # User stories and requirements
└── requirements.txt       # Dependencies
```

### Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
""",
                "src/main.py": """# Main application file
# Generated by Smart SDLC

def main():
    print("Hello from Smart SDLC generated project!")

if __name__ == "__main__":
    main()
""",
                "requirements.txt": """# Dependencies for Smart SDLC generated project
requests>=2.25.1
gradio>=3.0.0
transformers>=4.20.0
""",
                "tests/__init__.py": "# Test package",
                "docs/README.md": "# Documentation\n\nThis folder contains generated documentation.",
                "requirements/user_stories.md": "# User Stories\n\nGenerated user stories will be stored here.",
                ".gitignore": """# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
env/
venv/
.venv/
.env

# IDE
.vscode/
.idea/
""",
                "src/generated_code/README.md": "# Generated Code\n\nAI-generated code files will be stored here.",
                "src/fixed_code/README.md": "# Fixed Code\n\nBug-fixed code files will be stored here."
            }
            
            result = self.github_service.upload_multiple_files(
                project_files, 
                f"Initialize Smart SDLC project: {project_name}"
            )
            
            return result["message"] + "\n\n" + result.get("details", "")
        except Exception as e:
            return f"❌ Error creating project structure: {str(e)}"
    
    def create_bug_report_issue(self, bug_description: str, code_snippet: str, error_message: str, module_name: str) -> str:
        """Create GitHub issue for bug reports"""
        if not self.github_service:
            return "❌ Please setup GitHub connection first!"
        
        try:
            title = f"Bug Report: {module_name} - {bug_description[:50]}..."
            body = f"""## Bug Report from Smart SDLC

**Module:** {module_name}
**Date:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

### Description
{bug_description}

### Error Message
```
{error_message}
```

### Code Snippet
```python
{code_snippet}
```

### Additional Information
- Generated by Smart SDLC Bug Resolver
- Please review and provide feedback for model improvement

---
*This issue was automatically created by Smart SDLC*
"""
            
            result = self.github_service.create_issue(
                title=title,
                body=body,
                labels=["bug", "smart-sdlc", "ai-generated"]
            )
            
            return result["message"]
        except Exception as e:
            return f"❌ Error creating bug report: {str(e)}"
    
    def sync_documentation(self, documentation: str, doc_type: str, filename: str) -> str:
        """Sync generated documentation to GitHub"""
        if not self.github_service:
            return "❌ Please setup GitHub connection first!"
        
        try:
            file_path = f"docs/{doc_type}/{filename}"
            
            # Add documentation metadata
            doc_with_metadata = f"""# {doc_type.title()} Documentation

> Generated by Smart SDLC Documentation Generator  
> Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

{documentation}

---
*This documentation was automatically generated by Smart SDLC*
"""
            
            commit_message = f"Update {doc_type} documentation: {filename}"
            result = self.github_service.upload_file(file_path, doc_with_metadata, commit_message)
            
            return result["message"]
        except Exception as e:
            return f"❌ Error syncing documentation: {str(e)}"
    
    def create_feature_request(self, feature_title: str, feature_description: str, user_stories: str) -> str:
        """Create feature request issue from user stories"""
        if not self.github_service:
            return "❌ Please setup GitHub connection first!"
        
        try:
            title = f"Feature Request: {feature_title}"
            body = f"""## Feature Request from Smart SDLC

**Date:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

### Description
{feature_description}

### User Stories
{user_stories}

### Implementation Notes
- Generated by Smart SDLC AI Story Generator
- Review user stories for completeness
- Consider for next sprint planning

---
*This feature request was automatically created by Smart SDLC*
"""
            
            result = self.github_service.create_issue(
                title=title,
                body=body,
                labels=["enhancement", "feature-request", "smart-sdlc"]
            )
            
            return result["message"]
        except Exception as e:
            return f"❌ Error creating feature request: {str(e)}"

# Gradio Interface
def create_github_integration_interface():
    """Create Gradio interface for GitHub integration"""
    
    github_integration = SmartSDLCGitHubIntegration()
    
    with gr.Blocks(title="Smart SDLC - GitHub Integration") as github_interface:
        
        gr.Markdown("# 🐙 Smart SDLC GitHub Integration")
        gr.Markdown("Automate your GitHub workflows with AI-generated content!")
        
        with gr.Tab("🔧 Setup Connection"):
            gr.Markdown("### Configure GitHub Connection")
            
            with gr.Row():
                token_input = gr.Textbox(
                    label="GitHub Personal Access Token",
                    type="password",
                    placeholder="ghp_xxxxxxxxxxxx",
                    info="Create token at: Settings > Developer settings > Personal access tokens"
                )
                username_input = gr.Textbox(
                    label="GitHub Username",
                    placeholder="your-username"
                )
                repo_input = gr.Textbox(
                    label="Repository Name",
                    placeholder="my-smart-sdlc-project"
                )
            
            setup_btn = gr.Button("🔗 Setup Connection", variant="primary")
            setup_result = gr.Textbox(label="Connection Status", interactive=False)
            
            setup_btn.click(
                github_integration.setup_github_connection,
                inputs=[token_input, username_input, repo_input],
                outputs=setup_result
            )
        
        with gr.Tab("📁 Push Generated Code"):
            gr.Markdown("### Push AI-Generated Code to Repository")
            
            with gr.Row():
                module_select = gr.Dropdown(
                    choices=["code_generator", "bug_resolver", "doc_generator", "ai_story_generator", "test_generator"],
                    label="Source Module",
                    value="code_generator"
                )
                filename_input = gr.Textbox(
                    label="Filename",
                    placeholder="main.py",
                    value="main.py"
                )
            
            code_content = gr.Textbox(
                label="Generated Code",
                placeholder="Paste your AI-generated code here...",
                lines=10
            )
            
            description_input = gr.Textbox(
                label="Description",
                placeholder="Brief description of the generated code...",
                lines=2
            )
            
            push_btn = gr.Button("🚀 Push to GitHub", variant="primary")
            push_result = gr.Textbox(label="Push Result", interactive=False)
            
            push_btn.click(
                github_integration.push_generated_code,
                inputs=[code_content, filename_input, module_select, description_input],
                outputs=push_result
            )
        
        with gr.Tab("🏗️ Create Project Structure"):
            gr.Markdown("### Create Complete Project Structure")
            
            project_name_input = gr.Textbox(
                label="Project Name",
                placeholder="My Smart SDLC Project",
                value="Smart SDLC Project"
            )
            
            create_structure_btn = gr.Button("🏗️ Create Project Structure", variant="primary")
            structure_result = gr.Textbox(label="Creation Result", interactive=False, lines=10)
            
            create_structure_btn.click(
                github_integration.create_project_structure,
                inputs=project_name_input,
                outputs=structure_result
            )
        
        with gr.Tab("🐛 Bug Reports"):
            gr.Markdown("### Create Bug Report Issues")
            
            bug_description = gr.Textbox(
                label="Bug Description",
                placeholder="Describe the bug or issue...",
                lines=3
            )
            
            bug_code = gr.Textbox(
                label="Problematic Code",
                placeholder="Paste the code that has issues...",
                lines=5
            )
            
            error_message = gr.Textbox(
                label="Error Message",
                placeholder="Paste the error message...",
                lines=3
            )
            
            bug_module = gr.Dropdown(
                choices=["code_generator", "bug_resolver", "doc_generator", "ai_story_generator", "test_generator"],
                label="Module",
                value="code_generator"
            )
            
            bug_report_btn = gr.Button("🐛 Create Bug Report", variant="primary")
            bug_result = gr.Textbox(label="Bug Report Result", interactive=False)
            
            bug_report_btn.click(
                github_integration.create_bug_report_issue,
                inputs=[bug_description, bug_code, error_message, bug_module],
                outputs=bug_result
            )
        
        with gr.Tab("📚 Sync Documentation"):
            gr.Markdown("### Sync Generated Documentation")
            
            with gr.Row():
                doc_type_select = gr.Dropdown(
                    choices=["api", "code", "user_guide", "technical", "readme"],
                    label="Documentation Type",
                    value="api"
                )
                doc_filename = gr.Textbox(
                    label="Documentation Filename",
                    placeholder="api_documentation.md",
                    value="documentation.md"
                )
            
            documentation_content = gr.Textbox(
                label="Documentation Content",
                placeholder="Paste your generated documentation here...",
                lines=10
            )
            
            sync_doc_btn = gr.Button("📚 Sync Documentation", variant="primary")
            sync_result = gr.Textbox(label="Sync Result", interactive=False)
            
            sync_doc_btn.click(
                github_integration.sync_documentation,
                inputs=[documentation_content, doc_type_select, doc_filename],
                outputs=sync_result
            )
        
        with gr.Tab("💡 Feature Requests"):
            gr.Markdown("### Create Feature Requests from User Stories")
            
            feature_title = gr.Textbox(
                label="Feature Title",
                placeholder="New Feature Name"
            )
            
            feature_desc = gr.Textbox(
                label="Feature Description",
                placeholder="Describe the feature...",
                lines=4
            )
            
            user_stories = gr.Textbox(
                label="User Stories",
                placeholder="Paste user stories generated by AI Story Generator...",
                lines=6
            )
            
            feature_btn = gr.Button("💡 Create Feature Request", variant="primary")
            feature_result = gr.Textbox(label="Feature Request Result", interactive=False)
            
            feature_btn.click(
                github_integration.create_feature_request,
                inputs=[feature_title, feature_desc, user_stories],
                outputs=feature_result
            )
    
    return github_interface