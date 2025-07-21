#!/usr/bin/env python3
"""
Simple Data Cleaner - Streamlit App
Upload CSV, clean data, download results.
"""

import streamlit as st
import pandas as pd
import numpy as np
import json
import os


# Inline DataCleaner class to avoid import issues
class DataCleaner:
    """Simple data cleaner with optional LLM support."""
    
    from typing import Optional

    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key
        self.use_ai = False
        self.cleaning_log = []
        
        # Try to initialize OpenAI if API key provided
        if api_key:
            try:
                import openai
                self.client = openai.OpenAI(api_key=api_key)
                self.use_ai = True
                st.success("✅ AI mode enabled")
            except ImportError:
                st.warning("⚠️ OpenAI not installed. Using basic mode.")
            except Exception as e:
                st.warning(f"⚠️ AI setup failed. Using basic mode.")
    
    def clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean the dataframe."""
        if self.use_ai:
            return self._clean_with_ai(df)
        else:
            return self._clean_basic(df)
    
    def _clean_with_ai(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean data using AI recommendations."""
        try:
            # Get AI analysis
            analysis = self._get_ai_analysis(df)
            
            if analysis and "error" not in analysis:
                st.info("🤖 AI analysis successful, applying recommendations...")
                return self._apply_ai_cleaning(df, analysis)
            else:
                st.warning("⚠️ AI analysis failed, using basic cleaning...")
                return self._clean_basic(df)
                
        except Exception as e:
            st.error(f"❌ AI cleaning failed: {e}")
            return self._clean_basic(df)
    
    def _get_ai_analysis(self, df: pd.DataFrame) -> dict:
        """Get AI analysis of the data."""
        # Prepare data sample
        sample = {
            'columns': list(df.columns),
            'sample_data': df.head(2).to_dict('records'),
            'missing_counts': {col: int(df[col].isnull().sum()) for col in df.columns}
        }
        
        prompt = f"""Analyze this CSV data and suggest cleaning steps.

Data: {json.dumps(sample, default=str)}

Respond with JSON only:
{{
    "cleaning_steps": [
        {{"action": "remove_duplicates", "reason": "duplicate rows found"}},
        {{"action": "clean_names", "columns": ["name"], "method": "title_case"}},
        {{"action": "clean_emails", "columns": ["email"], "method": "lowercase"}},
        {{"action": "fill_missing", "columns": ["age"], "method": "median"}}
    ]
}}"""
        
        try:
            response = self.client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1,
                max_tokens=300
            )
            
            content = response.choices[0].message.content
            if content is not None:
                result = json.loads(content)
                return result
            else:
                return {"error": "No content returned from AI response."}
            
        except Exception as e:
            return {"error": str(e)}
    
    def _apply_ai_cleaning(self, df: pd.DataFrame, analysis: dict) -> pd.DataFrame:
        """Apply AI-recommended cleaning steps."""
        cleaned_df = df.copy()
        
        steps = analysis.get('cleaning_steps', [])
        
        for step in steps:
            action = step.get('action', '')
            columns = step.get('columns', [])
            method = step.get('method', '')
            
            try:
                if action == 'remove_duplicates':
                    before = len(cleaned_df)
                    cleaned_df = cleaned_df.drop_duplicates()
                    removed = before - len(cleaned_df)
                    if removed > 0:
                        self.cleaning_log.append(f"🤖 AI removed {removed} duplicate rows")
                
                elif action == 'clean_names':
                    for col in columns:
                        if col in cleaned_df.columns:
                            cleaned_df[col] = cleaned_df[col].astype(str).str.title().str.strip()
                            self.cleaning_log.append(f"🤖 AI cleaned names in {col}")
                
                elif action == 'clean_emails':
                    for col in columns:
                        if col in cleaned_df.columns:
                            cleaned_df[col] = cleaned_df[col].astype(str).str.lower().str.strip()
                            self.cleaning_log.append(f"🤖 AI cleaned emails in {col}")
                
                elif action == 'fill_missing':
                    for col in columns:
                        if col in cleaned_df.columns:
                            missing_count = cleaned_df[col].isnull().sum()
                            if missing_count > 0:
                                if method == 'median' and cleaned_df[col].dtype in ['int64', 'float64']:
                                    cleaned_df[col] = cleaned_df[col].fillna(cleaned_df[col].median())
                                else:
                                    cleaned_df[col] = cleaned_df[col].fillna('Unknown')
                                self.cleaning_log.append(f"🤖 AI filled {missing_count} missing values in {col}")
            
            except Exception as e:
                st.warning(f"⚠️ Failed to apply {action}: {e}")
        
        return cleaned_df
    
    def _clean_basic(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply basic cleaning rules."""
        cleaned_df = df.copy()
        
        # Remove duplicates
        before = len(cleaned_df)
        cleaned_df = cleaned_df.drop_duplicates()
        if before > len(cleaned_df):
            self.cleaning_log.append(f"Removed {before - len(cleaned_df)} duplicate rows")
        
        # Clean text columns
        for col in cleaned_df.columns:
            if cleaned_df[col].dtype == 'object':
                # Strip whitespace
                cleaned_df[col] = cleaned_df[col].astype(str).str.strip()
                
                # Clean based on column name
                if 'email' in col.lower():
                    cleaned_df[col] = cleaned_df[col].str.lower()
                    self.cleaning_log.append(f"Cleaned emails in {col}")
                elif 'name' in col.lower():
                    cleaned_df[col] = cleaned_df[col].str.title()
                    self.cleaning_log.append(f"Cleaned names in {col}")
        
        # Fill missing values
        for col in cleaned_df.columns:
            missing_count = cleaned_df[col].isnull().sum()
            if missing_count > 0:
                if cleaned_df[col].dtype in ['int64', 'float64']:
                    cleaned_df[col] = cleaned_df[col].fillna(cleaned_df[col].median())
                else:
                    cleaned_df[col] = cleaned_df[col].fillna('Unknown')
                self.cleaning_log.append(f"Filled {missing_count} missing values in {col}")
        
        return cleaned_df


def create_sample_data():
    """Create sample messy data."""
    data = {
        'id': [1, 2, 3, 4, 5, 1, 6, 7, 8, None],  # Duplicate + missing
        'name': ['john doe', 'JANE SMITH', ' Alice Johnson ', 'Bob Wilson', None, 'john doe', 'Carol Brown', 'david lee', '  Mike   ', 'Sarah'],
        'email': ['john@gmail.com', 'JANE@YAHOO.COM', 'alice@email.com', 'bad-email', None, 'john@gmail.com', 'carol@email.com', 'david@email.com', 'mike@email.com', 'sarah@email.com'],
        'age': [25, 30, -5, 150, 28, 25, 32, None, 29, 31],  # Invalid ages
        'amount': [100.50, 250.00, 999999.99, 75.25, 180.00, 100.50, None, 320.50, 12.99, 500.00]
    }
    return pd.DataFrame(data)


def main():
    """Main Streamlit app."""
    st.set_page_config(
        page_title="Data Cleaner",
        page_icon="🧹",
        layout="wide"
    )
    
    # Header
    st.markdown("""
        <div style="background: linear-gradient(90deg, #4CAF50 0%, #45a049 100%); 
                    padding: 2rem; border-radius: 10px; color: white; text-align: center; margin-bottom: 2rem;">
            <h1>🧹 LLM-Powered Data Cleaner</h1>
            <p>Upload CSV → Clean Data → Download Results</p>
        </div>
    """, unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.header("⚙️ Settings")
        
        api_key = st.text_input(
            "OpenAI API Key (optional):",
            type="password",
            help="Leave empty for basic cleaning"
        )
        
        if api_key:
            st.success("🤖 AI mode will be used")
        else:
            st.info("🔧 Basic cleaning mode")
    
    # Main content
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("📂 Input Data")
        
        # Data source
        source = st.radio("Data source:", ["Upload CSV", "Use Sample"])
        
        df = None
        
        if source == "Upload CSV":
            uploaded_file = st.file_uploader("Choose CSV file", type=['csv'])
            if uploaded_file:
                try:
                    df = pd.read_csv(uploaded_file)
                    st.success(f"✅ Loaded: {df.shape[0]} rows, {df.shape[1]} columns")
                except Exception as e:
                    st.error(f"Error: {e}")
        
        else:  # Sample data
            if st.button("Load Sample Data"):
                df = create_sample_data()
                st.success("✅ Sample data loaded")
        
        # Show original data
        if df is not None:
            st.write("**Original Data:**")
            st.dataframe(df, use_container_width=True)
            
            # Stats
            col_a, col_b, col_c = st.columns(3)
            with col_a:
                st.metric("Rows", f"{df.shape[0]:,}")
            with col_b:
                st.metric("Missing", f"{df.isnull().sum().sum():,}")
            with col_c:
                st.metric("Duplicates", f"{df.duplicated().sum():,}")
    
    with col2:
        st.subheader("🧹 Clean Data")
        
        if df is not None:
            if st.button("🚀 Clean Data", type="primary", use_container_width=True):
                
                with st.spinner("Cleaning data..."):
                    # Initialize cleaner
                    cleaner = DataCleaner(api_key=api_key if api_key else None)
                    
                    # Clean data
                    cleaned_df = cleaner.clean_data(df)
                    
                    # Show results
                    st.success("✅ Cleaning completed!")
                    
                    # Before/after comparison
                    col_a, col_b, col_c = st.columns(3)
                    with col_a:
                        st.metric("Rows", f"{cleaned_df.shape[0]:,}", 
                                delta=f"{cleaned_df.shape[0] - df.shape[0]:,}")
                    with col_b:
                        orig_missing = df.isnull().sum().sum()
                        clean_missing = cleaned_df.isnull().sum().sum()
                        st.metric("Missing", f"{clean_missing:,}", 
                                delta=f"{clean_missing - orig_missing:,}")
                    with col_c:
                        orig_dup = df.duplicated().sum()
                        clean_dup = cleaned_df.duplicated().sum()
                        st.metric("Duplicates", f"{clean_dup:,}", 
                                delta=f"{clean_dup - orig_dup:,}")
                    
                    # What was done
                    if cleaner.cleaning_log:
                        st.write("**Actions taken:**")
                        for action in cleaner.cleaning_log:
                            st.success(f"✅ {action}")
                    
                    # Show cleaned data
                    st.write("**Cleaned Data:**")
                    st.dataframe(cleaned_df, use_container_width=True)
                    
                    # Download
                    csv = cleaned_df.to_csv(index=False)
                    st.download_button(
                        "📥 Download Cleaned Data",
                        data=csv,
                        file_name="cleaned_data.csv",
                        mime="text/csv",
                        use_container_width=True
                    )
        
        else:
            st.info("👈 Upload or load data to start cleaning")


if __name__ == "__main__":
    main()