#!/usr/bin/env python3
"""
Simple LLM-Powered Data Cleaner
Clean CSV files with optional AI assistance.
"""

import pandas as pd
import numpy as np
import json
import os
from typing import Dict, List, Optional


class DataCleaner:
    """Simple data cleaner with optional LLM support."""
    
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
                print("✅ AI mode enabled")
            except ImportError:
                print("⚠️ OpenAI not installed. Using basic mode.")
            except Exception as e:
                print(f"⚠️ AI setup failed: {e}. Using basic mode.")
    
    def clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean the dataframe."""
        print(f"🧹 Cleaning data: {df.shape[0]} rows, {df.shape[1]} columns")
        
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
                print("🤖 AI analysis successful, applying recommendations...")
                return self._apply_ai_cleaning(df, analysis)
            else:
                print("⚠️ AI analysis failed, using basic cleaning...")
                return self._clean_basic(df)
                
        except Exception as e:
            print(f"❌ AI cleaning failed: {e}")
            return self._clean_basic(df)
    
    def _get_ai_analysis(self, df: pd.DataFrame) -> Dict:
        """Get AI analysis of the data."""
        # Prepare data sample
        sample = {
            'columns': list(df.columns),
            'sample_data': df.head(3).to_dict('records'),
            'data_types': {col: str(df[col].dtype) for col in df.columns},
            'missing_counts': {col: int(df[col].isnull().sum()) for col in df.columns}
        }
        
        prompt = f"""Analyze this CSV data and suggest cleaning steps.

Data sample: {json.dumps(sample, default=str)}

Respond with JSON only:
{{
    "cleaning_steps": [
        {{"action": "remove_duplicates", "reason": "duplicate rows found"}},
        {{"action": "clean_names", "columns": ["name"], "method": "title_case"}},
        {{"action": "clean_emails", "columns": ["email"], "method": "lowercase"}},
        {{"action": "fill_missing", "columns": ["age"], "method": "median"}},
        {{"action": "remove_outliers", "columns": ["amount"]}}
    ]
}}"""
        
        try:
            response = self.client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1,
                max_tokens=500
            )
            
            content = response.choices[0].message.content
            if content is not None:
                result = json.loads(content)
                return result
            else:
                return {"error": "No content returned from AI response"}
            
        except Exception as e:
            return {"error": str(e)}
    
    def _apply_ai_cleaning(self, df: pd.DataFrame, analysis: Dict) -> pd.DataFrame:
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
                            if method == 'title_case':
                                cleaned_df[col] = cleaned_df[col].astype(str).str.title().str.strip()
                                self.cleaning_log.append(f"🤖 AI cleaned names in {col}")
                
                elif action == 'clean_emails':
                    for col in columns:
                        if col in cleaned_df.columns:
                            if method == 'lowercase':
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
                
                elif action == 'remove_outliers':
                    for col in columns:
                        if col in cleaned_df.columns and cleaned_df[col].dtype in ['int64', 'float64']:
                            Q1 = cleaned_df[col].quantile(0.25)
                            Q3 = cleaned_df[col].quantile(0.75)
                            IQR = Q3 - Q1
                            lower = Q1 - 3 * IQR
                            upper = Q3 + 3 * IQR
                            
                            before = len(cleaned_df)
                            cleaned_df = cleaned_df[(cleaned_df[col] >= lower) & (cleaned_df[col] <= upper)]
                            removed = before - len(cleaned_df)
                            if removed > 0:
                                self.cleaning_log.append(f"🤖 AI removed {removed} outliers from {col}")
            
            except Exception as e:
                print(f"⚠️ Failed to apply {action}: {e}")
        
        return cleaned_df
    
    def _clean_basic(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply basic cleaning rules."""
        print("🔧 Using basic cleaning rules...")
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