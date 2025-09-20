import streamlit as st
import pandas as pd
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
import torch
import os
import numpy as np
from datetime import datetime

# --- Page Configuration ---
st.set_page_config(
    page_title="Query-Category Relevance Classifier",
    layout="wide",
    page_icon="🛍️"
)

# --- Helper Functions ---
def check_model_files(model_path):
    """Check if all required model files exist."""
    required_files = ['config.json', 'pytorch_model.bin', 'tokenizer_config.json']
    missing_files = []
    
    if not os.path.exists(model_path):
        return False, [f"Model directory '{model_path}' not found"]
    
    for file in required_files:
        if not os.path.exists(os.path.join(model_path, file)):
            missing_files.append(file)
    
    return len(missing_files) == 0, missing_files

# --- Load Fine-Tuned Model ---
@st.cache_resource
def load_model(model_path='./finetuned_classifier'):
    """Loads the fine-tuned model and tokenizer with error handling."""
    try:
        # Check if model files exist
        model_exists, missing_files = check_model_files(model_path)
        if not model_exists:
            raise FileNotFoundError(f"Missing model files: {missing_files}")
        
        # Determine device
        if torch.cuda.is_available():
            device = 0
            st.sidebar.success("🚀 Using GPU for inference")
        else:
            device = -1
            st.sidebar.info("💻 Using CPU for inference")
        
        # Load the model with explicit error handling
        try:
            classifier = pipeline(
                "text-classification",
                model=model_path,
                tokenizer=model_path,
                device=device
            )
        except Exception as e:
            # Fallback: try loading model and tokenizer separately
            st.warning("Standard loading failed, trying alternative method...")
            tokenizer = AutoTokenizer.from_pretrained(model_path)
            model = AutoModelForSequenceClassification.from_pretrained(model_path)
            classifier = pipeline(
                "text-classification",
                model=model,
                tokenizer=tokenizer,
                device=device
            )
        
        return classifier
    except Exception as e:
        raise Exception(f"Failed to load model: {str(e)}")

# --- Main App ---
st.title("🛍️ E-Commerce: Multilingual Query–Category Relevance")
st.markdown("""
This demo uses a fine-tuned model to predict if a search query is relevant to a product category.
- **Input**: CSV file with `origin_query` and `cate_path` columns
- **Output**: Binary predictions (1 = Relevant, 0 = Not Relevant)
""")

# Sidebar information
st.sidebar.header("ℹ️ Information")
st.sidebar.markdown("""
### Expected CSV Format:
- `origin_query`: Search query in any language
- `cate_path`: Category path in English
- `language` (optional): Language code

### Model Details:
- Base: DistilBERT Multilingual
- Task: Binary Classification
- Languages: 20+ supported
""")

# Load model with better error handling
try:
    with st.spinner("Loading model... This may take a moment on first run."):
        classifier = load_model()
    st.success("✅ Model loaded successfully!")
    
    # Display model info
    model_info = st.expander("📊 Model Information", expanded=False)
    with model_info:
        st.write(f"Model: {classifier.model.config._name_or_path}")
        st.write(f"Number of labels: {classifier.model.config.num_labels}")
        
except FileNotFoundError as e:
    st.error(f"❌ Model files not found: {e}")
    st.info("""
    Please ensure you have trained the model first:
    ```bash
    python train.py --train_csv your_training_data.csv
    ```
    """)
    st.stop()
except Exception as e:
    st.error(f"❌ Error loading model: {e}")
    st.stop()

# --- File Uploader ---
st.header("📤 Upload Test Data")
uploaded_file = st.file_uploader(
    "Upload your test set (CSV)", 
    type=["csv"],
    help="CSV file must contain 'origin_query' and 'cate_path' columns"
)

if uploaded_file is not None:
    try:
        # Load and validate data
        df_test = pd.read_csv(uploaded_file)
        
        # Check for required columns
        required_columns = ['origin_query', 'cate_path']
        missing_columns = [col for col in required_columns if col not in df_test.columns]
        
        if missing_columns:
            st.error(f"❌ Missing required columns: {missing_columns}")
            st.stop()
        
        # Display data info
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Rows", len(df_test))
        with col2:
            st.metric("Unique Queries", df_test['origin_query'].nunique())
        with col3:
            if 'language' in df_test.columns:
                st.metric("Languages", df_test['language'].nunique())
        
        # Data preview
        st.write("### 📋 Data Preview")
        st.dataframe(df_test.head(10))
        
        # Check for missing values
        if df_test[required_columns].isnull().any().any():
            st.warning("⚠️ Found missing values in required columns. They will be skipped during prediction.")
            
    except Exception as e:
        st.error(f"❌ Error reading CSV file: {e}")
        st.stop()

    # Run inference button
    if st.button("🚀 Run Inference", type="primary"):
        try:
            with st.spinner('Running predictions... This may take a moment for large datasets.'):
                # Prepare data
                df_test = df_test.dropna(subset=required_columns)
                df_test['text_input'] = df_test['origin_query'].astype(str) + " [SEP] " + df_test['cate_path'].astype(str)
                texts = df_test['text_input'].tolist()
                
                # Progress bar for large datasets
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                # Batch processing for large datasets
                batch_size = 32
                all_predictions = []
                all_scores = []
                
                for i in range(0, len(texts), batch_size):
                    batch_texts = texts[i:i + batch_size]
                    batch_results = classifier(batch_texts)
                    
                    # Extract predictions and scores
                    for res in batch_results:
                        # Handle different label formats
                        if isinstance(res['label'], str):
                            if 'LABEL_' in res['label']:
                                pred = int(res['label'].split('_')[1])
                            else:
                                pred = 1 if res['label'].lower() in ['positive', 'relevant', '1'] else 0
                        else:
                            pred = int(res['label'])
                        
                        all_predictions.append(pred)
                        all_scores.append(res['score'])
                    
                    # Update progress
                    progress = min((i + batch_size) / len(texts), 1.0)
                    progress_bar.progress(progress)
                    status_text.text(f"Processed {min(i + batch_size, len(texts))}/{len(texts)} samples")
                
                # Clear progress indicators
                progress_bar.empty()
                status_text.empty()
                
                # Add predictions to dataframe
                df_test['prediction'] = all_predictions
                df_test['confidence'] = all_scores
                
                # Display results summary
                st.success(f"✅ Predictions completed for {len(all_predictions)} samples!")
                
                # Results statistics
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Total Predictions", len(all_predictions))
                with col2:
                    relevant_count = sum(all_predictions)
                    st.metric("Relevant", relevant_count)
                with col3:
                    st.metric("Not Relevant", len(all_predictions) - relevant_count)
                with col4:
                    avg_confidence = np.mean(all_scores)
                    st.metric("Avg Confidence", f"{avg_confidence:.2%}")
                
                # Show results
                st.write("### 📊 Inference Results")
                
                # Display options
                display_cols = ['origin_query', 'cate_path', 'prediction', 'confidence']
                if 'language' in df_test.columns:
                    display_cols.insert(0, 'language')
                
                # Filter options
                col1, col2 = st.columns(2)
                with col1:
                    show_only_relevant = st.checkbox("Show only relevant predictions")
                with col2:
                    confidence_threshold = st.slider("Confidence threshold", 0.0, 1.0, 0.5)
                
                # Apply filters
                display_df = df_test[display_cols].copy()
                if show_only_relevant:
                    display_df = display_df[display_df['prediction'] == 1]
                display_df = display_df[display_df['confidence'] >= confidence_threshold]
                
                st.dataframe(
                    display_df,
                    use_container_width=True,
                    hide_index=True
                )
                
                # Download results
                st.write("### 💾 Download Results")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    # Download predictions only (for competition submission)
                    @st.cache_data
                    def convert_predictions_to_csv(predictions):
                        df_submit = pd.DataFrame({'label': predictions})
                        return df_submit.to_csv(index=False).encode('utf-8')
                    
                    csv_predictions = convert_predictions_to_csv(all_predictions)
                    st.download_button(
                        label="⬇️ Download Predictions (Competition Format)",
                        data=csv_predictions,
                        file_name=f'predictions_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv',
                        mime='text/csv',
                        help="CSV with only 'label' column for competition submission"
                    )
                
                with col2:
                    # Download full results
                    @st.cache_data
                    def convert_full_results_to_csv(df):
                        return df.to_csv(index=False).encode('utf-8')
                    
                    csv_full = convert_full_results_to_csv(df_test)
                    st.download_button(
                        label="⬇️ Download Full Results",
                        data=csv_full,
                        file_name=f'full_results_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv',
                        mime='text/csv',
                        help="CSV with all columns including confidence scores"
                    )
                
                # Language-wise analysis if language column exists
                if 'language' in df_test.columns:
                    st.write("### 🌍 Language-wise Performance")
                    lang_stats = df_test.groupby('language').agg({
                        'prediction': ['sum', 'count', 'mean'],
                        'confidence': 'mean'
                    }).round(3)
                    lang_stats.columns = ['Relevant Count', 'Total', 'Relevance Rate', 'Avg Confidence']
                    st.dataframe(lang_stats)
                    
        except Exception as e:
            st.error(f"❌ Error during inference: {e}")
            st.exception(e)  # Show full traceback in expander

else:
    # Show sample data format
    st.info("👆 Please upload a CSV file to begin")
    
    with st.expander("📝 Sample Data Format"):
        sample_data = pd.DataFrame({
            'language': ['en', 'es', 'ko'],
            'origin_query': ['red running shoes', 'zapatos para correr', '런닝화'],
            'cate_path': ['Footwear > Athletic > Running Shoes'] * 3
        })
        st.dataframe(sample_data)
        
        # Provide sample CSV download
        @st.cache_data
        def get_sample_csv():
            return sample_data.to_csv(index=False).encode('utf-8')
      
        st.download_button(
            label="📥 Download Sample CSV",
            data=get_sample_csv(),
            file_name="sample_test_data.csv",
            mime="text/csv"
        )

# Footer
st.markdown("---")
st.markdown(
    """
    <div style='text-align: center'>
        <p>Built for E-Commerce Search Relevance Challenge 2024</p>
    </div>
    """, 
    unsafe_allow_html=True
)
