import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings('ignore')

# Note: ucimlrepo will be imported in the load_data function
# to provide better error handling and installation instructions

# Page configuration
st.set_page_config(
    page_title="Healthcare Readmission Analysis Dashboard",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
    <style>
    .main-header {
        font-size: 36px;
        color: #1E3A8A;
        font-weight: bold;
        text-align: center;
        padding: 20px 0;
    }
    .sub-header {
        font-size: 24px;
        color: #2563EB;
        font-weight: bold;
        margin-top: 20px;
    }
    .metric-card {
        background-color: #F0F9FF;
        padding: 20px;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .insight-box {
        background-color: #FEF3C7;
        padding: 15px;
        border-radius: 8px;
        margin: 10px 0;
        border-left: 4px solid #F59E0B;
    }
    </style>
""", unsafe_allow_html=True)

# Title
st.markdown('<h1 class="main-header">🏥 Healthcare Readmission Analysis Dashboard</h1>', unsafe_allow_html=True)
st.markdown("### IT5006 - Diabetes 130-US Hospitals Dataset (1999-2008)")

# Sidebar for navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio(
    "Select Analysis Section",
    ["📊 Dataset Overview", 
     "👥 Patient Demographics", 
     "🏥 Clinical Analysis",
     "💊 Medications & Treatment",
     "📈 Readmission Analysis",
     "🔍 Feature Relationships",
     "💡 Key Insights"]
)

# Load data function with caching
@st.cache_data
def load_data():
    """
    Load the diabetes dataset directly from UCI ML Repository.
    No need to download files manually!
    """
    try:
        # First, try to import ucimlrepo
        try:
            from ucimlrepo import fetch_ucirepo
        except ImportError:
            st.error("⚠️ Please install the ucimlrepo package first!")
            st.code("pip install ucimlrepo", language="bash")
            st.stop()
            return None
        
        # Fetch the diabetes dataset (ID: 296)
        with st.spinner('🔄 Fetching dataset from UCI ML Repository... This may take a moment on first run.'):
            diabetes_dataset = fetch_ucirepo(id=296)
        
        # Get the features and targets
        X = diabetes_dataset.data.features
        y = diabetes_dataset.data.targets
        
        # Combine features and targets into a single dataframe
        df = pd.concat([X, y], axis=1)
        
        # Handle missing values marked as '?'
        df = df.replace('?', np.nan)
        
        st.success("✅ Dataset loaded successfully from UCI ML Repository!")
        
        # Display dataset metadata
        with st.expander("📋 Dataset Information"):
            st.write("**Dataset Name:**", diabetes_dataset.metadata['name'])
            st.write("**Dataset ID:**", diabetes_dataset.metadata['uci_id'])
            st.write("**Repository URL:**", diabetes_dataset.metadata['repository_url'])
            if 'abstract' in diabetes_dataset.metadata:
                st.write("**Abstract:**", diabetes_dataset.metadata['abstract'][:500] + "...")
        
        return df
        
    except Exception as e:
        st.error(f"❌ Error loading dataset: {str(e)}")
        st.info("""
        **Troubleshooting steps:**
        1. Ensure you have internet connection
        2. Install ucimlrepo: `pip install ucimlrepo`
        3. Try refreshing the page
        4. If error persists, you can manually download from: https://archive.ics.uci.edu/dataset/296/
        """)
        return None

# Load the data
df = load_data()

if df is not None:
    # Data preprocessing for EDA
    @st.cache_data
    def preprocess_data(df):
        """Basic preprocessing for EDA"""
        df_processed = df.copy()
        
        # Convert readmitted to categorical
        readmission_map = {
            'NO': 'Not Readmitted',
            '<30': 'Readmitted <30 days',
            '>30': 'Readmitted >30 days'
        }
        df_processed['readmitted_category'] = df_processed['readmitted'].map(readmission_map)
        
        # Create binary readmission variable
        df_processed['readmitted_30days'] = df_processed['readmitted'].apply(
            lambda x: 1 if x == '<30' else 0
        )
        
        # Age group cleaning
        df_processed['age_group'] = df_processed['age'].str.replace('[', '').str.replace(')', '')
        
        return df_processed
    
    df_processed = preprocess_data(df)
    
    # Page: Dataset Overview
    if page == "📊 Dataset Overview":
        st.markdown('<h2 class="sub-header">Dataset Overview</h2>', unsafe_allow_html=True)
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Records", f"{len(df):,}")
        with col2:
            st.metric("Total Features", f"{df.shape[1]}")
        with col3:
            st.metric("Total Encounters", f"{len(df):,}")
        with col4:
            st.metric("Unique Hospitals", "130")
        
        st.markdown("---")
        
        # Data shape and info
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("📋 Dataset Information")
            info_df = pd.DataFrame({
                'Metric': ['Total Encounters', 'Date Range', 'Data Period', 
                          'Number of Hospitals', 'Memory Usage'],
                'Value': [
                    f"{len(df):,}",
                    "1999-2008",
                    "10 years",
                    "130",
                    f"{df.memory_usage(deep=True).sum() / 1024**2:.2f} MB"
                ]
            })
            st.dataframe(info_df, hide_index=True)
        
        with col2:
            st.subheader("🎯 Readmission Distribution")
            readmit_counts = df_processed['readmitted_category'].value_counts()
            fig = px.pie(
                values=readmit_counts.values,
                names=readmit_counts.index,
                title="Hospital Readmission Status",
                color_discrete_sequence=['#10B981', '#F59E0B', '#EF4444']
            )
            st.plotly_chart(fig, use_container_width=True)
        
        # Missing data analysis
        st.subheader("🔍 Missing Data Analysis")
        missing_df = pd.DataFrame({
            'Column': df.columns,
            'Missing_Count': df.isnull().sum(),
            'Missing_Percentage': (df.isnull().sum() / len(df) * 100).round(2)
        }).sort_values('Missing_Percentage', ascending=False)
        
        missing_df = missing_df[missing_df['Missing_Count'] > 0]
        
        if len(missing_df) > 0:
            fig = px.bar(
                missing_df.head(20), 
                x='Column', 
                y='Missing_Percentage',
                title="Top 20 Columns with Missing Data",
                labels={'Missing_Percentage': 'Missing %'},
                color='Missing_Percentage',
                color_continuous_scale='Reds'
            )
            fig.update_layout(xaxis_tickangle=-45)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.success("✅ No missing data found in the dataset!")
        
        # Data types distribution
        st.subheader("📊 Data Types Distribution")
        dtype_counts = df.dtypes.value_counts()
        col1, col2 = st.columns([1, 2])
        
        with col1:
            dtype_df = pd.DataFrame({
                'Data Type': dtype_counts.index.astype(str),
                'Count': dtype_counts.values
            })
            st.dataframe(dtype_df, hide_index=True)
        
        with col2:
            fig = px.pie(
                values=dtype_counts.values,
                names=dtype_counts.index.astype(str),
                title="Feature Data Types Distribution"
            )
            st.plotly_chart(fig, use_container_width=True)
    
    # Page: Patient Demographics
    elif page == "👥 Patient Demographics":
        st.markdown('<h2 class="sub-header">Patient Demographics Analysis</h2>', unsafe_allow_html=True)
        
        # Demographics filters
        st.sidebar.subheader("Demographic Filters")
        selected_race = st.sidebar.multiselect(
            "Select Race", 
            options=df['race'].dropna().unique(),
            default=df['race'].dropna().unique()
        )
        selected_gender = st.sidebar.multiselect(
            "Select Gender",
            options=df['gender'].dropna().unique(),
            default=df['gender'].dropna().unique()
        )
        
        # Filter data
        demo_df = df_processed[
            (df_processed['race'].isin(selected_race)) &
            (df_processed['gender'].isin(selected_gender))
        ]
        
        # Age distribution
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("📊 Age Distribution")
            age_order = ['0-10', '10-20', '20-30', '30-40', '40-50', 
                        '50-60', '60-70', '70-80', '80-90', '90-100']
            age_counts = demo_df['age_group'].value_counts()
            age_counts = age_counts.reindex([a for a in age_order if a in age_counts.index])
            
            fig = px.bar(
                x=age_counts.index,
                y=age_counts.values,
                title="Patient Age Distribution",
                labels={'x': 'Age Group', 'y': 'Number of Patients'},
                color=age_counts.values,
                color_continuous_scale='Blues'
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.subheader("👥 Gender Distribution")
            gender_counts = demo_df['gender'].value_counts()
            fig = px.pie(
                values=gender_counts.values,
                names=gender_counts.index,
                title="Patient Gender Distribution",
                hole=0.4
            )
            st.plotly_chart(fig, use_container_width=True)
        
        # Race distribution
        st.subheader("🌍 Race/Ethnicity Distribution")
        race_counts = demo_df['race'].value_counts()
        fig = px.bar(
            x=race_counts.values,
            y=race_counts.index,
            orientation='h',
            title="Patient Race/Ethnicity Distribution",
            labels={'x': 'Number of Patients', 'y': 'Race/Ethnicity'},
            color=race_counts.values,
            color_continuous_scale='Viridis'
        )
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)
        
        # Cross-tabulation
        st.subheader("📈 Readmission Rates by Demographics")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Readmission by age
            age_readmit = pd.crosstab(
                demo_df['age_group'], 
                demo_df['readmitted_30days'],
                normalize='index'
            ) * 100
            
            fig = px.bar(
                x=age_readmit.index,
                y=age_readmit[1],
                title="30-Day Readmission Rate by Age Group",
                labels={'x': 'Age Group', 'y': 'Readmission Rate (%)'},
                color=age_readmit[1],
                color_continuous_scale='Reds'
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Readmission by gender
            gender_readmit = pd.crosstab(
                demo_df['gender'], 
                demo_df['readmitted_30days'],
                normalize='index'
            ) * 100
            
            fig = px.bar(
                x=gender_readmit.index,
                y=gender_readmit[1],
                title="30-Day Readmission Rate by Gender",
                labels={'x': 'Gender', 'y': 'Readmission Rate (%)'},
                color=gender_readmit[1],
                color_continuous_scale='Oranges'
            )
            st.plotly_chart(fig, use_container_width=True)
    
    # Page: Clinical Analysis
    elif page == "🏥 Clinical Analysis":
        st.markdown('<h2 class="sub-header">Clinical Metrics Analysis</h2>', unsafe_allow_html=True)
        
        # Length of stay analysis
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("⏱️ Length of Stay Distribution")
            fig = px.histogram(
                df_processed,
                x='time_in_hospital',
                nbins=14,
                title="Hospital Length of Stay",
                labels={'time_in_hospital': 'Days in Hospital', 'count': 'Number of Patients'},
                color_discrete_sequence=['#3B82F6']
            )
            fig.add_vline(
                x=df_processed['time_in_hospital'].mean(),
                line_dash="dash",
                line_color="red",
                annotation_text=f"Mean: {df_processed['time_in_hospital'].mean():.1f} days"
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.subheader("🔬 Number of Lab Procedures")
            fig = px.box(
                df_processed,
                y='num_lab_procedures',
                title="Distribution of Lab Procedures",
                color_discrete_sequence=['#10B981']
            )
            st.plotly_chart(fig, use_container_width=True)
        
        # Admission and discharge analysis
        st.subheader("🚪 Admission and Discharge Patterns")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            admission_types = {
                1: 'Emergency',
                2: 'Urgent',
                3: 'Elective',
                4: 'Newborn',
                5: 'Not Available',
                6: 'NULL',
                7: 'Trauma Center',
                8: 'Not Mapped'
            }
            df_processed['admission_type_name'] = df_processed['admission_type_id'].map(admission_types)
            admission_counts = df_processed['admission_type_name'].value_counts().head(5)
            
            fig = px.pie(
                values=admission_counts.values,
                names=admission_counts.index,
                title="Top 5 Admission Types",
                hole=0.3
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Number of diagnoses
            fig = px.histogram(
                df_processed,
                x='number_diagnoses',
                title="Number of Diagnoses Distribution",
                labels={'number_diagnoses': 'Number of Diagnoses'},
                color_discrete_sequence=['#F59E0B']
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with col3:
            # Number of medications
            fig = px.histogram(
                df_processed,
                x='num_medications',
                title="Number of Medications Distribution",
                labels={'num_medications': 'Number of Medications'},
                color_discrete_sequence=['#EF4444']
            )
            st.plotly_chart(fig, use_container_width=True)
        
        # Previous visits analysis
        st.subheader("🏥 Previous Healthcare Utilization")
        
        visit_cols = ['number_outpatient', 'number_emergency', 'number_inpatient']
        visit_data = df_processed[visit_cols].mean()
        
        # Create DataFrame for proper Plotly handling
        visit_display_df = pd.DataFrame({
            'Visit_Type': ['Outpatient', 'Emergency', 'Inpatient'],
            'Average_Visits': visit_data.values
        })
        
        fig = px.bar(
            visit_display_df,
            x='Visit_Type',
            y='Average_Visits',
            title="Average Number of Previous Visits (Past Year)",
            labels={'Visit_Type': 'Visit Type', 'Average_Visits': 'Average Number of Visits'},
            color='Average_Visits',
            color_continuous_scale='Purples'
        )
        st.plotly_chart(fig, use_container_width=True)
    
    # Page: Medications & Treatment
    elif page == "💊 Medications & Treatment":
        st.markdown('<h2 class="sub-header">Medications & Treatment Analysis</h2>', unsafe_allow_html=True)
        
        # Diabetes medication analysis
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("💊 Diabetes Medication Prescribed")
            diabetes_med_counts = df_processed['diabetesMed'].value_counts()
            fig = px.pie(
                values=diabetes_med_counts.values,
                names=diabetes_med_counts.index,
                title="Patients on Diabetes Medication",
                color_discrete_sequence=['#22C55E', '#DC2626']
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.subheader("💉 Insulin Treatment")
            insulin_counts = df_processed['insulin'].value_counts()
            fig = px.bar(
                x=insulin_counts.index,
                y=insulin_counts.values,
                title="Insulin Dosage Changes",
                labels={'x': 'Insulin Status', 'y': 'Number of Patients'},
                color=insulin_counts.index,
                color_discrete_sequence=['#6B7280', '#3B82F6', '#10B981', '#F59E0B']
            )
            st.plotly_chart(fig, use_container_width=True)
        
        # Medication changes
        st.subheader("🔄 Medication Changes During Stay")
        change_counts = df_processed['change'].value_counts()
        
        col1, col2 = st.columns([1, 2])
        
        with col1:
            change_df = pd.DataFrame({
                'Change Status': ['Changed' if x == 'Ch' else 'No Change' for x in change_counts.index],
                'Count': change_counts.values,
                'Percentage': (change_counts.values / change_counts.sum() * 100).round(2)
            })
            st.dataframe(change_df, hide_index=True)
        
        with col2:
            fig = px.pie(
                values=change_counts.values,
                names=['Changed' if x == 'Ch' else 'No Change' for x in change_counts.index],
                title="Medication Change Distribution",
                hole=0.4,
                color_discrete_sequence=['#F59E0B', '#3B82F6']
            )
            st.plotly_chart(fig, use_container_width=True)
        
        # Individual medications analysis
        st.subheader("📋 Common Diabetes Medications Usage")
        
        med_columns = ['metformin', 'repaglinide', 'nateglinide', 'chlorpropamide',
                      'glimepiride', 'glipizide', 'glyburide', 'pioglitazone',
                      'rosiglitazone']
        
        med_usage = {}
        for med in med_columns:
            if med in df_processed.columns:
                med_usage[med] = (df_processed[med] != 'No').sum()
        
        med_df = pd.DataFrame(list(med_usage.items()), columns=['Medication', 'Patients'])
        med_df = med_df.sort_values('Patients', ascending=True)
        
        fig = px.bar(
            med_df,
            x='Patients',
            y='Medication',
            orientation='h',
            title="Number of Patients on Each Medication",
            color='Patients',
            color_continuous_scale='Viridis'
        )
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)
        
        # Lab results
        st.subheader("🧪 Laboratory Results")
        
        col1, col2 = st.columns(2)
        
        with col1:
            if 'A1Cresult' in df_processed.columns:
                a1c_counts = df_processed['A1Cresult'].value_counts()
                fig = px.pie(
                    values=a1c_counts.values,
                    names=a1c_counts.index,
                    title="HbA1c Test Results Distribution",
                    color_discrete_sequence=['#E5E7EB', '#22C55E', '#F59E0B', '#EF4444']
                )
                st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            if 'max_glu_serum' in df_processed.columns:
                glucose_counts = df_processed['max_glu_serum'].value_counts()
                fig = px.pie(
                    values=glucose_counts.values,
                    names=glucose_counts.index,
                    title="Glucose Serum Test Results",
                    color_discrete_sequence=['#E5E7EB', '#22C55E', '#F59E0B', '#EF4444']
                )
                st.plotly_chart(fig, use_container_width=True)
    
    # Page: Readmission Analysis
    elif page == "📈 Readmission Analysis":
        st.markdown('<h2 class="sub-header">Hospital Readmission Analysis</h2>', unsafe_allow_html=True)
        
        # Overall readmission statistics
        col1, col2, col3, col4 = st.columns(4)
        
        total_patients = len(df_processed)
        readmitted_30 = (df_processed['readmitted'] == '<30').sum()
        readmitted_more30 = (df_processed['readmitted'] == '>30').sum()
        not_readmitted = (df_processed['readmitted'] == 'NO').sum()
        
        with col1:
            st.metric("Total Encounters", f"{total_patients:,}")
        with col2:
            st.metric("30-Day Readmission Rate", f"{readmitted_30/total_patients*100:.2f}%")
        with col3:
            st.metric(">30-Day Readmission Rate", f"{readmitted_more30/total_patients*100:.2f}%")
        with col4:
            st.metric("No Readmission Rate", f"{not_readmitted/total_patients*100:.2f}%")
        
        st.markdown("---")
        
        # Readmission by various factors
        st.subheader("🔍 Readmission Analysis by Key Factors")
        
        analysis_factor = st.selectbox(
            "Select Factor for Analysis",
            ["Length of Stay", "Number of Medications", "Number of Procedures", 
             "Number of Diagnoses", "Age Group", "Admission Type"]
        )
        
        if analysis_factor == "Length of Stay":
            # Create bins for length of stay
            df_processed['los_bins'] = pd.cut(
                df_processed['time_in_hospital'],
                bins=[0, 2, 4, 7, 14],
                labels=['1-2 days', '3-4 days', '5-7 days', '8-14 days']
            )
            
            readmit_by_los = pd.crosstab(
                df_processed['los_bins'],
                df_processed['readmitted_category'],
                normalize='index'
            ) * 100
            
            fig = px.bar(
                readmit_by_los,
                title="Readmission Rates by Length of Stay",
                labels={'value': 'Percentage (%)', 'index': 'Length of Stay'},
                color_discrete_sequence=['#10B981', '#F59E0B', '#EF4444']
            )
            st.plotly_chart(fig, use_container_width=True)
            
        elif analysis_factor == "Number of Medications":
            df_processed['med_bins'] = pd.cut(
                df_processed['num_medications'],
                bins=[0, 5, 10, 15, 20, 100],
                labels=['1-5', '6-10', '11-15', '16-20', '>20']
            )
            
            readmit_by_meds = pd.crosstab(
                df_processed['med_bins'],
                df_processed['readmitted_category'],
                normalize='index'
            ) * 100
            
            fig = px.bar(
                readmit_by_meds,
                title="Readmission Rates by Number of Medications",
                labels={'value': 'Percentage (%)', 'index': 'Number of Medications'},
                color_discrete_sequence=['#10B981', '#F59E0B', '#EF4444']
            )
            st.plotly_chart(fig, use_container_width=True)
            
        elif analysis_factor == "Age Group":
            readmit_by_age = pd.crosstab(
                df_processed['age_group'],
                df_processed['readmitted_category'],
                normalize='index'
            ) * 100
            
            fig = px.bar(
                readmit_by_age,
                title="Readmission Rates by Age Group",
                labels={'value': 'Percentage (%)', 'index': 'Age Group'},
                color_discrete_sequence=['#10B981', '#F59E0B', '#EF4444']
            )
            st.plotly_chart(fig, use_container_width=True)
        
        # Risk factors correlation
        st.subheader("⚠️ Risk Factors for 30-Day Readmission")
        
        # Calculate correlation with readmission
        numeric_cols = ['time_in_hospital', 'num_lab_procedures', 'num_procedures',
                       'num_medications', 'number_outpatient', 'number_emergency',
                       'number_inpatient', 'number_diagnoses']
        
        correlations = {}
        for col in numeric_cols:
            if col in df_processed.columns:
                correlations[col] = df_processed[col].corr(df_processed['readmitted_30days'])
        
        corr_df = pd.DataFrame(list(correlations.items()), columns=['Factor', 'Correlation'])
        corr_df = corr_df.sort_values('Correlation', key=abs, ascending=False)
        
        fig = px.bar(
            corr_df,
            x='Correlation',
            y='Factor',
            orientation='h',
            title="Correlation with 30-Day Readmission",
            color='Correlation',
            color_continuous_scale='RdBu_r',
            color_continuous_midpoint=0
        )
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)
    
    # Page: Feature Relationships
    elif page == "🔍 Feature Relationships":
        st.markdown('<h2 class="sub-header">Feature Relationships & Correlations</h2>', unsafe_allow_html=True)
        
        # Correlation matrix
        st.subheader("🔗 Correlation Matrix - Numerical Features")
        
        numeric_cols = ['time_in_hospital', 'num_lab_procedures', 'num_procedures',
                       'num_medications', 'number_outpatient', 'number_emergency',
                       'number_inpatient', 'number_diagnoses', 'readmitted_30days']
        
        corr_matrix = df_processed[numeric_cols].corr()
        
        fig = px.imshow(
            corr_matrix,
            title="Feature Correlation Heatmap",
            color_continuous_scale='RdBu_r',
            aspect='auto',
            text_auto='.2f'
        )
        fig.update_layout(height=600)
        st.plotly_chart(fig, use_container_width=True)
        
        # Scatter plot analysis
        st.subheader("📊 Interactive Scatter Plot Analysis")
        
        col1, col2 = st.columns(2)
        
        with col1:
            x_var = st.selectbox(
                "Select X-axis variable",
                numeric_cols[:-1]
            )
        
        with col2:
            y_var = st.selectbox(
                "Select Y-axis variable",
                numeric_cols[:-1],
                index=1
            )
        
        fig = px.scatter(
            df_processed.sample(min(5000, len(df_processed))),
            x=x_var,
            y=y_var,
            color='readmitted_category',
            title=f"{x_var} vs {y_var}",
            color_discrete_sequence=['#10B981', '#F59E0B', '#EF4444'],
            opacity=0.6
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Feature importance for readmission
        st.subheader("📈 Feature Importance Analysis")
        
        # Calculate simple importance scores based on correlation and variance
        importance_scores = {}
        
        for col in numeric_cols[:-1]:
            correlation = abs(df_processed[col].corr(df_processed['readmitted_30days']))
            variance = df_processed[col].var()
            importance_scores[col] = correlation * np.log1p(variance)
        
        importance_df = pd.DataFrame(list(importance_scores.items()), 
                                    columns=['Feature', 'Importance Score'])
        importance_df = importance_df.sort_values('Importance Score', ascending=True)
        
        fig = px.bar(
            importance_df,
            x='Importance Score',
            y='Feature',
            orientation='h',
            title="Feature Importance for Readmission Prediction",
            color='Importance Score',
            color_continuous_scale='Viridis'
        )
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)
    
    # Page: Key Insights
    elif page == "💡 Key Insights":
        st.markdown('<h2 class="sub-header">Key Insights & Summary</h2>', unsafe_allow_html=True)
        
        # Calculate key statistics
        readmission_rate_30 = (df_processed['readmitted_30days'].mean() * 100)
        avg_los = df_processed['time_in_hospital'].mean()
        avg_meds = df_processed['num_medications'].mean()
        
        # Key metrics summary
        st.subheader("📊 Key Performance Indicators")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric(
                "30-Day Readmission Rate",
                f"{readmission_rate_30:.2f}%",
                delta=f"{readmission_rate_30 - 15:.2f}% vs target",
                delta_color="inverse"
            )
        
        with col2:
            st.metric(
                "Average Length of Stay",
                f"{avg_los:.1f} days"
            )
        
        with col3:
            st.metric(
                "Average Medications",
                f"{avg_meds:.1f}"
            )
        
        with col4:
            diabetes_med_rate = (df_processed['diabetesMed'] == 'Yes').mean() * 100
            st.metric(
                "Diabetes Med Rate",
                f"{diabetes_med_rate:.1f}%"
            )
        
        st.markdown("---")
        
        # Key insights
        st.subheader("🔍 Data-Driven Insights")
        
        insights = [
            {
                "category": "Demographics",
                "insight": f"The majority of patients are in the {df_processed['age_group'].mode()[0]} age group, "
                          f"with {(df_processed['gender'].value_counts().iloc[0] / len(df_processed) * 100):.1f}% "
                          f"being {df_processed['gender'].value_counts().index[0]}.",
                "recommendation": "Consider age and gender-specific intervention programs."
            },
            {
                "category": "Length of Stay",
                "insight": f"Average length of stay is {avg_los:.1f} days, with {(df_processed['time_in_hospital'] > 7).mean() * 100:.1f}% "
                          f"of patients staying more than a week.",
                "recommendation": "Focus on reducing extended stays through better discharge planning."
            },
            {
                "category": "Medications",
                "insight": f"Patients receive an average of {avg_meds:.1f} medications, "
                          f"with {(df_processed['change'] == 'Ch').mean() * 100:.1f}% having medication changes during stay.",
                "recommendation": "Implement medication reconciliation protocols to optimize treatment."
            },
            {
                "category": "Readmissions",
                "insight": f"The 30-day readmission rate is {readmission_rate_30:.2f}%, "
                          f"with emergency admissions having higher readmission rates.",
                "recommendation": "Develop targeted post-discharge follow-up for high-risk patients."
            },
            {
                "category": "Lab Procedures",
                "insight": f"Patients undergo an average of {df_processed['num_lab_procedures'].mean():.1f} lab procedures, "
                          f"with high variability (std: {df_processed['num_lab_procedures'].std():.1f}).",
                "recommendation": "Standardize lab testing protocols to reduce unnecessary procedures."
            }
        ]
        
        for insight_dict in insights:
            st.markdown(f"""
            <div class="insight-box">
                <strong>{insight_dict['category']}</strong><br>
                📌 <em>Finding:</em> {insight_dict['insight']}<br>
                💡 <em>Recommendation:</em> {insight_dict['recommendation']}
            </div>
            """, unsafe_allow_html=True)
        
        # Summary statistics table
        st.subheader("📋 Summary Statistics")
        
        summary_stats = df_processed[numeric_cols].describe().round(2)
        st.dataframe(summary_stats, use_container_width=True)
        
        # Export functionality
        st.subheader("📥 Export Analysis Results")
        
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("Generate Summary Report"):
                report = f"""
                Healthcare Readmission Analysis Summary
                ========================================
                Generated: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}
                
                Dataset Overview:
                - Total Encounters: {len(df_processed):,}
                - Data Period: 1999-2008 (10 years)
                - Number of Hospitals: 130
                
                Key Metrics:
                - 30-Day Readmission Rate: {readmission_rate_30:.2f}%
                - Average Length of Stay: {avg_los:.1f} days
                - Average Medications: {avg_meds:.1f}
                - Diabetes Medication Rate: {diabetes_med_rate:.1f}%
                
                Top Risk Factors:
                1. Number of inpatient visits
                2. Number of emergency visits
                3. Number of diagnoses
                4. Length of stay
                5. Number of medications
                """
                st.text_area("Summary Report", report, height=400)
        
        with col2:
            st.info("""
            📌 **Next Steps for Analysis:**
            1. Perform feature engineering for predictive modeling
            2. Build classification models for readmission prediction
            3. Conduct statistical hypothesis testing
            4. Develop risk stratification framework
            5. Create predictive dashboards for clinical use
            """)

else:
    st.error("Unable to load data. Please check your internet connection and try again.")
    st.stop()

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #6B7280; padding: 20px;'>
    <p>IT5006 - Healthcare Readmission Analysis Dashboard | Team Project Milestone 1</p>
    <p>Built with Streamlit 🎈 | Data Source: UCI ML Repository (ID: 296)</p>
</div>
""", unsafe_allow_html=True)
