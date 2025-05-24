import streamlit as st
import pandas as pd
from datetime import datetime, timedelta
from src.pipeline.predict_pipeline import PredictPipeline, CustomData
from src.logger import logging
import openai
import json

# Configure page
st.set_page_config(page_title="Facebook Ad Analytics & AI Assistant", layout="wide")

# Title of the Streamlit app
st.title("🚀 Facebook Ad Analytics & AI Assistant")
st.write("Predict conversions and get AI-powered insights about your Facebook advertising campaigns.")

# Sidebar for navigation
st.sidebar.title("Navigation")
app_mode = st.sidebar.selectbox("Choose Mode", ["Prediction Tool", "AI Chat Assistant"])

if app_mode == "Prediction Tool":
    # Original prediction functionality
    st.header("📊 Facebook Ad Conversion Prediction")
    
    # Input fields for user data
    st.sidebar.header("Campaign Information")

    # Campaign details
    campaign_id = st.sidebar.number_input("Campaign ID", min_value=1, value=1234, step=1)
    fb_campaign_id = st.sidebar.number_input("Facebook Campaign ID", min_value=1, value=5678, step=1)

    # Date selection
    st.sidebar.subheader("Campaign Duration")
    start_date = st.sidebar.date_input("Campaign Start Date", datetime.now() - timedelta(days=7))
    end_date = st.sidebar.date_input("Campaign End Date", datetime.now())

    # Target demographics
    st.sidebar.subheader("Target Demographics")
    age = st.sidebar.selectbox("Age Group", ["13-17", "18-24", "25-34", "35-44", "45-54", "55-64", "65+"])
    gender = st.sidebar.selectbox("Gender", ["M", "F"])

    # Interest targeting
    st.sidebar.subheader("Interest Targeting")
    interest1 = st.sidebar.number_input("Interest 1 ID", min_value=1, value=100, step=1)
    interest2 = st.sidebar.number_input("Interest 2 ID", min_value=1, value=101, step=1)
    interest3 = st.sidebar.number_input("Interest 3 ID", min_value=1, value=102, step=1)

    # Campaign metrics
    st.sidebar.subheader("Campaign Metrics")
    impressions = st.sidebar.number_input("Impressions", min_value=0.0, value=10000.0, step=100.0)
    clicks = st.sidebar.number_input("Clicks", min_value=0, value=500, step=1)
    spent = st.sidebar.number_input("Amount Spent ($)", min_value=0.0, value=100.0, step=1.0)
    total_conversion = st.sidebar.number_input("Total Conversions", min_value=0.0, value=25.0, step=1.0)

    # Display campaign summary
    st.subheader("Campaign Summary")
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Impressions", f"{impressions:,.0f}")
    with col2:
        st.metric("Clicks", f"{clicks:,}")
    with col3:
        st.metric("Spent", f"${spent:,.2f}")
    with col4:
        st.metric("Total Conversions", f"{total_conversion:,.0f}")

    # Calculate and display key performance indicators
    st.subheader("Key Performance Indicators")
    col1, col2, col3 = st.columns(3)

    with col1:
        ctr = (clicks / impressions * 100) if impressions > 0 else 0
        st.metric("CTR (Click-Through Rate)", f"{ctr:.2f}%")
    with col2:
        cpc = (spent / clicks) if clicks > 0 else 0
        st.metric("CPC (Cost per Click)", f"${cpc:.2f}")
    with col3:
        cvr = (total_conversion / clicks * 100) if clicks > 0 else 0
        st.metric("CVR (Conversion Rate)", f"{cvr:.2f}%")

    # Predict button
    if st.button("Predict Approved Conversions"):
        try:
            # Convert dates to strings
            reporting_start = start_date.strftime("%Y-%m-%d")
            reporting_end = end_date.strftime("%Y-%m-%d")
            
            # Create CustomData object with input values
            data = CustomData(
                campaign_id=campaign_id,
                fb_campaign_id=fb_campaign_id,
                age=age,
                gender=gender,
                interest1=interest1,
                interest2=interest2,
                interest3=interest3,
                impressions=impressions,
                clicks=clicks,
                spent=spent,
                total_conversion=total_conversion,
                reporting_start=reporting_start,
                reporting_end=reporting_end
            )
            
            # Get prediction
            predict_pipeline = PredictPipeline()
            pred_df = data.get_data_as_data_frame()
            prediction = predict_pipeline.predict(pred_df)
            
            # Display prediction
            st.success(f"🎯 Predicted Approved Conversions: {prediction[0]:.0f}")
            
            # Store prediction in session state for AI chat
            st.session_state['last_prediction'] = {
                'approved_conversions': prediction[0],
                'campaign_metrics': {
                    'impressions': impressions,
                    'clicks': clicks,
                    'spent': spent,
                    'total_conversion': total_conversion,
                    'ctr': ctr,
                    'cpc': cpc,
                    'cvr': cvr
                }
            }
            
            # Log the prediction
            logging.info(f"Prediction made for campaign {campaign_id}: {prediction[0]:.0f}")
            
        except Exception as e:
            st.error(f"An error occurred during prediction: {str(e)}")
            logging.error(f"Prediction error: {str(e)}")

elif app_mode == "AI Chat Assistant":
    st.header("🤖 AI Chat Assistant for Facebook Advertising")
    
    # API Configuration
    st.sidebar.header("🔑 AI Configuration")
    api_key = st.sidebar.text_input("Enter your OpenAI API Key", type="password", help="Your API key is not stored and only used for this session")
    
    model_options = ["gpt-3.5-turbo", "gpt-4", "gpt-4-turbo-preview", "gpt-3.5-turbo-16k"]
    selected_model = st.sidebar.selectbox("Choose OpenAI Model", model_options)
    
    # System context about your model and data
    system_context = """
    You are an AI assistant specialized in Facebook advertising analytics. You have access to a machine learning model that predicts approved conversions for Facebook ad campaigns.

    Here's what you know about the model and data:
    
    **Dataset Information:**
    - The model works with Facebook advertising campaign data
    - Key features include: campaign_id, fb_campaign_id, age groups, gender, interest targeting (interest1, interest2, interest3)
    - Campaign metrics: impressions, clicks, spent, total_conversion
    - Target variable: approved_conversion (what the model predicts)
    
    **Model Features:**
    - Predicts approved conversions based on campaign parameters
    - Uses features like CTR (Click-Through Rate), CPC (Cost Per Click), conversion rates
    - Handles demographic targeting (age groups: 13-17, 18-24, 25-34, 35-44, 45-54, 55-64, 65+)
    - Gender targeting (M/F)
    - Interest-based targeting with multiple interest IDs
    
    **Key Metrics Calculated:**
    - CTR = (clicks / impressions) * 100
    - CPC = spent / clicks
    - CVR = total_conversion / clicks * 100
    
    You can help users with:
    1. Understanding their Facebook ad performance
    2. Interpreting prediction results
    3. Optimizing campaigns based on the model insights
    4. Explaining advertising metrics and KPIs
    5. Providing recommendations for better campaign performance
    
    Always provide practical, actionable advice based on Facebook advertising best practices and the model's capabilities.
    """
    
    if api_key:
        try:
            # Initialize OpenAI client
            openai.api_key = api_key
            
            # Initialize chat history
            if "messages" not in st.session_state:
                st.session_state.messages = []
            
            # Display chat history
            for message in st.session_state.messages:
                with st.chat_message(message["role"]):
                    st.markdown(message["content"])
            
            # Chat input
            if prompt := st.chat_input("Ask me anything about Facebook advertising or your campaign predictions..."):
                # Add user message to chat history
                st.session_state.messages.append({"role": "user", "content": prompt})
                with st.chat_message("user"):
                    st.markdown(prompt)
                
                # Prepare context with recent prediction if available
                context_addition = ""
                if 'last_prediction' in st.session_state:
                    pred_data = st.session_state['last_prediction']
                    context_addition = f"""
                    
                    **Recent Prediction Context:**
                    The user just made a prediction with the following results:
                    - Predicted Approved Conversions: {pred_data['approved_conversions']:.0f}
                    - Campaign Metrics:
                      * Impressions: {pred_data['campaign_metrics']['impressions']:,.0f}
                      * Clicks: {pred_data['campaign_metrics']['clicks']:,}
                      * Spent: ${pred_data['campaign_metrics']['spent']:,.2f}
                      * Total Conversions: {pred_data['campaign_metrics']['total_conversion']:.0f}
                      * CTR: {pred_data['campaign_metrics']['ctr']:.2f}%
                      * CPC: ${pred_data['campaign_metrics']['cpc']:.2f}
                      * CVR: {pred_data['campaign_metrics']['cvr']:.2f}%
                    """
                
                # Generate AI response
                with st.chat_message("assistant"):
                    try:
                        response = openai.ChatCompletion.create(
                            model=selected_model,
                            messages=[
                                {"role": "system", "content": system_context + context_addition},
                                *st.session_state.messages
                            ],
                            temperature=0.7,
                            max_tokens=1000
                        )
                        
                        ai_response = response.choices[0].message.content
                        st.markdown(ai_response)
                        
                        # Add AI response to chat history
                        st.session_state.messages.append({"role": "assistant", "content": ai_response})
                        
                    except Exception as e:
                        st.error(f"Error generating AI response: {str(e)}")
                        st.info("Please check your API key and try again.")
            
            # Clear chat button
            if st.button("🗑️ Clear Chat History"):
                st.session_state.messages = []
                st.rerun()
                
        except Exception as e:
            st.error(f"Error initializing AI chat: {str(e)}")
    else:
        st.info("👆 Please enter your OpenAI API key in the sidebar to start chatting with the AI assistant.")
        
        # Show example questions
        st.subheader("💡 Example Questions You Can Ask:")
        example_questions = [
            "How can I improve my Facebook ad CTR?",
            "What does my predicted conversion rate mean?",
            "How should I optimize my campaign budget?",
            "What age group typically performs best for e-commerce ads?",
            "How do I interpret my CPC and is it good?",
            "What factors influence approved conversions the most?",
            "Should I increase my daily budget based on these metrics?",
            "How can I better target my audience?"
        ]
        
        for i, question in enumerate(example_questions, 1):
            st.write(f"{i}. {question}")

# Add footer with additional information
st.markdown("---")
st.markdown("""
    #### 📋 Notes:
    - **CTR** indicates the percentage of people who clicked on your ad after seeing it
    - **CPC** shows how much you're paying for each click
    - **CVR** shows the percentage of clicks that resulted in conversions
    - **AI Assistant** can help you interpret results and optimize your campaigns
    - Your API key is only used for this session and is not stored anywhere
""")

# Add some CSS for better styling
st.markdown("""
<style>
    .stMetric {
        background-color: #f0f2f6;
        border: 1px solid #e0e0e0;
        padding: 10px;
        border-radius: 5px;
    }
    .chat-message {
        padding: 10px;
        border-radius: 10px;
        margin: 5px 0;
    }
</style>
""", unsafe_allow_html=True)
