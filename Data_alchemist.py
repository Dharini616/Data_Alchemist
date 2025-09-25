# Filename: combined_app.py

import streamlit as st
import pandas as pd
import plotly.express as px
import json
import numpy as np
import warnings
import requests
import torch
import torch.nn as nn
from io import BytesIO
from sklearn.preprocessing import MinMaxScaler
from sklearn.impute import KNNImputer
import re
from pandasql import sqldf
import os 


# Suppress warnings for a cleaner interface
warnings.filterwarnings("ignore")

# --- Deep Learning Model for Time Series Forecasting (from trying.py) ---
# Using LSTM for more advanced and accurate time-series predictions.
class LSTMForecaster(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(LSTMForecaster, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])
        return out


# ----------------------------
# Download helpers
# ----------------------------

def download_csv(df, key_suffix=""):
    csv = df.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="Download Cleaned Data as CSV",
        data=csv,
        file_name='cleaned_data.csv',
        mime='text/csv',
        key=f'csv-download-button-{key_suffix}'
    )

def download_excel(df, key_suffix=""):
    output = BytesIO()
    writer = pd.ExcelWriter(output, engine='xlsxwriter')
    df.to_excel(writer, index=False, sheet_name='Sheet1')
    writer.close()
    processed_data = output.getvalue()
    st.download_button(
        label="Download Cleaned Data as Excel",
        data=processed_data,
        file_name='cleaned_data.xlsx',
        mime='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
        key=f'excel-download-button-{key_suffix}'
    )

# --- LLM API Configuration (from trying.py) ---
# NOTE: Replace with your actual Google AI API Key.
API_KEY = "AIzaSyApsQOK9ajYH49sHNF0qs2xuTZFWfMW2eY"  # IMPORTANT: Use your own API key
API_URL = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent?key={API_KEY}"

def call_llm(user_query, df_columns, last_bot_question):
    """
    Sends the user query and data context to the Gemini LLM and returns a structured JSON command.
    This prompt is improved to understand all features from both files.
    """
    system_instruction = f"""
    You are an expert data analysis assistant. Your task is to interpret a user's request about a pandas DataFrame and translate it into a single, structured JSON object. You must not add any explanatory text, only the JSON.

    The DataFrame has these columns: {', '.join(df_columns)}.
    The last question the bot asked was: "{last_bot_question}". Use this for context, especially for follow-up answers like "yes" or "use the mean".

    Recognize the user's intent and entities. Your JSON output must have an "intent" and an "entities" object.

    Here are the possible intents and their entities:

    1. intent: "get_summary" -> For a general overview of the data.
       entities: {{}}
       Example: "give me a summary"

    2. intent: "calculate_statistic" -> For calculations like sum, mean, max, min, count, median.
       entities: {{"statistic": "mean", "column": "age"}}
       Example: "what is the average age"

    3. intent: "calculate_percentage" -> For finding the percentage of a specific value in a column.
       entities: {{"column": "country", "value": "US"}}
       Example: "what percentage are from the US"

    4. intent: "plot_chart" -> For creating visualizations (bar, pie, histogram, line, scatter, box, treemap, stackedbar, bubble).
       entities: {{"chart_type": "bar", "columns": ["region", "sales"]}}
       Example: "plot a bar chart of sales by region"

    5. intent: "manage_data" -> For data cleaning.
       action can be: "drop_column", "fill_na", "fill_all_na".
       method can be: "mean", "median", "NA", "0".
       Example 1: "drop the customer_id column" -> {{"intent": "manage_data", "entities": {{"action": "drop_column", "column": "customer_id"}}}}
       Example 2: "fill missing age with the median" -> {{"intent": "manage_data", "entities": {{"action": "fill_na", "column": "age", "method": "median"}}}}

    6. intent: "show_rows" -> To display the first or last few rows.
       action can be: "head" or "tail".
       Example: "show me the first 10 rows" -> {{"intent": "show_rows", "entities": {{"action": "head", "number": 10}}}}

    7. intent: "filter_data" -> To filter the dataset based on a condition.
       Example: "show me data where sales > 500" -> {{"intent": "filter_data", "entities": {{"column": "sales", "operator": ">", "value": "500"}}}}

    8. intent: "concatenate_columns" -> To combine two or more columns.
       Example: "combine first name and last name into full_name" -> {{"intent": "concatenate_columns", "entities": {{"columns": ["first name", "last name"], "new_column_name": "full_name"}}}}

    9. intent: "predictive_analysis" -> For time series forecasting.
       Example: "predict future sales" -> {{"intent": "predictive_analysis", "entities": {{"target": "sales"}}}}

    10. intent: "unknown" -> If the request is unclear.
    """
    
    headers = {'Content-Type': 'application/json'}
    payload = {
        "contents": [{"parts": [{"text": user_query}]}],
        "systemInstruction": {"parts": [{"text": system_instruction}]},
        "generationConfig": {"responseMimeType": "application/json"}
    }
    
    try:
        response = requests.post(API_URL, headers=headers, json=payload)
        response.raise_for_status()
        json_text = response.json()["candidates"][0]["content"]["parts"][0]["text"]
        return json.loads(json_text)
    except Exception as e:
        st.error(f"LLM API Error: {e}")
        st.error("Please check your API key and ensure the Gemini API is enabled.")
        return {"intent": "unknown", "entities": {}}


# --- Streamlit UI and Session State ---
st.set_page_config(layout="wide", page_title="Advanced Data Bot")
st.title("Data Alchemist ")

if "chat_history" not in st.session_state:
    st.session_state.chat_history = [{"role": "bot", "content": "Hello! I am ready to help you analyze your data. Please upload a CSV or Excel file to begin."}]
if "df" not in st.session_state:
    st.session_state.df = None
if "last_bot_question" not in st.session_state:
    st.session_state.last_bot_question = ""

# --- File Upload and Data Loading ---
with st.sidebar:
    st.header("Upload Your Data")
    uploaded_file = st.file_uploader("Choose a CSV or Excel file", type=["csv", "xlsx"])

if uploaded_file and st.session_state.df is None:
    try:
        if uploaded_file.name.endswith('.csv'):
            st.session_state.df = pd.read_csv(uploaded_file)
        else:
            st.session_state.df = pd.read_excel(uploaded_file)
        
        initial_message = f"Successfully loaded '{uploaded_file.name}'. It has {st.session_state.df.shape[0]} rows and {st.session_state.df.shape[1]} columns. What would you like to do?"
        st.session_state.chat_history.append({"role": "bot", "content": initial_message})
        
        # This logic from basic.py is useful for proactively handling missing data
        na_counts = st.session_state.df.isnull().sum()
        columns_with_na = [col for col, count in na_counts.items() if count > 0]
        if columns_with_na:
            na_message = f"I've detected missing values in: {', '.join(columns_with_na)}. Would you like to handle them?"
            st.session_state.chat_history.append({"role": "bot", "content": na_message})
            st.session_state.last_bot_question = na_message
        else:
            st.session_state.last_bot_question = initial_message

    except Exception as e:
        st.session_state.chat_history.append({"role": "bot", "content": f"Error loading file: {e}"})

# Sidebar upload + manual cleaning tools
with st.sidebar:

    if st.session_state.df is not None:
        st.header("Manual Cleaning & Transform")
        # Missing values handling
        with st.expander("Handle Missing Values"):
            missing_col = st.selectbox("Select column", options=st.session_state.df.columns, key='missing_col')
            missing_method = st.selectbox("Method", options=['Mean', 'Median', 'Mode', 'Drop Column', 'NA', '0', 'KNN Imputation', 'Interpolation'], key='missing_method')
            if st.button("Apply Missing Handling"):
                if missing_method == 'Drop Column':
                    st.session_state.df = st.session_state.df.drop(columns=[missing_col])
                    st.success(f"Dropped {missing_col}")
                else:
                    try:
                        temp_df = st.session_state.df.copy()
                        if missing_method == 'Mean':
                            temp_df[missing_col] = temp_df[missing_col].fillna(temp_df[missing_col].mean())
                        elif missing_method == 'Median':
                            temp_df[missing_col] = temp_df[missing_col].fillna(temp_df[missing_col].median())
                        elif missing_method == 'Mode':
                            temp_df[missing_col] = temp_df[missing_col].fillna(temp_df[missing_col].mode()[0])
                        elif missing_method == 'NA':
                            temp_df[missing_col] = temp_df[missing_col].fillna("NA")
                        elif missing_method == '0':
                            temp_df[missing_col] = temp_df[missing_col].fillna(0)
                        elif missing_method == 'KNN Imputation':
                            imputer = KNNImputer(n_neighbors=5)
                            num_cols = temp_df.select_dtypes(include=np.number).columns
                            temp_df[num_cols] = imputer.fit_transform(temp_df[num_cols])
                        elif missing_method == 'Interpolation':
                            temp_df[missing_col] = temp_df[missing_col].interpolate()
                        st.session_state.df = temp_df
                        st.success(f"Applied '{missing_method}' to {missing_col}")
                    except Exception as e:
                        st.error(f"Error: {e}")

        # Duplicates
        with st.expander("Duplicates"):
            if st.button("Remove All Duplicate Rows"):
                initial = st.session_state.df.shape[0]
                st.session_state.df = st.session_state.df.drop_duplicates()
                removed = initial - st.session_state.df.shape[0]
                st.success(f"Removed {removed} duplicates.")

        # Text cleaning
        with st.expander("Text Cleaning"):
            text_col = st.selectbox("Text column", options=st.session_state.df.columns, key='text_col')
            if st.button("Convert to Lowercase"):
                st.session_state.df[text_col] = st.session_state.df[text_col].astype(str).str.lower()
                st.success(f"{text_col} -> lowercase")
            if st.button("Remove Special Characters"):
                st.session_state.df[text_col] = st.session_state.df[text_col].astype(str).apply(lambda x: re.sub(r'[^a-zA-Z0-9\s]', '', x))
                st.success("Removed special characters")

        # Data type conversion
        with st.expander("Change Data Type"):
            dtype_col = st.selectbox("Column", options=st.session_state.df.columns, key='dtype_col')
            new_dtype = st.selectbox("New type", options=['int', 'float', 'str', 'datetime'], key='new_dtype')
            if st.button("Apply Data Type Change"):
                try:
                    if new_dtype == 'datetime':
                        st.session_state.df[dtype_col] = pd.to_datetime(st.session_state.df[dtype_col])
                    else:
                        st.session_state.df[dtype_col] = st.session_state.df[dtype_col].astype(new_dtype)
                    st.success(f"Changed {dtype_col} to {new_dtype}")
                except Exception as e:
                    st.error(f"Failed: {e}")

        # Outlier treatment
        with st.expander("Outlier Treatment (IQR)"):
            num_cols = st.session_state.df.select_dtypes(include=np.number).columns.tolist()
            if num_cols:
                out_col = st.selectbox("Numeric column", options=num_cols, key='out_col')
                if st.button("Remove Outliers (IQR)"):
                    try:
                        q1 = st.session_state.df[out_col].quantile(0.25)
                        q3 = st.session_state.df[out_col].quantile(0.75)
                        iqr = q3 - q1
                        lower = q1 - 1.5 * iqr
                        upper = q3 + 1.5 * iqr
                        before = st.session_state.df.shape[0]
                        st.session_state.df = st.session_state.df[(st.session_state.df[out_col] >= lower) & (st.session_state.df[out_col] <= upper)]
                        removed = before - st.session_state.df.shape[0]
                        st.success(f"Removed {removed} outliers")
                    except Exception as e:
                        st.error(f"Error: {e}")

        # Downloads
        st.header("Download Cleaned Data")
        download_csv(st.session_state.df, "sidebar")
        download_excel(st.session_state.df, "sidebar")

# --- Functions to Perform Tasks (Combined from both files) ---

def get_summary_response(df):
    """Provides a detailed summary of the DataFrame."""
    summary = df.describe(include='all').to_markdown()
    col_details = (
        f"### Column Details\n"
        f"**Number of Rows:** {df.shape[0]}\n"
        f"**Number of Columns:** {df.shape[1]}\n"
    )
    return f"Here is a summary of the dataset:\n\n{summary}\n\n{col_details}"

def calculate_statistic_response(df, statistic, column):
    """Calculates a specific statistic for a column."""
    if not column: return "I need a column name for the calculation. Please specify one."
    if column not in df.columns: return f"Column '{column}' not found. Please check the name."
    
    try:
        if statistic == 'count':
            result = df[column].count()
            return f"The count of non-null values in '{column}' is **{result}**."
        
        numeric_col = pd.to_numeric(df[column], errors='coerce')
        if numeric_col.isnull().all():
            return f"Column '{column}' is not numeric, so I can't calculate the {statistic}."
        
        result = getattr(numeric_col, statistic)()
        return f"The **{statistic}** of column '{column}' is **{result:.2f}**."
    except Exception as e: return f"Error calculating statistic: {e}"

def calculate_percentage_response(df, column, value):
    """Calculates the percentage of a given value in a column."""
    if not column or not value: return "Please specify a column and a value to calculate the percentage."
    if column not in df.columns: return f"Column '{column}' not found."
    
    try:
        total_count = len(df[column])
        value_count = df[df[column].astype(str).str.lower() == str(value).lower()].shape[0]
        percentage = (value_count / total_count) * 100 if total_count > 0 else 0
        return f"The percentage of '{value}' in the '{column}' column is **{percentage:.2f}%**."
    except Exception as e: return f"Error calculating percentage: {e}"

def plot_chart_response(df, chart_type, columns):
    """Generates a Plotly chart based on user request."""
    if not columns: return "I need at least one column to create a chart."
    
    try:
        fig = None
        if chart_type == 'bar':
            if len(columns) < 2: return "A bar chart requires an X and a Y column."
            fig = px.bar(df, x=columns[0], y=columns[1], title=f"Bar Chart of {columns[1]} by {columns[0]}")
        elif chart_type == 'pie':
            fig = px.pie(df, names=columns[0], title=f"Pie Chart for {columns[0]}")
        elif chart_type == 'histogram':
            fig = px.histogram(df, x=columns[0], title=f"Histogram of {columns[0]}")
        elif chart_type == 'line':
            if len(columns) < 2: return "A line chart requires an X and a Y column."
            fig = px.line(df, x=columns[0], y=columns[1], title=f"Line Chart of {columns[1]} over {columns[0]}")
        elif chart_type == 'scatter':
            if len(columns) < 2: return "A scatter plot requires an X and a Y column."
            fig = px.scatter(df, x=columns[0], y=columns[1], title=f"Scatter Plot of {columns[1]} vs {columns[0]}")
        elif chart_type == 'scatter':
            if len(columns) < 2: return "A scatter plot requires two columns for X and Y axes."
            fig = px.scatter(df, x=columns[0], y=columns[1])
        elif chart_type == 'box':
            if len(columns) < 1: return "A box plot requires one column."
            fig = px.box(df, y=columns[0])
        elif chart_type == 'treemap':
            if len(columns) < 2: return "A treemap requires at least two columns for hierarchy."
            fig = px.treemap(df, path=columns)
        elif chart_type == 'stackedbar':
            if len(columns) < 3: return "A stacked bar chart requires at least three columns for X, Y, and color."
            fig = px.bar(df, x=columns[0], y=columns[1], color=columns[2])
        elif chart_type == 'bubble':
            if len(columns) < 3: return "A bubble chart requires at least three columns for X, Y, and size."
            fig = px.scatter(df, x=columns[0], y=columns[1], size=columns[2], color=columns[3] if len(columns) > 3 else None)
        # Add other chart types from your original code if needed...

        if fig:
            return {"type": "chart", "figure": fig}
        else:
            return "Could not create the requested plot. Please check chart type and column names."
    except Exception as e:
        return f"An error occurred while plotting: {e}"

def manage_data_response(df, action, column=None, method=None):
    """Handles data manipulation like dropping columns or filling NAs."""
    if action == "drop_column":
        if column and column in df.columns:
            st.session_state.df = df.drop(columns=[column])
            return f"The column '{column}' has been dropped."
        return "Please specify a valid column to drop."
    
    elif action == "fill_na":
        if not column or column not in df.columns: return "Please specify a valid column to fill."
        
        fill_value_map = {
            "mean": df[column].mean(),
            "median": df[column].median(),
            "0": 0,
            "NA": "NA"
        }
        if method in fill_value_map:
            try:
                fill_value = fill_value_map[method]
                st.session_state.df[column].fillna(fill_value, inplace=True)
                return f"Missing values in '{column}' filled with {method}."
            except Exception:
                return f"Cannot fill non-numeric column '{column}' with {method}."
        return "Please specify a valid fill method (mean, median, 0, or NA)."

    elif action == "fill_all_na":
        if method == "mean" or method == "median":
            numeric_cols = df.select_dtypes(include=np.number).columns
            fill_values = getattr(df[numeric_cols], method)()
            st.session_state.df.fillna(fill_values, inplace=True)
            return f"All numeric missing values have been filled with the {method}."
        elif method == "0":
            st.session_state.df.fillna(0, inplace=True)
            return "All missing values have been filled with 0."
        elif method == "NA":
            st.session_state.df.fillna("NA", inplace=True)
            return "All missing values have been filled with 'NA'."
    
    return "I couldn't perform that data management task."

def filter_data_response(df, column, operator, value):
    """Filters the DataFrame based on a condition."""
    if column not in df.columns: return f"Column '{column}' not found."
    try:
        query_str = f"`{column}` {operator} {float(value)}" if pd.api.types.is_numeric_dtype(df[column]) else f"`{column}` {operator} '{value}'"
        filtered_df = df.query(query_str)
        if filtered_df.empty: return f"No rows found matching the filter."
        return {"type": "data", "data": filtered_df, "content": f"Here is the filtered data ({filtered_df.shape[0]} rows):"}
    except Exception as e: return f"Error while filtering: {e}"

def concatenate_columns_response(df, columns, new_column_name):
    """Concatenates multiple columns into a new one."""
    if len(columns) < 2: return "Please specify at least two columns to concatenate."
    if not new_column_name: return "Please provide a name for the new column."
    try:
        st.session_state.df[new_column_name] = df[columns].astype(str).agg(' '.join, axis=1)
        return f"Successfully created new column '{new_column_name}'."
    except Exception as e: return f"Error concatenating columns: {e}"

def predictive_analysis_response(df, target_col):
    """Performs time series forecasting using an LSTM model."""
    if target_col not in df.columns: return f"Column '{target_col}' not found for prediction."
    
    try:
        # Simple data preparation for LSTM
        data = df[target_col].dropna().values.astype(float).reshape(-1, 1)
        if len(data) < 10: return "Not enough data points to make a forecast."
        
        scaler = MinMaxScaler(feature_range=(-1, 1))
        data_normalized = scaler.fit_transform(data)
        
        # Create sequences
        seq_length = 5
        x_list, y_list = [], []
        for i in range(len(data_normalized) - seq_length):
            x_list.append(data_normalized[i:i+seq_length])
            y_list.append(data_normalized[i+seq_length])
        
        x_train = torch.tensor(x_list).float()
        y_train = torch.tensor(y_list).float()

        model = LSTMForecaster(input_size=1, hidden_size=50, num_layers=1, output_size=1)
        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

        # Train the model
        for epoch in range(100):
            outputs = model(x_train)
            optimizer.zero_grad()
            loss = criterion(outputs, y_train)
            loss.backward()
            optimizer.step()
            
        # Make forecast
        last_seq = torch.tensor(data_normalized[-seq_length:]).view(1, seq_length, 1).float()
        with torch.no_grad():
            forecast_normalized = model(last_seq)
        forecast = scaler.inverse_transform(forecast_normalized.numpy())

        # Create plot
        forecast_index = pd.RangeIndex(start=len(df), stop=len(df) + len(forecast))
        fig = px.line(y=df[target_col], title=f"Forecast for {target_col}")
        fig.add_scatter(x=forecast_index, y=forecast.flatten(), mode='lines', name='Forecast')
        
        return {"type": "chart", "figure": fig}
    except Exception as e:
        return f"An error occurred during predictive analysis: {e}"

# --- Main Chatbot Logic ---
if user_input := st.chat_input("Enter your command here..."):
    st.session_state.chat_history.append({"role": "user", "content": user_input})
    
    if st.session_state.df is not None:
        with st.spinner("Thinking..."):
            df_cols = st.session_state.df.columns.tolist()
            llm_response = call_llm(user_input, df_cols, st.session_state.last_bot_question)
            intent = llm_response.get("intent")
            entities = llm_response.get("entities", {})
            
            response_content = None # To hold text, chart, or data
            
            if intent == "get_summary":
                response_content = get_summary_response(st.session_state.df)
            elif intent == "calculate_statistic":
                response_content = calculate_statistic_response(st.session_state.df, entities.get("statistic"), entities.get("column"))
            elif intent == "calculate_percentage":
                response_content = calculate_percentage_response(st.session_state.df, entities.get("column"), entities.get("value"))
            elif intent == "plot_chart":
                response_content = plot_chart_response(st.session_state.df, entities.get("chart_type"), entities.get("columns"))
            elif intent == "manage_data":
                response_content = manage_data_response(st.session_state.df, entities.get("action"), entities.get("column"), entities.get("method"))
            elif intent == "show_rows":
                num = entities.get("number", 5)
                df_slice = st.session_state.df.head(num) if entities.get("action") == "head" else st.session_state.df.tail(num)
                response_content = {"type": "data", "data": df_slice, "content": f"Showing the {'first' if entities.get('action') == 'head' else 'last'} {num} rows."}
            elif intent == "filter_data":
                response_content = filter_data_response(st.session_state.df, entities.get("column"), entities.get("operator"), entities.get("value"))
            elif intent == "concatenate_columns":
                response_content = concatenate_columns_response(st.session_state.df, entities.get("columns"), entities.get("new_column_name"))
            elif intent == "predictive_analysis":
                response_content = predictive_analysis_response(st.session_state.df, entities.get("target"))
            else: # Unknown intent
                response_content = "I'm sorry, I don't understand that command. Can you please rephrase?"

            # Append bot's response to chat history
            bot_message = {"role": "bot"}
            if isinstance(response_content, str):
                bot_message["content"] = response_content
                st.session_state.last_bot_question = response_content
            elif isinstance(response_content, dict) and response_content.get("type") == "chart":
                bot_message["content"] = "Here is your chart:"
                bot_message["chart"] = response_content["figure"]
                st.session_state.last_bot_question = ""
            elif isinstance(response_content, dict) and response_content.get("type") == "data":
                bot_message["content"] = response_content["content"]
                bot_message["data"] = response_content["data"]
                st.session_state.last_bot_question = ""
            
            st.session_state.chat_history.append(bot_message)

    else:
        st.session_state.chat_history.append({"role": "bot", "content": "Please upload a data file first."})

    st.rerun()

# --- Display Chat History (Final Rendering) ---
for message in st.session_state.chat_history:
    with st.chat_message(message["role"]):
        st.write(message.get("content", ""))
        if "chart" in message:
            st.plotly_chart(message["chart"], use_container_width=True)
        if "data" in message:
            st.dataframe(message["data"])

