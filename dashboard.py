import streamlit as st
import pandas as pd
import altair as alt

# --- Configuration ---
st.set_page_config(page_title="Car Sentiment Dashboard", layout="wide")

# --- Main App ---
st.title("Electric Car Sentiment Dashboard")
st.markdown("Analyze customer sentiment and model mentions across multiple data sources.")

def adjust(df: pd.DataFrame):
  df['DATE'] = pd.to_datetime(df['DATE'])
  df = df.query('SERVICE != "Facebook" or BRAND != "Unknown"')
  return df

try:
    df_raw, df = adjust(pd.read_csv('./versions/raw.csv')), adjust(pd.read_csv('./versions/final.csv'))
except Exception as e:
    st.error(f"Error processing data: {e}")
    st.stop()

if df_raw.empty:
    st.error("No valid data found in the provided files.")
    st.stop()

# --- Sidebar Filters ---
st.sidebar.header("Filters")

# Brand Filter
available_brands = sorted(df_raw['BRAND'].dropna().unique().tolist())
brand_options = ["All Brands"] + available_brands
selected_brand = st.sidebar.selectbox("Select Brand", brand_options, index=0)

# Service Filter
available_services = sorted(df_raw['SERVICE'].dropna().unique().tolist())
service_options = ["All Services"] + available_services
selected_service = st.sidebar.selectbox("Select Service", service_options, index=0)

# Filter raw data by brand and service for subsequent filters
mask_raw_initial = pd.Series(True, index=df_raw.index)
if selected_brand != "All Brands":
    mask_raw_initial &= (df_raw['BRAND'] == selected_brand)
if selected_service != "All Services":
    mask_raw_initial &= (df_raw['SERVICE'] == selected_service)

df_raw_filtered_initial = df_raw[mask_raw_initial]

# Model Filter
# Find models actually present in the brand/service filtered data
available_models = sorted(set([m for models in df_raw_filtered_initial['MODELS_MENTIONED'] for m in models]))
selected_models = st.sidebar.multiselect(
    "Select Models", 
    available_models, 
    default=[] # Overview by default
)

# Date Filter
if not df_raw_filtered_initial.empty:
    min_date = df_raw_filtered_initial['DATE'].min().to_pydatetime()
    max_date = df_raw_filtered_initial['DATE'].max().to_pydatetime()
    selected_dates = st.sidebar.date_input(
        "Select Date Range",
        value=(min_date, max_date),
        min_value=min_date,
        max_value=max_date
    )
else:
    st.sidebar.warning("No data found for the selected criteria.")
    selected_dates = []

# Final Filtering Logic
mask_raw = mask_raw_initial.copy()

if len(selected_dates) == 2:
    start_date, end_date = pd.to_datetime(selected_dates[0]), pd.to_datetime(selected_dates[1])
    mask_raw &= (df_raw['DATE'] >= start_date) & (df_raw['DATE'] <= end_date)

# Model filter for raw data (KPIs, Charts)
if selected_models:
    # A post matches if any of its mentioned models is in the selected list
    mask_raw &= df_raw['MODELS_MENTIONED'].apply(
        lambda models: any(m in selected_models for m in models)
    )

filtered_raw_df = df_raw[mask_raw]

# For the "Top Models" chart, we use the exploded dataframe
# Ensure we only include models from the filtered posts
if 'ID' in filtered_raw_df.columns:
    filtered_df = df[df['ID'].isin(filtered_raw_df['ID'])]
else:
    filtered_df = df[df['CONTENT'].isin(filtered_raw_df['CONTENT'])]

if selected_models:
    filtered_df = filtered_df[filtered_df['MODELS_MENTIONED'].isin(selected_models)]

# --- Dashboard Layout ---

if filtered_raw_df.empty:
    st.warning("⚠️ No data matching your current filters. Please adjust your criteria.")
else:
    # Metric Row
    col1, col2, col3, col4 = st.columns(4)
    total_posts = len(filtered_raw_df)
    pos_count = len(filtered_raw_df[filtered_raw_df['SENTIMENT'] == 'POSITIVE'])
    neg_count = len(filtered_raw_df[filtered_raw_df['SENTIMENT'] == 'NEGATIVE'])

    col1.metric("Total Posts", total_posts)
    col2.metric("Positive Sentiment", f"{(pos_count/total_posts*100):.1f}%" if total_posts > 0 else "0%")
    col3.metric("Negative Sentiment", f"{(neg_count/total_posts*100):.1f}%" if total_posts > 0 else "0%")
    if not filtered_raw_df.empty:
        col4.metric("Date Range", f"{(filtered_raw_df['DATE'].max() - filtered_raw_df['DATE'].min()).days} days")

    st.divider()

    # Charts Row
    chart_col1, chart_col2 = st.columns(2)

    with chart_col1:
        st.subheader("Sentiment Distribution")
        sentiment_data = filtered_raw_df['SENTIMENT'].value_counts().reset_index()
        sentiment_data.columns = ['Sentiment', 'Count']
        
        pie_chart = alt.Chart(sentiment_data).mark_arc(innerRadius=50).encode(
            theta=alt.Theta(field="Count", type="quantitative"),
            color=alt.Color(field="Sentiment", type="nominal", scale=alt.Scale(domain=['POSITIVE', 'NEGATIVE', 'NEUTRAL'], range=['#2ecc71', '#e74c3c', '#f1c40f'])),
            tooltip=['Sentiment', 'Count']
        ).properties(height=300)
        st.altair_chart(pie_chart, width='stretch')

    with chart_col2:
        st.subheader("Top Models")
        model_data = filtered_df[filtered_df['MODELS_MENTIONED'] != 'General / Other']['MODELS_MENTIONED'].value_counts().reset_index()
        model_data.columns = ['Model', 'Count']
        
        if not model_data.empty:
            bar_chart = alt.Chart(model_data).mark_bar(cornerRadiusTopLeft=3, cornerRadiusTopRight=3).encode(
                x=alt.X('Count:Q', title='Number of Mentions'),
                y=alt.Y('Model:N', sort='-x', title='Car Model'),
                color=alt.value("#3498db"),
                tooltip=['Model', 'Count']
            ).properties(height=300)
            st.altair_chart(bar_chart, width='stretch')
        else:
            st.info("No specific models identified in the current selection.")

    # Sentiment over Time
    st.subheader("Sentiment Over Time")
    time_data = filtered_raw_df.groupby([filtered_raw_df['DATE'].dt.date, 'SENTIMENT']).size().reset_index(name='Count')
    time_data.columns = ['Date', 'Sentiment', 'Count']

    line_chart = alt.Chart(time_data).mark_line(point=True).encode(
        x=alt.X('Date:T', title='Date'),
        y=alt.Y('Count:Q', title='Posts'),
        color=alt.Color('Sentiment:N', scale=alt.Scale(domain=['POSITIVE', 'NEGATIVE'], range=['#2ecc71', '#e74c3c'])),
        tooltip=['Date', 'Sentiment', 'Count']
    ).properties(height=400).interactive()
    st.altair_chart(line_chart, width='stretch')

    # Data Table
    st.subheader("Detailed Comments")
    st.dataframe(
        filtered_raw_df[['DATE', 'BRAND', 'SERVICE', 'CONTENT', 'SENTIMENT']].sort_values(by='DATE', ascending=False),
        width='stretch',
        hide_index=True
    )

st.sidebar.markdown("---")
st.sidebar.info("Data processed and cleaned via data_processing.py")
