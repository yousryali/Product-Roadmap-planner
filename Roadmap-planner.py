import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from io import BytesIO
from datetime import timedelta
from streamlit_chat import message

# Set up the page layout
st.set_page_config(layout="wide", page_title="Product Roadmap and Task Planner")

st.title("📈 Product Roadmap and Task Planner")

# --- Display both sections without asking for file initially ---
# Roadmap and Task Planner ready to load without file
st.subheader("📅 Gantt Chart (Product Roadmap)")

# Placeholder for the roadmap (this will display even before the file is uploaded)
roadmap_placeholder = st.empty()

# Initialize a flag to handle file upload later
file_uploaded = False
uploaded_file = None

# Show the file uploader button after displaying roadmap
if roadmap_placeholder:
    st.warning("⚠️ Please upload an Excel file to proceed with the roadmap and task planner.")
    uploaded_file = st.file_uploader("Choose an Excel file (must include sheets: **Features List** and **Sprint Definitions)", type=["xlsx"])

if uploaded_file:
    file_uploaded = True
    # --- Proceed to read the file and generate the roadmap and task planner once the file is uploaded ---
    try:
        xls = pd.ExcelFile(uploaded_file)
        features_df = xls.parse('Features List')
        sprints_df = xls.parse('Sprint Definitions')
    except ValueError as e:
        st.error(f"Error loading sheets: {e}")
        st.stop()

    # Prepare Dates for features and sprints (same logic as before)
    sprints_df['Start Date'] = pd.to_datetime(sprints_df['Start Date'], errors='coerce')
    sprints_df['End Date'] = pd.to_datetime(sprints_df['End Date'], errors='coerce')

    features_merged = features_df.copy()
    features_merged = features_merged.merge(
        sprints_df[['Sprint Name', 'Start Date']],
        left_on='Target Start Sprint',
        right_on='Sprint Name',
        how='left'
    ).rename(columns={'Start Date': 'Start'})

    features_merged = features_merged.merge(
        sprints_df[['Sprint Name', 'End Date']],
        left_on='Target End Sprint',
        right_on='Sprint Name',
        how='left'
    ).rename(columns={'End Date': 'Finish'})

    features_merged.drop(columns=['Sprint Name_x', 'Sprint Name_y'], inplace=True)
    features_merged['Feature ID'] = features_merged['Feature ID'].astype(str)

    # --- Create the roadmap Gantt chart ---
    fig = px.timeline(
        features_merged,
        x_start='Start',
        x_end='Finish',
        y=features_merged.index,
        color='Target Release',
        hover_data=['Story Points', 'Target Start Sprint', 'Target End Sprint'],
        color_discrete_sequence=px.colors.qualitative.Set1
    )

    fig.update_xaxes(
        tickformat="%b %d, %Y",
        tickmode="array",
        tickvals=pd.date_range(
            start=features_merged['Start'].min(),
            end=features_merged['Finish'].max(),
            freq="W-MON"
        )
    )

    sprint_line_y = len(features_merged) + 2
    sprints_df['Sprint Center'] = sprints_df['Start Date'] + (sprints_df['End Date'] - sprints_df['Start Date']) / 2

    for _, row in sprints_df.iterrows():
        x0, x1 = row['Start Date'], row['End Date']
        x_center = row['Sprint Center']
        y_pos = sprint_line_y

        fig.add_trace(go.Scatter(x=[x0, x1], y=[y_pos, y_pos], mode='lines',
                                 line=dict(color='black', width=2), showlegend=False))

        fig.add_trace(go.Scatter(x=[x0, x1], y=[y_pos, y_pos], mode='markers',
                                 marker=dict(symbol="triangle-down", size=10, color='black'), showlegend=False))

        fig.add_trace(go.Scatter(x=[x_center], y=[y_pos + 0.4], mode='text',
                                 text=[row['Sprint Name']], textposition='top center', showlegend=False))

    fig.update_layout(
        title='🗓️ Feature Timeline by Sprint (Sorted by Release)',
        xaxis_title='Date',
        yaxis_title='Feature ID',
        yaxis=dict(
            tickvals=list(range(len(features_merged))),
            ticktext=features_merged['Feature ID'].tolist()
        ),
        height=900,
        showlegend=True
    )

    st.plotly_chart(fig, use_container_width=True)

    # --- Proceed with the tasks planner ---
    st.subheader("📝 Product Tasks Planner")

    # Function to generate tasks for each product
    def generate_tasks(row):
        product_version = row['Title']
        qa_date = row['QA Deployment Date']  # This is 'n'
        uat_date = row['UAT Deployment Date']
        prod_date = row['Prod Date']

        tasks = [
            {"Task Name": f"{product_version} Initial QA Deployment", "Start Date": qa_date, "Due Date": qa_date},
            {"Task Name": f"{product_version} Internal DB Scripts Review", "Start Date": qa_date + timedelta(days=4), "Due Date": qa_date + timedelta(days=4)},
            {"Task Name": f"{product_version} External DB Scripts Review", "Start Date": qa_date + timedelta(days=4), "Due Date": qa_date + timedelta(days=4)},
            {"Task Name": f"{product_version} CAB", "Start Date": qa_date + timedelta(days=4), "Due Date": qa_date + timedelta(days=4)},
            {"Task Name": f"{product_version} Prod Prep", "Start Date": qa_date + timedelta(days=5), "Due Date": qa_date + timedelta(days=5)},
            {"Task Name": f"{product_version} Initial UAT Deployment", "Start Date": uat_date, "Due Date": uat_date},
            {"Task Name": f"{product_version} Prod Deployment", "Start Date": prod_date, "Due Date": qa_date}
        ]

        for task in tasks:
            task["Duration (Days)"] = (task["Start Date"] - task["Due Date"]).days

        return tasks

    # File upload section for product tasks
    uploaded_file_tasks = st.file_uploader("Upload an Excel file for Product Tasks", type=["xlsx"])

    if uploaded_file_tasks is not None:
        df_tasks = pd.read_excel(uploaded_file_tasks)

        # Ensure the date columns are in datetime format
        df_tasks['QA Deployment Date'] = pd.to_datetime(df_tasks['QA Deployment Date'], errors='coerce')
        df_tasks['UAT Deployment Date'] = pd.to_datetime(df_tasks['UAT Deployment Date'], errors='coerce')
        df_tasks['Prod Date'] = pd.to_datetime(df_tasks['Prod Date'], errors='coerce')

        # Extract Product Name
        df_tasks['Product'] = df_tasks['Title'].apply(lambda x: str(x).split()[0] if pd.notna(x) else "Unknown")

        # Button to generate tasks
        if st.button("Generate Tasks for Products"):
            task_data = []
            for product in df_tasks['Product'].unique():
                product_df = df_tasks[df_tasks['Product'] == product]
                for _, row in product_df.iterrows():
                    task_data.extend(generate_tasks(row))

            task_df = pd.DataFrame(task_data)
            st.write("Task Planner", task_df)

            # Create download button for the new task file
            task_output = BytesIO()
            with pd.ExcelWriter(task_output, engine='xlsxwriter') as writer:
                task_df.to_excel(writer, sheet_name="Product Tasks", index=False)

            st.download_button(
                label="Download Product Task Planner",
                data=task_output.getvalue(),
                file_name="product_task_planner.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )
