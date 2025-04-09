import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from io import BytesIO

st.set_page_config(layout="wide", page_title="Product Roadmap Planner")

st.title("📈 Product Roadmap Planner")
st.markdown("Upload your Excel file (must include sheets: **Features List** and **Sprint Definitions**)")

uploaded_file = st.file_uploader("Choose an Excel file", type=["xlsx"])

# === Provide a Template Download Option for New Users ===
if uploaded_file is None:
    st.warning("⚠️ Please upload an Excel file to proceed.")
    
    # Template content (you can customize this to suit the structure of the file you're working with)
    template_data = {
        'Feature ID': [''],
        'Target Release': [''],
        'Target Start Sprint': [''],
        'Target End Sprint': [''],
        'Story Points': [0],
    }
    
    template_df = pd.DataFrame(template_data)
    
    # Save the template to a BytesIO buffer
    template_output = BytesIO()
    with pd.ExcelWriter(template_output, engine='xlsxwriter') as writer:
        template_df.to_excel(writer, sheet_name="Features List", index=False)
        sprints_template = pd.DataFrame({
            'Sprint Name': ['Sprint 1', 'Sprint 2'],
            'Start Date': ['2025-05-01', '2025-06-01'],
            'End Date': ['2025-05-14', '2025-06-14'],
            'Sprint Capacity': [50, 50]
        })
        sprints_template.to_excel(writer, sheet_name="Sprint Definitions", index=False)
    
    template_output.seek(0)  # Move to the beginning of the BytesIO buffer

    st.download_button(
        label="📥 Download Template",
        data=template_output,
        file_name="roadmap_template.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )
    st.stop()  # Stop further execution since no file has been uploaded

# === Proceed with Processing if File is Uploaded ===
else:
    # === Read Excel Sheets ===
    xls = pd.ExcelFile(uploaded_file)
    features_df = xls.parse('Features List')
    sprints_df = xls.parse('Sprint Definitions')

    # === Prepare Dates ===
    sprints_df['Start Date'] = pd.to_datetime(sprints_df['Start Date'], errors='coerce')
    sprints_df['End Date'] = pd.to_datetime(sprints_df['End Date'], errors='coerce')

    # === Merge Sprint Dates into Features ===
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

    # === Sort by Target Release Version ===
    features_merged['Release Sort'] = features_merged['Target Release'].str.extract(r'(\d+\.\d+\.\d+)')
    features_merged['Release Sort'] = features_merged['Release Sort'].apply(lambda x: tuple(map(int, x.split('.'))) if pd.notnull(x) else (0, 0, 0))
    features_merged.sort_values('Release Sort', inplace=True)
    features_merged.reset_index(drop=True, inplace=True)

    # === Gantt Chart ===
    fig = px.timeline(
        features_merged,
        x_start='Start',
        x_end='Finish',
        y=features_merged.index,
        color='Target Release',
        hover_data=['Story Points', 'Target Start Sprint', 'Target End Sprint'],
        color_discrete_sequence=px.colors.qualitative.Set1
    )

    # === Weekly Ticks ===
    fig.update_xaxes(
        tickformat="%b %d, %Y",
        tickmode="array",
        tickvals=pd.date_range(
            start=features_merged['Start'].min(),
            end=features_merged['Finish'].max(),
            freq="W-MON"
        )
    )

    # === Sprint Lines ===
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

    # === Layout ===
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

    st.subheader("📅 Gantt Chart")
    st.plotly_chart(fig, use_container_width=True)

    # === Capacity Check ===
    capacity_check = features_df.groupby('Target Start Sprint')['Story Points'].sum().reset_index()
    capacity_check = capacity_check.merge(
        sprints_df[['Sprint Name', 'Sprint Capacity']],
        left_on='Target Start Sprint',
        right_on='Sprint Name',
        how='left'
    )
    capacity_check['Over Capacity'] = capacity_check['Story Points'] > capacity_check['Sprint Capacity']

    # === Sprint Utilization ===
    sprint_utilization = capacity_check.copy()
    sprint_utilization['Utilization (%)'] = (sprint_utilization['Story Points'] / sprint_utilization['Sprint Capacity']) * 100

    # === Completion Metrics ===
    today = pd.to_datetime('today')
    completed_features = features_merged[features_merged['Finish'] <= today].copy()
    completed_features['Completed'] = 'Yes'
    total_story_points_completed = completed_features['Story Points'].sum()
    total_story_points_planned = features_merged['Story Points'].sum()

    # === Metrics ===
    st.subheader("📊 Metrics")
    col1, col2, col3 = st.columns(3)
    col1.metric("Total Story Points Planned", f"{total_story_points_planned:.0f}")
    col2.metric("Completed", f"{total_story_points_completed:.0f}")
    col3.metric("Completion Rate", f"{(total_story_points_completed / total_story_points_planned) * 100:.2f}%")

    # === Overdue Features ===
    overdue_features = features_merged[(features_merged['Finish'] < today) & (features_merged['Story Points'] > 0)]
    if not overdue_features.empty:
        st.subheader("⚠️ Overdue Features")
        st.dataframe(overdue_features[['Feature ID', 'Target Release', 'Story Points', 'Finish']])
    else:
        st.success("✅ No overdue features.")

    # === Needed Hours Chart (1 SP = 5 hours) ===
    capacity_check['Needed Hours'] = capacity_check['Story Points'] * 5

    hours_fig = px.bar(
        capacity_check,
        x='Sprint Name',
        y='Needed Hours',
        color='Over Capacity',
        text='Needed Hours',
        title='Estimated Hours per Sprint (1 Story Point = 5 Hours)',
        color_discrete_map={True: 'red', False: 'green'}
    )
    hours_fig.update_layout(xaxis_title='Sprint', yaxis_title='Hours', height=500)

    st.subheader("📉 Sprint Effort Estimation")
    st.plotly_chart(hours_fig, use_container_width=True)

    # === Top 10 Features ===
    top_features = features_merged.nlargest(10, 'Story Points')[['Feature ID', 'Story Points', 'Target Release']]
    st.subheader("🏆 Top 10 Features by Story Points")
    st.dataframe(top_features)

    # === Excel Report Export (as download button) ===
    output = BytesIO()
    with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
        capacity_check.to_excel(writer, sheet_name="Capacity", index=False)
        sprint_utilization.to_excel(writer, sheet_name="Utilization", index=False)
        overdue_features.to_excel(writer, sheet_name="Overdue", index=False)
        top_features.to_excel(writer, sheet_name="Top Features", index=False)

    st.download_button(
        label="📥 Download Sprint Report as Excel",
        data=output,
        file_name="sprint_report.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )
