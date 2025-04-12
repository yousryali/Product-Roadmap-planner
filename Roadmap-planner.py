import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from io import BytesIO
import os

st.set_page_config(layout="wide")
st.title("📊 Feature Timeline & Sprint Capacity Tracker")

# === File Upload ===
uploaded_file = st.file_uploader("Upload Excel File", type=["xlsx"])

if uploaded_file:
    xls = pd.ExcelFile(uploaded_file)
    features_df = xls.parse('Features List')
    sprints_df = xls.parse('Sprint Definitions')

    # === Prepare Dates ===
    sprints_df['Start Date'] = pd.to_datetime(sprints_df['Start Date'], errors='coerce')
    sprints_df['End Date'] = pd.to_datetime(sprints_df['End Date'], errors='coerce')

    # === Merge Dates into Features ===
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

    # === Sort by Release ===
    features_merged['Release Sort'] = features_merged['Target Release'].str.extract(r'(\d+\.\d+\.\d+)')
    features_merged['Release Sort'] = features_merged['Release Sort'].apply(lambda x: tuple(map(int, x.split('.'))))
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

    fig.update_xaxes(
        tickformat="%b %d, %Y",
        tickmode="array",
        tickvals=pd.date_range(
            start=features_merged['Start'].min(),
            end=features_merged['Finish'].max(),
            freq="W-MON"
        )
    )

    # === Add Sprint Lines ===
    sprint_line_y = len(features_merged) + 2
    sprints_df['Sprint Center'] = sprints_df['Start Date'] + (sprints_df['End Date'] - sprints_df['Start Date']) / 2

    for _, row in sprints_df.iterrows():
        x0, x1 = row['Start Date'], row['End Date']
        x_center = row['Sprint Center']

        fig.add_trace(go.Scatter(x=[x0, x1], y=[sprint_line_y]*2, mode='lines',
                                 line=dict(color='black', width=2), showlegend=False))
        fig.add_trace(go.Scatter(x=[x0, x1], y=[sprint_line_y]*2, mode='markers',
                                 marker=dict(symbol="triangle-down", size=10, color='black'), showlegend=False))
        fig.add_trace(go.Scatter(x=[x_center], y=[sprint_line_y + 0.4], mode='text',
                                 text=[row['Sprint Name']], textposition='top center', showlegend=False))

    fig.update_layout(
        title='Feature Timeline by Sprint (Sorted by Release)',
        xaxis_title='Date',
        yaxis_title='Feature ID',
        yaxis=dict(tickvals=list(range(len(features_merged))),
                   ticktext=features_merged['Feature ID'].tolist()),
        height=900,
        showlegend=True,
        plot_bgcolor='white',
        paper_bgcolor='white',
        font=dict(color='black')
    )

    st.plotly_chart(fig, use_container_width=True)

    # === Capacity Check ===
    capacity_check = features_df.groupby('Target Start Sprint')['Story Points'].sum().reset_index()
    capacity_check = capacity_check.merge(
        sprints_df[['Sprint Name', 'Sprint Capacity']],
        left_on='Target Start Sprint', right_on='Sprint Name', how='left')
    capacity_check['Over Capacity'] = capacity_check['Story Points'] > capacity_check['Sprint Capacity']

    # === Sprint Utilization ===
    sprint_utilization = capacity_check.copy()
    sprint_utilization['Utilization (%)'] = (sprint_utilization['Story Points'] / sprint_utilization['Sprint Capacity']) * 100

    # === Metrics ===
    st.subheader("📊 Sprint Capacity and Utilization")
    st.dataframe(sprint_utilization)

    avg_story_points = features_merged.groupby('Target Start Sprint')['Story Points'].sum().mean()
    st.metric("📈 Average Story Points per Sprint", f"{avg_story_points:.2f}")

    # === Overdue Features ===
    today = pd.to_datetime('today')
    overdue_features = features_merged[(features_merged['Finish'] < today) & (features_merged['Story Points'] > 0)]
    if not overdue_features.empty:
        st.subheader("⚠️ Overdue Features")
        st.dataframe(overdue_features[['Feature ID', 'Target Release', 'Story Points', 'Finish']])

    # === Completion Metrics ===
    completed_features = features_merged[features_merged['Finish'] <= today].copy()
    completed_features['Completed'] = 'Yes'
    total_completed = completed_features['Story Points'].sum()
    total_planned = features_merged['Story Points'].sum()

    st.subheader("📅 Completion Metrics")
    st.metric("Total Story Points Planned", f"{total_planned:.0f}")
    st.metric("✅ Completed Story Points", f"{total_completed:.0f}")
    st.metric("📉 Completion Rate", f"{(total_completed / total_planned * 100):.2f}%")

    # === Top 10 Features ===
    st.subheader("🏆 Top 10 Features by Story Points")
    top_features = features_merged.nlargest(10, 'Story Points')[['Feature ID', 'Story Points', 'Target Release']]
    st.dataframe(top_features)

    # === Hours Bar Chart ===
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
    st.plotly_chart(hours_fig, use_container_width=True)

    # === Export Options ===
    st.subheader("📤 Export Reports")
    output = BytesIO()
    with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
        capacity_check.to_excel(writer, sheet_name="Capacity", index=False)
        sprint_utilization.to_excel(writer, sheet_name="Utilization", index=False)
        overdue_features.to_excel(writer, sheet_name="Overdue", index=False)
        top_features.to_excel(writer, sheet_name="Top Features", index=False)
    output.seek(0)
    st.download_button("Download Excel Report", output, file_name="sprint_report.xlsx")
