import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from tabulate import tabulate
import os

# === Load Excel File ===
file_path = "/content/sample-data.xlsx"
xls = pd.ExcelFile(file_path)
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

# === Update X-Axis with Weekly Ticks ===
fig.update_xaxes(
    tickformat="%b %d, %Y",
    tickmode="array",
    tickvals=pd.date_range(
        start=features_merged['Start'].min(),
        end=features_merged['Finish'].max(),
        freq="W-MON"
    )
)

# === Add Sprint Lines in One Row ===
sprint_line_y = len(features_merged) + 2
sprints_df['Sprint Center'] = sprints_df['Start Date'] + (sprints_df['End Date'] - sprints_df['Start Date']) / 2

for _, row in sprints_df.iterrows():
    x0, x1 = row['Start Date'], row['End Date']
    x_center = row['Sprint Center']
    y_pos = sprint_line_y

    # Line
    fig.add_trace(go.Scatter(
        x=[x0, x1], y=[y_pos, y_pos],
        mode='lines',
        line=dict(color='black', width=2),
        showlegend=False
    ))

    # Markers
    fig.add_trace(go.Scatter(
        x=[x0, x1], y=[y_pos, y_pos],
        mode='markers',
        marker=dict(symbol="triangle-down", size=10, color='black'),
        showlegend=False
    ))

    # Label
    fig.add_trace(go.Scatter(
        x=[x_center], y=[y_pos + 0.4],
        mode='text',
        text=[row['Sprint Name']],
        textposition='top center',
        showlegend=False
    ))

# === Update Layout ===
fig.update_layout(
    title='Feature Timeline by Sprint (Sorted by Release)',
    xaxis_title='Date',
    yaxis_title='Feature ID',
    yaxis=dict(
        tickvals=list(range(len(features_merged))),
        ticktext=features_merged['Feature ID'].tolist()
    ),
    height=900,
    showlegend=True
)

fig.show()

# === Capacity Check ===
capacity_check = features_df.groupby('Target Start Sprint')['Story Points'].sum().reset_index()
capacity_check = capacity_check.merge(
    sprints_df[['Sprint Name', 'Sprint Capacity']],
    left_on='Target Start Sprint',
    right_on='Sprint Name',
    how='left'
)
capacity_check['Over Capacity'] = capacity_check['Story Points'] > capacity_check['Sprint Capacity']

overloaded = capacity_check[capacity_check['Over Capacity']]
if not overloaded.empty:
    print("⚠️ Overloaded Sprints:")
    print(tabulate(overloaded[['Target Start Sprint', 'Story Points', 'Sprint Name', 'Sprint Capacity', 'Over Capacity']], headers='keys', tablefmt='fancy_grid'))
else:
    print("✅ All sprints are within capacity.")

# === Sprint Utilization ===
sprint_utilization = capacity_check.copy()
sprint_utilization['Utilization (%)'] = (sprint_utilization['Story Points'] / sprint_utilization['Sprint Capacity']) * 100
print("\n🏃 Sprint Utilization (Percentage of Sprint Capacity Used):")
print(tabulate(sprint_utilization[['Sprint Name', 'Story Points', 'Sprint Capacity', 'Utilization (%)']], headers='keys', tablefmt='fancy_grid'))

# === Average Story Points Per Sprint ===
avg_story_points_per_sprint = features_merged.groupby('Target Start Sprint')['Story Points'].sum().mean()
print(f"\n📊 Average Story Points per Sprint: {avg_story_points_per_sprint:.2f}")

# === Overdue Features ===
today = pd.to_datetime('today')
overdue_features = features_merged[(features_merged['Finish'] < today) & (features_merged['Story Points'] > 0)]
if not overdue_features.empty:
    print("\n⚠️ Overdue Features:")
    print(tabulate(overdue_features[['Feature ID', 'Target Release', 'Story Points', 'Finish']], headers='keys', tablefmt='fancy_grid'))
else:
    print("✅ No overdue features.")

# === Completion Metrics ===
completed_features = features_merged[features_merged['Finish'] <= today].copy()
completed_features['Completed'] = 'Yes'

total_story_points_completed = completed_features['Story Points'].sum()
total_story_points_planned = features_merged['Story Points'].sum()

print(f"\n📅 Total Story Points Planned: {total_story_points_planned}")
print(f"✅ Total Story Points Completed: {total_story_points_completed}")
print(f"📉 Completion Rate: {total_story_points_completed / total_story_points_planned * 100:.2f}%")

# === Top 10 Features ===
top_features = features_merged.nlargest(10, 'Story Points')[['Feature ID', 'Story Points', 'Target Release']]
print("\n🏆 Top 10 Features by Story Points:")
print(tabulate(top_features, headers='keys', tablefmt='fancy_grid'))

# === Needed Hours per Sprint (1 Story Point = 5 hours) ===
capacity_check['Needed Hours'] = capacity_check['Story Points'] * 5

# === Plot Needed Hours Chart ===
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
hours_fig.show()

# === Export Chart as Image (after creating directory safely) ===
os.makedirs("outputs", exist_ok=True)
hours_fig.write_image("outputs/hours_chart.png")

# === Export Report as Excel ===
with pd.ExcelWriter("outputs/sprint_report.xlsx") as writer:
    capacity_check.to_excel(writer, sheet_name="Capacity", index=False)
    sprint_utilization.to_excel(writer, sheet_name="Utilization", index=False)
    overdue_features.to_excel(writer, sheet_name="Overdue", index=False)
    top_features.to_excel(writer, sheet_name="Top Features", index=False)

print("📤 Reports exported to 'outputs/' folder.")