import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

# Set style for better aesthetics
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("colorblind")

# Create a figure with subplots, with more space between them
fig = plt.figure(figsize=(20, 16))
plt.subplots_adjust(hspace=0.4, wspace=0.3)

# Define a custom color palette for consistency
colors = sns.color_palette("colorblind", 10)
colors_extended = sns.color_palette("bright", 15)

# Create the four subplots with specific positions and sizes
ax1 = plt.subplot2grid((2, 2), (0, 0), fig=fig)  # Top votes bar chart
ax2 = plt.subplot2grid((2, 2), (0, 1), fig=fig)  # Vote share pie chart
ax3 = plt.subplot2grid((2, 2), (1, 0), fig=fig)  # Seats won chart
ax4 = plt.subplot2grid((2, 2), (1, 1), fig=fig)  # Top candidates chart

# Load dataset
df = pd.read_csv("pakistan_election.csv")

# Clean up party names (remove extra spaces)
df['candidate_party'] = df['candidate_party'].str.strip()

# Group by total votes per party
party_votes = df.groupby('candidate_party')['candidate_votes'].sum().sort_values(ascending=False)

# Select top N parties
top_n = 15
top_party_votes = party_votes.head(top_n)

# Precompute other variables for efficiency
top_n_pie = 6
top_parties = party_votes.head(top_n_pie)
others = party_votes[top_n_pie:].sum()
pie_data = pd.Series(top_parties.values.tolist() + [others], 
                     index=top_parties.index.tolist() + ['Others'])

# Get winners and party seats
winners = df[df['outcome'].str.lower() == 'win']
party_seats = winners['candidate_party'].value_counts()

# Top 10 candidates by votes
top_candidates = df.sort_values(by='candidate_votes', ascending=False).head(10)

# 1. Total Votes by Top 15 Political Parties (Horizontal Bar Chart)
bars = ax1.barh(top_party_votes.index[::-1], top_party_votes.values[::-1], color=colors_extended)
ax1.set_title('Total Votes by Top 15 Political Parties', fontsize=16, fontweight='bold')
ax1.set_xlabel('Total Votes (in millions)', fontsize=12)
ax1.set_yticklabels(top_party_votes.index[::-1], fontsize=11)
ax1.grid(axis='x', linestyle='--', alpha=0.6)

# Format x-axis to show millions
ax1.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, loc: f"{x/1000000:.1f}M"))

# Add labels to the bars
for bar in bars:
    width = bar.get_width()
    ax1.text(width + 200000, bar.get_y() + bar.get_height()/2, 
             f'{width/1000000:.2f}M', va='center', fontsize=10)

# 2. Vote Share by Party (Nationwide) - Pie Chart
wedges, texts, autotexts = ax2.pie(
    pie_data, 
    labels=None,
    autopct='%1.1f%%', 
    startangle=90, 
    colors=colors,
    wedgeprops={'edgecolor': 'white', 'linewidth': 1.5}
)

# Improve the pie chart legend
ax2.legend(
    wedges, 
    pie_data.index, 
    title="Political Parties",
    loc="center left", 
    bbox_to_anchor=(1, 0, 0.5, 1)
)

# Set properties for the percentage labels
for autotext in autotexts:
    autotext.set_color('white')
    autotext.set_fontsize(11)
    autotext.set_fontweight('bold')

ax2.set_title('Vote Share by Party (Top 6 + Others)', fontsize=16, fontweight='bold')

# 3. Total Seats Won by Party
# Get top 15 parties by seats for better visualization
top_seats = party_seats.head(15)
bars = ax3.bar(np.arange(len(top_seats)), top_seats.values, color=colors_extended)
ax3.set_title('Total Seats Won by Top 15 Parties', fontsize=16, fontweight='bold')
ax3.set_ylabel('Number of Seats', fontsize=12)
ax3.set_xticks(np.arange(len(top_seats)))
ax3.set_xticklabels(top_seats.index, rotation=45, ha='right', fontsize=11)
ax3.grid(axis='y', linestyle='--', alpha=0.6)

# Add value labels on top of each bar
for bar in bars:
    height = bar.get_height()
    ax3.text(bar.get_x() + bar.get_width()/2., height + 5,
             f'{height:.0f}', ha='center', va='bottom', fontsize=10)

# 4. Top 10 Candidates by Votes (Horizontal Bar Chart)
bars = ax4.barh(top_candidates['candidate_name'], top_candidates['candidate_votes'], color=colors[0:10])
ax4.set_xlabel('Votes', fontsize=12)
ax4.set_title('Top 10 Candidates by Votes', fontsize=16, fontweight='bold')
ax4.invert_yaxis()  # Highest on top
ax4.grid(axis='x', linestyle='--', alpha=0.6)

# Format the x-axis to show thousands
ax4.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, loc: f"{x/1000:.0f}K"))

# Add labels to the bars
for bar in bars:
    width = bar.get_width()
    ax4.text(width + 5000, bar.get_y() + bar.get_height()/2, 
             f'{width/1000:.1f}K', va='center', fontsize=10)

# Add a note about the party abbreviation for each candidate
for i, candidate in enumerate(top_candidates['candidate_name']):
    party = top_candidates.iloc[i]['candidate_party']
    ax4.text(5000, bars[i].get_y() + bars[i].get_height()/2, 
             f"({party})", va='center', ha='left', fontsize=9, color='black')

# Add an overall title for the entire figure
plt.suptitle('Pakistan Elections Analysis', fontsize=20, fontweight='bold', y=0.98)

# Adjust layout
plt.tight_layout(rect=[0, 0, 1, 0.96])

# Save the figure with high resolution
plt.savefig('election_analysis.png', dpi=300, bbox_inches='tight')

# Show the plot
plt.show()