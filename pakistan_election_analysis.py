import pandas as pd
import matplotlib.pyplot as plt

# Load dataset
df = pd.read_csv("pakistan_election.csv")

# Clean up party names (remove extra spaces)
df['candidate_party'] = df['candidate_party'].str.strip()

# Group and sort by total votes per party
party_votes = df.groupby('candidate_party')['candidate_votes'].sum().sort_values(ascending=False)

# Select top N parties
top_n = 15
top_party_votes = party_votes.head(top_n)

# Plot horizontal bar chart
plt.figure(figsize=(10, 6))
plt.barh(top_party_votes.index[::-1], top_party_votes.values[::-1], color='skyblue')
plt.title('Total Votes by Top 15 Political Parties (Pakistan Elections)')
plt.xlabel('Total Votes')
plt.grid(axis='x', linestyle='--', alpha=0.6)
plt.tight_layout()
plt.savefig('top_15_party_votes.png')
plt.show()

# 2. Vote Share by Party (Nationwide) - Pie Chart
top_n = 6
top_parties = party_votes.head(top_n)
others = party_votes[top_n:].sum()
top_parties['Others'] = others

plt.figure(figsize=(8, 8))
top_parties.plot(kind='pie', autopct='%1.1f%%', startangle=140)
plt.title('Vote Share by Party (Top 6 + Others)')
plt.ylabel('')
plt.tight_layout()
plt.savefig('vote_share_pie_chart.png')
plt.show()


# 3. Total Seats Won by Party
winners = df[df['outcome'].str.lower() == 'win']
party_seats = winners['candidate_party'].value_counts()

plt.figure(figsize=(12, 6))
party_seats.plot(kind='bar', color='green')
plt.title('Total Seats Won by Party')
plt.xlabel('Political Party')
plt.ylabel('Number of Seats')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.savefig('seats_won_by_party.png')
plt.show()


# 4. Top 10 Candidates by Votes (Barh)
top_candidates = df.sort_values(by='candidate_votes', ascending=False).head(10)

plt.figure(figsize=(10, 6))
plt.barh(top_candidates['candidate_name'], top_candidates['candidate_votes'], color='orange')
plt.xlabel('Votes')
plt.title('Top 10 Candidates by Votes')
plt.gca().invert_yaxis()  # Highest on top
plt.tight_layout()
plt.grid(axis='x', linestyle='--', alpha=0.7)
plt.savefig('top_10_candidates.png')
plt.show()
