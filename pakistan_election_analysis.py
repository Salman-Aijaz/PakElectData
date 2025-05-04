# ──────────────────────────────────────────────────────────────────────────────
# IMPORTS
# ──────────────────────────────────────────────────────────────────────────────
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import logging

# ──────────────────────────────────────────────────────────────────────────────
# LOGGER CONFIGURATION
# ──────────────────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.FileHandler("election_analysis.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────────────────────────────────────
# GLOBAL STYLES & CONSTANTS
# ──────────────────────────────────────────────────────────────────────────────
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("colorblind")

# Font and style constants
TITLE_FONT_SIZE = 16
LABEL_FONT_SIZE = 12
TICK_FONT_SIZE = 11
ANNOTATION_FONT_SIZE = 10
LEGEND_FONT_SIZE = 11
FIGURE_TITLE_SIZE = 20
FONT_WEIGHT = 'bold'

# General constants
TOP_N_PARTIES = 15
TOP_N_PIE = 6
TOP_N_CANDIDATES = 10

# ──────────────────────────────────────────────────────────────────────────────
# DATA LOADING & CLEANING
# ──────────────────────────────────────────────────────────────────────────────
def load_and_clean_data(filepath):
    try:
        df = pd.read_csv(filepath)
        df['candidate_party'] = df['candidate_party'].str.strip()

        required_columns = ['candidate_party', 'candidate_votes', 'candidate_name', 'outcome']
        initial_count = len(df)
        df.dropna(subset=required_columns, inplace=True)
        dropped = initial_count - len(df)
        if dropped > 0:
            logger.warning(f"Dropped {dropped} rows due to missing critical values.")
        return df

    except FileNotFoundError:
        logger.error(f"File '{filepath}' not found.")
        exit(1)
    except pd.errors.EmptyDataError:
        logger.error("File is empty.")
        exit(1)
    except pd.errors.ParserError:
        logger.error("Failed to parse CSV file. Please check its format.")
        exit(1)
    except Exception as e:
        logger.exception(f"An unexpected error occurred: {e}")
        exit(1)

# ──────────────────────────────────────────────────────────────────────────────
# DATA PROCESSING FUNCTIONS
# ──────────────────────────────────────────────────────────────────────────────
def get_party_votes(df):
    return df.groupby('candidate_party')['candidate_votes'].sum().sort_values(ascending=False)

def get_pie_data(party_votes):
    top_parties = party_votes.head(TOP_N_PIE)
    others = party_votes[TOP_N_PIE:].sum()
    return pd.Series(top_parties.tolist() + [others], 
                     index=top_parties.index.tolist() + ['Others'])

def get_party_seats(df):
    winners = df[df['outcome'].str.lower() == 'win']
    return winners['candidate_party'].value_counts()

def get_top_candidates(df):
    return df.sort_values(by='candidate_votes', ascending=False).head(TOP_N_CANDIDATES)

# ──────────────────────────────────────────────────────────────────────────────
# PLOTTING FUNCTIONS
# ──────────────────────────────────────────────────────────────────────────────
def plot_total_votes(ax, top_party_votes, colors):
    bars = ax.barh(top_party_votes.index[::-1], top_party_votes.values[::-1], color=colors)
    ax.set_title('Total Votes by Top 15 Political Parties', fontsize=TITLE_FONT_SIZE, fontweight=FONT_WEIGHT)
    ax.set_xlabel('Total Votes (in millions)', fontsize=LABEL_FONT_SIZE)
    ax.set_yticks(np.arange(len(top_party_votes)))
    ax.set_yticklabels(top_party_votes.index[::-1], fontsize=TICK_FONT_SIZE)
    ax.grid(axis='x', linestyle='--', alpha=0.6)
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x/1_000_000:.1f}M"))

    offset = top_party_votes.max() * 0.01
    for bar in bars:
        width = bar.get_width()
        ax.text(width + offset, bar.get_y() + bar.get_height()/2, 
                f'{width/1_000_000:.2f}M', va='center', fontsize=ANNOTATION_FONT_SIZE)

def plot_vote_share_pie(ax, pie_data, colors):
    wedges, _, autotexts = ax.pie(
        pie_data,
        labels=None,
        autopct='%1.1f%%',
        startangle=90,
        colors=colors,
        wedgeprops={'edgecolor': 'white', 'linewidth': 1.5}
    )
    ax.legend(wedges, pie_data.index, title="Political Parties", loc="center left", 
              bbox_to_anchor=(1, 0, 0.5, 1), fontsize=LEGEND_FONT_SIZE)
    for autotext in autotexts:
        autotext.set_color('white')
        autotext.set_fontsize(ANNOTATION_FONT_SIZE)
        autotext.set_fontweight(FONT_WEIGHT)
    ax.set_title('Vote Share by Party (Top 6 + Others)', fontsize=TITLE_FONT_SIZE, fontweight=FONT_WEIGHT)

def plot_seats_won(ax, party_seats, colors):
    top_seats = party_seats.head(TOP_N_PARTIES)
    bars = ax.bar(np.arange(len(top_seats)), top_seats.values, color=colors)
    ax.set_title('Total Seats Won by Top 15 Parties', fontsize=TITLE_FONT_SIZE, fontweight=FONT_WEIGHT)
    ax.set_ylabel('Number of Seats', fontsize=LABEL_FONT_SIZE)
    ax.set_xticks(np.arange(len(top_seats)))
    ax.set_xticklabels(top_seats.index, rotation=45, ha='right', fontsize=TICK_FONT_SIZE)
    ax.grid(axis='y', linestyle='--', alpha=0.6)

    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 5,
                f'{height:.0f}', ha='center', va='bottom', fontsize=ANNOTATION_FONT_SIZE)

def plot_top_candidates(ax, top_candidates, colors):
    bars = ax.barh(top_candidates['candidate_name'], top_candidates['candidate_votes'], color=colors)
    ax.set_xlabel('Votes', fontsize=LABEL_FONT_SIZE)
    ax.set_title('Top 10 Candidates by Votes', fontsize=TITLE_FONT_SIZE, fontweight=FONT_WEIGHT)
    ax.invert_yaxis()
    ax.grid(axis='x', linestyle='--', alpha=0.6)
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x/1_000:.0f}K"))

    offset = top_candidates['candidate_votes'].max() * 0.01
    label_offset = top_candidates['candidate_votes'].min() * 0.005

    for bar in bars:
        width = bar.get_width()
        ax.text(width + offset, bar.get_y() + bar.get_height()/2, 
                f'{width/1_000:.1f}K', va='center', fontsize=ANNOTATION_FONT_SIZE)

    for i, candidate in enumerate(top_candidates['candidate_name']):
        party = top_candidates.iloc[i]['candidate_party']
        ax.text(label_offset, bars[i].get_y() + bars[i].get_height()/2, 
                f"({party})", va='center', ha='left', fontsize=ANNOTATION_FONT_SIZE, color='black')

# ──────────────────────────────────────────────────────────────────────────────
# MAIN FUNCTION
# ──────────────────────────────────────────────────────────────────────────────
def main():
    df = load_and_clean_data("pakistan_election.csv")

    party_votes = get_party_votes(df)
    pie_data = get_pie_data(party_votes)
    party_seats = get_party_seats(df)
    top_candidates = get_top_candidates(df)

    colors = sns.color_palette("colorblind", 10)
    colors_extended = sns.color_palette("bright", 15)

    fig = plt.figure(figsize=(20, 16))
    plt.subplots_adjust(hspace=0.4, wspace=0.3)

    ax1 = plt.subplot2grid((2, 2), (0, 0), fig=fig)
    ax2 = plt.subplot2grid((2, 2), (0, 1), fig=fig)
    ax3 = plt.subplot2grid((2, 2), (1, 0), fig=fig)
    ax4 = plt.subplot2grid((2, 2), (1, 1), fig=fig)

    plot_total_votes(ax1, party_votes.head(TOP_N_PARTIES), colors_extended)
    plot_vote_share_pie(ax2, pie_data, colors)
    plot_seats_won(ax3, party_seats, colors_extended)
    plot_top_candidates(ax4, top_candidates, colors)

    plt.suptitle('Pakistan Elections Analysis', fontsize=FIGURE_TITLE_SIZE, fontweight=FONT_WEIGHT, y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig('election_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()

# ──────────────────────────────────────────────────────────────────────────────
# ENTRY POINT
# ──────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    main()
