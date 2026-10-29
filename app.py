import streamlit as st
import re
import pandas as pd

from custom_modules import func_analysis as analysis

# ---------------------------
# Emoji extractor
# ---------------------------
def extract_emojis(text):
    emoji_pattern = re.compile("["
        u"\U0001F600-\U0001F64F"
        u"\U0001F300-\U0001F5FF"
        u"\U0001F680-\U0001F6FF"
        u"\U0001F1E0-\U0001F1FF"
        "]+", flags=re.UNICODE)
    return emoji_pattern.findall(text)

# ---------------------------
# UI
# ---------------------------
st.title("WhatsApp Chat Analyzer 📊")
st.sidebar.image("./assets/images/logo.jpg", width="stretch")
st.sidebar.title("Analyze")

file = st.sidebar.file_uploader("Upload chat file", type=["txt"])
run_sentiment = st.sidebar.checkbox("Run Sentiment Analysis")

# ---------------------------
# Data Loader
# ---------------------------
if file is not None:

    @st.cache_data
    def load_data():
        content = file.read().decode('utf-8', errors='ignore')
        lines = content.splitlines()

        rows = []

        for line in lines:
            try:
                # Remove system messages
                if any(x in line.lower() for x in [
                    "end-to-end encrypted",
                    "created this group",
                    "pinned a message",
                    "changed this group's icon",
                    "media omitted"
                ]):
                    continue

                match = re.match(
                    r"(\d{1,2}/\d{1,2}/\d{2,4}), (\d{1,2}:\d{2}) ?- (.*)",
                    line
                )

                if not match:
                    continue

                date, time_str, rest = match.groups()

                if ": " in rest:
                    author, message = rest.split(": ", 1)
                else:
                    continue

                rows.append([date, time_str, author.strip(), message.strip()])

            except:
                continue

        if len(rows) == 0:
            return None

        df = pd.DataFrame(rows, columns=["Date", "Time", "Author", "Message"])
        df["Emojis"] = df["Message"].apply(extract_emojis)

        return df

    data = load_data()

    if data is None:
        st.error("No valid messages found.")
        st.stop()

    # ---------------------------
    # Sentiment
    # ---------------------------
    data = analysis.apply_sentiment_analysis(data)

    if st.sidebar.checkbox("Show raw data", True):
        st.write(data)

    # ---------------------------
    # Member selection
    # ---------------------------
    members = list(data['Author'].unique())
    members.append("All")

    selected = st.sidebar.selectbox("Select Member", members)

    # ---------------------------
    # Sentiment charts
    # ---------------------------
    if run_sentiment:
        st.plotly_chart(analysis.sentiment_distribution(data))
        st.plotly_chart(analysis.sentiment_by_user(data))

    # ---------------------------
    # Analysis
    # ---------------------------
    try:
        if selected == "All":
            st.markdown(analysis.stats(data), unsafe_allow_html=True)

            st.write("Top Emojis:")
            emoji_counts = data["Emojis"].explode().value_counts()
            for e, c in emoji_counts.head(10).items():
                st.write(f"{e} : {c}")

            st.plotly_chart(analysis.visualize_emoji(data))
            st.pyplot(analysis.word_cloud(data))
            st.pyplot(analysis.active_date(data))
            st.pyplot(analysis.active_time(data))

        else:
            user_df = data[data['Author'] == selected]

            st.markdown(analysis.stats(user_df))

            emoji_counts = user_df["Emojis"].explode().value_counts()
            for e, c in emoji_counts.head(10).items():
                st.write(f"{e} : {c}")

            st.plotly_chart(analysis.visualize_emoji(user_df))
            st.pyplot(analysis.word_cloud(user_df))
            st.pyplot(analysis.active_date(user_df))
            st.pyplot(analysis.active_time(user_df))

    except Exception as e:
        st.error(f"Error: {e}")