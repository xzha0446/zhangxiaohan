# zhangxiaohan
research project
This project analyzes the emotional characteristics of fake news and true news tweets using the NRC Emotion Lexicon. It combines natural language processing, lexicon-based emotion analysis, statistical testing, and rich visualizations to explore how emotional content differs between misinformation and factual reporting.

1. Text Preprocessing
The raw tweet texts are cleaned by:
- Removing URLs and non-alphabetic characters.
- Converting all text to lowercase.
- Filtering out standard stopwords and a few custom terms (e.g., "rt", "via", "news").
- This step ensures that only meaningful words remain for emotion analysis.

2. Emotion Mapping
Each word in the cleaned text is matched against the NRC Emotion Lexicon, which categorizes words into basic emotions such as anger, fear, trust, joy, and more. The code calculates how often words from each emotional category appear in each tweet.

3. Emotion Metrics
Two core metrics are computed:
- Emotion Diversity: Using Shannon entropy, this measures how varied the emotional expressions are within a single tweet.

- Dominant Emotion Ratio: The proportion of the most frequent emotion relative to all emotions in the tweet, capturing how strongly one emotion dominates.

4. Statistical Analysis
To evaluate differences between fake and true news tweets:
- A Chi-Square test is used to compare overall emotion distributions between the two groups.
- Cramér’s V is calculated to assess the strength of association.
- Shapiro-Wilk tests check for normality in emotion metric distributions.
- T-tests and Mann–Whitney U tests are used to compare dominant emotion ratios and emotion diversity between groups.
- Cohen’s d provides effect size estimates.

5. Visualizations
A range of plots illustrates the emotional patterns:
- Heatmaps show emotion frequency differences between fake and true news.
- Bar charts compare average emotion word counts.
- KDE plots visualize the distribution of emotion diversity.
- Boxplots and violin plots display the spread and concentration of emotion metrics.
- Word clouds highlight frequently used words in each category.
