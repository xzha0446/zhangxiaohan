import os
import json
import pandas as pd
import re
import matplotlib.pyplot as plt
from nltk.corpus import stopwords
from scipy.stats import chi2_contingency
import seaborn as sns
import nltk
import logging
from wordcloud import WordCloud
import numpy as np
from scipy.stats import chi2_contingency, ttest_ind, mannwhitneyu, shapiro


# Initialize NLTK
nltk.download('stopwords')

# Logging configuration
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Clean text function
def clean_text_advanced(text):
    if not isinstance(text, str):
        return ''
    custom_stopwords = set(stopwords.words('english') + ['rt', 'via', 'amp', 'news', 'charlie'])
    text = re.sub(r'http\S+', '', text)  
    text = re.sub(r'[^a-zA-Z ]', '', text)  
    text = text.lower()
    text = ' '.join(word for word in text.split() if word not in custom_stopwords)
    return text

# Load NRC Emotion Lexicon
def load_nrc_emotion_lexicon(filepath):
    emotion_lexicon = {}
    try:
        with open(filepath, 'r', encoding='utf-8') as file:
            for line in file:
                parts = line.strip().split('\t')
                if len(parts) == 3 and int(parts[2]) == 1:
                    words = re.split(r'--|,', parts[0])
                    emotion = parts[1]
                    for word in words:
                        word = word.strip()
                        if emotion not in emotion_lexicon:
                            emotion_lexicon[emotion] = []
                        emotion_lexicon[emotion].append(word)
    except Exception as e:
        logging.error(f"Error loading NRC Emotion Lexicon: {e}")
    return emotion_lexicon

# Extract source tweet data
def extract_source_tweet_data(dataset_path, limit=None):
    """Extract JSON data from nested source-tweet folders."""
    data = []
    for root, dirs, files in os.walk(dataset_path):
        if os.path.basename(root) == "source-tweet":  
            for file in files:
                if file.endswith('.json'):
                    file_path = os.path.join(root, file)
                    try:
                        with open(file_path, 'r', encoding='utf-8') as f:
                            content = json.load(f)
                            if 'text' in content:
                                data.append({'text': content['text']})  
                            else:
                                logging.warning(f"No 'text' field in {file_path}")
                    except Exception as e:
                        logging.error(f"Error reading {file_path}: {e}")
                    if limit and len(data) >= limit:
                        return pd.DataFrame(data)
    return pd.DataFrame(data)

# Count emotions
def count_emotions(text, emotion_lexicon):
    counts = {emotion: 0 for emotion in emotion_lexicon}
    words = text.split()
    for emotion, word_list in emotion_lexicon.items():
        counts[emotion] = sum(word in word_list for word in words)
    return counts

# Compute emotion diversity
def compute_emotion_diversity(emotion_counts):
    values = np.array(list(emotion_counts.values()))
    proportions = values / values.sum() if values.sum() > 0 else np.zeros_like(values)
    entropy = -np.sum(proportions * np.log2(proportions + 1e-9)) 
    return entropy

# Compute dominant emotion ratio
def dominant_emotion_ratio(emotion_counts):
    total = sum(emotion_counts.values())
    if total == 0:
        return 0
    return max(emotion_counts.values()) / total

# File paths
rumours_path = r'C:\Users\zxiao\OneDrive\桌面\pheme-rnr-dataset\charliehebdo\rumours'
non_rumours_path = r'C:\Users\zxiao\OneDrive\桌面\pheme-rnr-dataset\charliehebdo\non-rumours'
nrc_lexicon_path = r"C:\Users\zxiao\OneDrive\桌面\NRC_Emotion_Lexicon.txt"

# Load emotion lexicon
emotion_lexicon = load_nrc_emotion_lexicon(nrc_lexicon_path)

# Load datasets
fake_news_df = extract_source_tweet_data(rumours_path)
true_news_df = extract_source_tweet_data(non_rumours_path)

if fake_news_df.empty or true_news_df.empty:
    logging.error("No data extracted. Check the directory structure or file contents.")
else:
    print("\n=== Sample of Fake News Data ===")
    print(fake_news_df.head())

    print("\n=== Sample of True News Data ===")
    print(true_news_df.head())

    fake_news_df['cleaned_text'] = fake_news_df['text'].apply(clean_text_advanced)
    true_news_df['cleaned_text'] = true_news_df['text'].apply(clean_text_advanced)

    # Analyze emotions
    fake_news_df['emotions'] = fake_news_df['cleaned_text'].apply(lambda x: count_emotions(x, emotion_lexicon))
    true_news_df['emotions'] = true_news_df['cleaned_text'].apply(lambda x: count_emotions(x, emotion_lexicon))

    # 过滤情绪总和为 0 的文本（即完全没有情绪词的）
    fake_news_df = fake_news_df[fake_news_df['emotions'].apply(lambda x: sum(x.values()) > 0)]
    true_news_df = true_news_df[true_news_df['emotions'].apply(lambda x: sum(x.values()) > 0)]

    # Compute emotion diversity
    fake_news_df['emotion_diversity'] = fake_news_df['emotions'].apply(compute_emotion_diversity)
    true_news_df['emotion_diversity'] = true_news_df['emotions'].apply(compute_emotion_diversity)

    fake_news_df['dominant_emotion_ratio'] = fake_news_df['emotions'].apply(dominant_emotion_ratio)
    true_news_df['dominant_emotion_ratio'] = true_news_df['emotions'].apply(dominant_emotion_ratio)

    # Convert to DataFrame
    fake_emotions_df = pd.DataFrame(list(fake_news_df['emotions']))
    true_emotions_df = pd.DataFrame(list(true_news_df['emotions']))

    # Chi-Square Analysis
    emotion_categories = fake_emotions_df.columns
    contingency_table = pd.DataFrame({
        'Fake News': fake_emotions_df.sum(axis=0),
        'True News': true_emotions_df.sum(axis=0)
    }, index=emotion_categories)

    print("\n=== Emotion Ratio Comparisons ===")
    print("Fake News Emotion Ratios:\n", fake_emotions_df.sum() / fake_emotions_df.sum().sum())
    print("\nTrue News Emotion Ratios:\n", true_emotions_df.sum() / true_emotions_df.sum().sum())

    print("\n=== Contingency Table ===")
    print(contingency_table)


    # Perform Chi-Square test
    chi2_stat, p_value, dof, expected = chi2_contingency(contingency_table)

    # Display Chi-Squared Test results
    print("\n=== Chi-Squared Test Results ===")
    print(f"Chi-Squared Statistic: {chi2_stat:.4f}")
    print(f"Degrees of Freedom: {dof}")
    print(f"P-Value: {p_value:.4f}")

    # === Dominant Emotion Ratio Statistical Test ===
    print("\n>>> Running dominant emotion ratio t-test...")

    fake_dom = fake_news_df['dominant_emotion_ratio'].dropna()
    true_dom = true_news_df['dominant_emotion_ratio'].dropna()

    
    # === Shapiro-Wilk Test for Normality ===
    print("\n>>> Shapiro-Wilk Normality Test for Dominant Emotion Ratio")

    stat_fake, p_fake = shapiro(fake_dom)
    stat_true, p_true = shapiro(true_dom)

    print(f"Fake News: W = {stat_fake:.4f}, p = {p_fake:.4f}")
    print(f"True News: W = {stat_true:.4f}, p = {p_true:.4f}")

    # Interpretation
    if p_fake < 0.05:
        print("→ Fake News distribution significantly deviates from normality (p < 0.05)")
    else:
        print("→ Fake News distribution does not significantly deviate from normality")

    if p_true < 0.05:
        print("→ True News distribution significantly deviates from normality (p < 0.05)")
    else:
        print("→ True News distribution does not significantly deviate from normality")

    print(f"Fake N = {len(fake_dom)} True N = {len(true_dom)}")

    t_stat_dom, p_val_dom = ttest_ind(fake_dom, true_dom)

    print("=== Dominant Emotion Ratio Test ===")
    print(f"T-statistic: {t_stat_dom:.4f}")
    print(f"P-value: {p_val_dom:.4f}")
    if p_val_dom < 0.05:
        print("→ Fake news exhibits significantly stronger dominant emotional framing.")
    else:
        print("→ No significant difference in dominant emotional tone.")

# === Calculate Cohen's d for Dominant Emotion Ratio ===
def cohens_d(x, y):
    nx, ny = len(x), len(y)
    pooled_std = np.sqrt(((nx - 1) * np.var(x, ddof=1) + (ny - 1) * np.var(y, ddof=1)) / (nx + ny - 2))
    return (np.mean(x) - np.mean(y)) / pooled_std

cohen_d_value = cohens_d(fake_dom, true_dom)
print(f"\n=== Effect Size (Cohen's d) ===")
print(f"Cohen's d for Dominant Emotion Ratio: {cohen_d_value:.4f}")
if abs(cohen_d_value) >= 0.8:
    print("→ Large effect size")
elif abs(cohen_d_value) >= 0.5:
    print("→ Medium effect size")
elif abs(cohen_d_value) >= 0.2:
    print("→ Small effect size")
else:
    print("→ Negligible effect size")

# === Calculate Cramér's V for Chi-Square ===
def cramers_v(confusion_matrix):
    chi2, _, _, _ = chi2_contingency(confusion_matrix)
    n = confusion_matrix.to_numpy().sum()
    phi2 = chi2 / n
    r, k = confusion_matrix.shape
    phi2_corr = max(0, phi2 - ((k - 1)*(r - 1)) / (n - 1))
    r_corr = r - ((r - 1)**2) / (n - 1)
    k_corr = k - ((k - 1)**2) / (n - 1)
    return np.sqrt(phi2_corr / min((k_corr - 1), (r_corr - 1)))

cramers_v_value = cramers_v(contingency_table)
print(f"\n=== Association Strength (Cramér's V) ===")
print(f"Cramér's V for Emotion Distribution: {cramers_v_value:.4f}")
if cramers_v_value >= 0.3:
    print("→ Strong association")
elif cramers_v_value >= 0.1:
    print("→ Moderate association")
else:
    print("→ Weak or negligible association")

    # Interpret results
    alpha = 0.05
    if p_value < alpha:
        print("→ Result: Significant difference found between Fake News and True News emotions (p < 0.05).")
    else:
        print("→ Result: No significant difference found between Fake News and True News emotions (p >= 0.05).")

    # Emotion Diversity t-test
from scipy.stats import mannwhitneyu

# Mann–Whitney U Test for Emotion Diversity
print("\n>>> Running Mann–Whitney U test for emotion diversity...")

fake_div = fake_news_df['emotion_diversity'].dropna()
true_div = true_news_df['emotion_diversity'].dropna()

u_stat, p_val_mwu = mannwhitneyu(fake_div, true_div, alternative='two-sided')

print("=== Mann–Whitney U Test Results ===")
print(f"U statistic: {u_stat:.4f}")
print(f"P-value: {p_val_mwu:.4f}")

if p_val_mwu < 0.05:
    print("→ Statistically significant difference in emotion diversity (p < 0.05)")
else:
    print("→ No statistically significant difference in emotion diversity (p ≥ 0.05)")

# === Shapiro-Wilk Test for Normality (Entropy) ===
print("\n>>> Shapiro-Wilk Normality Test for Emotion Diversity")

stat_fake_entropy, p_fake_entropy = shapiro(fake_div)
stat_true_entropy, p_true_entropy = shapiro(true_div)

print(f"Fake News Entropy: W = {stat_fake_entropy:.4f}, p = {p_fake_entropy:.4f}")
print(f"True News Entropy: W = {stat_true_entropy:.4f}, p = {p_true_entropy:.4f}")

if p_fake_entropy < 0.05:
    print("→ Fake News entropy distribution significantly deviates from normality (p < 0.05)")
else:
    print("→ Fake News entropy distribution does not significantly deviate from normality")

if p_true_entropy < 0.05:
    print("→ True News entropy distribution significantly deviates from normality (p < 0.05)")
else:
    print("→ True News entropy distribution does not significantly deviate from normality")

from scipy.stats import mannwhitneyu

print("\n>>> Running Mann–Whitney U test for dominant emotion ratio...")

fake_dom = fake_news_df['dominant_emotion_ratio'].dropna()
true_dom = true_news_df['dominant_emotion_ratio'].dropna()

u_stat_dom, p_val_dom_u = mannwhitneyu(fake_dom, true_dom, alternative='two-sided')

print("=== Mann–Whitney U Test for Dominant Emotion Ratio ===")
print(f"U statistic: {u_stat_dom:.4f}")
print(f"P-value: {p_val_dom_u:.4f}")

if p_val_dom_u < 0.05:
    print("→ Statistically significant difference in dominant emotion ratio (p < 0.05)")
else:
    print("→ No significant difference in dominant emotion ratio (p ≥ 0.05)")

    # WordCloud
    all_fake_text = ' '.join(fake_news_df['cleaned_text'])
    all_true_text = ' '.join(true_news_df['cleaned_text'])
    fake_wc = WordCloud(background_color='white', colormap='Reds').generate(all_fake_text)
    true_wc = WordCloud(background_color='white', colormap='Greens').generate(all_true_text)

    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.imshow(fake_wc, interpolation='bilinear')
    plt.axis('off')
    plt.title('Fake News Word Cloud', fontsize=16)

    plt.subplot(1, 2, 2)
    plt.imshow(true_wc, interpolation='bilinear')
    plt.axis('off')
    plt.title('True News Word Cloud', fontsize=16)

    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(10, 6))
    data = pd.DataFrame({
        'Dominant Emotion Ratio': pd.concat([fake_news_df['dominant_emotion_ratio'], true_news_df['dominant_emotion_ratio']]),
        'Group': ['Fake News'] * len(fake_news_df) + ['True News'] * len(true_news_df)
    })
    sns.boxplot(x='Group', y='Dominant Emotion Ratio', data=data, palette='Set2')
    plt.title("Boxplot of Dominant Emotion Ratio by News Type", fontsize=14)
    plt.ylabel("Dominant Emotion Ratio")
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(10, 6))
    data = pd.DataFrame({
        'Emotion Diversity': pd.concat([fake_news_df['emotion_diversity'], true_news_df['emotion_diversity']]),
        'Group': ['Fake News'] * len(fake_news_df) + ['True News'] * len(true_news_df)
    })
    sns.violinplot(x='Group', y='Emotion Diversity', data=data, palette='Set1')
    plt.title("Violin Plot of Emotion Diversity by News Type", fontsize=14)
    plt.tight_layout()
    plt.show()

    # Print contingency table
    print("\n=== Contingency Table ===")
    print(contingency_table)

    emotion_results = {}

    for emotion in fake_emotions_df.columns:
        emotion_table = np.array([
            [fake_emotions_df[emotion].sum(), true_emotions_df[emotion].sum()],
            [fake_emotions_df.sum().sum() - fake_emotions_df[emotion].sum(),
            true_emotions_df.sum().sum() - true_emotions_df[emotion].sum()]
        ])

        if np.any(emotion_table == 0):
            continue

        chi2, p, _, _ = chi2_contingency(emotion_table, correction=False)
        
        emotion_results[emotion] = {'Chi-Square': chi2, 'p-value': p}

    chi_square_results_df = pd.DataFrame.from_dict(emotion_results, orient='index')

    print("\n=== Chi-Square Test Results for Each Emotion ===")
    print(chi_square_results_df)

    significant_results = chi_square_results_df[chi_square_results_df['p-value'] < 0.05]

    if significant_results.empty:
        print("No significant emotional differences found (p >= 0.05). No plot will be generated.")
    else:
        significant_results = significant_results.reset_index()
        significant_results.columns = ['Emotion', 'Chi-Square', 'p-value']

        plt.figure(figsize=(12, 6))
        sns.barplot(x='Emotion', y='Chi-Square', data=significant_results, palette='coolwarm')
        plt.axhline(y=3.84, color='r', linestyle='--', label='Significance Threshold (p=0.05)')
        plt.title("Significant Emotional Differences (p < 0.05)", fontsize=16)
        plt.xlabel("Emotion Categories", fontsize=12)
        plt.ylabel("Chi-Square Value", fontsize=12)
        plt.xticks(rotation=45)
        plt.legend()
        plt.show()

    # Visualize contingency table
    plt.figure(figsize=(12, 6))
    sns.heatmap(contingency_table, annot=True, fmt=".0f", cmap="coolwarm", cbar=True)
    plt.title("Emotion Distribution Contingency Table: Fake News vs. True News", fontsize=16)
    plt.tight_layout()
    plt.show()

    # Bar plot
    plt.figure(figsize=(12, 6))
    fake_emotions_avg = fake_emotions_df.mean()
    true_emotions_avg = true_emotions_df.mean()
    x = range(len(fake_emotions_avg))
    plt.bar(x, fake_emotions_avg, color='#FF6347', alpha=0.7, width=0.4, label='Fake News')
    plt.bar([i + 0.4 for i in x], true_emotions_avg, color='#32CD32', alpha=0.7, width=0.4, label='True News')
    plt.xticks([i + 0.2 for i in x], fake_emotions_avg.index, rotation=30, fontsize=12)
    plt.title("Emotion Comparison: Fake News vs. True News", fontsize=16)
    plt.ylabel("Average Word Count", fontsize=12)
    plt.xlabel("Emotion Categories", fontsize=12)
    plt.legend(fontsize=12)
    plt.tight_layout()
    plt.show()

    # Heatmap
    plt.figure(figsize=(12, 6))
    combined = pd.DataFrame({
        'Fake News': fake_emotions_avg,
        'True News': true_emotions_avg
    })
    sns.heatmap(combined.T, annot=True, fmt=".2f", cmap="coolwarm", cbar=True)
    plt.title("Emotion Distribution Heatmap", fontsize=16)
    plt.tight_layout()
    plt.show()

    # KDE plot for emotion diversity
    plt.figure(figsize=(12, 6))
    sns.kdeplot(fake_news_df['emotion_diversity'], label='Fake News', fill=True, color='red')
    sns.kdeplot(true_news_df['emotion_diversity'], label='True News', fill=True, color='green')
    plt.title("Emotion Diversity Distribution", fontsize=16)
    plt.xlabel("Emotion Diversity (Shannon Entropy)", fontsize=12)
    plt.ylabel("Density", fontsize=12)
    plt.legend(fontsize=12)
    plt.tight_layout()
    plt.show()