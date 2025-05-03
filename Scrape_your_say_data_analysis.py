import pandas as pd
import numpy as np
from scipy.stats import trim_mean, skew, kurtosis, chi2_contingency, ks_2samp, chisquare
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
import plotly.express as px
import seaborn as sns
import plotly.graph_objects as go
import geopandas as gpd
import matplotlib.colors as mcolors
import plotly.colors as pc
import yaml
import pandas as pd
import numpy as np
from dateutil import parser

reference_str = '09 January 2025'
ref_date = parser.parse(reference_str)

population_data_updated = {
    "Austria":         9.0,
    "Belgium":         11.7,
    "Bulgaria":        6.9,
    "Croatia":         4.1,
    "Cyprus":          1.2,
    "Czech Republic":  10.7,
    "Czechia":         10.7,
    "Denmark":         5.9,
    "Estonia":         1.3,
    "Finland":         5.6,
    "France":          67.8,
    "Germany":         83.2,
    "Greece":          10.7,
    "Hungary":         9.6,
    "Ireland":         5.0,
    "Italy":           59.0,
    "Latvia":          1.9,
    "Lithuania":       2.8,
    "Luxembourg":      0.64,
    "Netherlands":     17.7,
    "Poland":          37.7,
    "Portugal":        10.3,
    "Romania":         19.1,
    "Slovakia":        5.4,
    "Slovenia":        2.1,
    "Spain":           47.4,
    "Sweden":          10.5,
    "United Kingdom":  67.3,
    "Norway":          5.4,
    "Switzerland":     8.7,
    "Ukraine":         41.0
}

def compute_topic_usage_for_feedback(yaml_data, feedback_key, dict_key='by_country_respondent'):
    """
    Counts how many times each topic is introduced ACROSS consultations
    *where* the given feedback_key actually has some non-zero data.

    - feedback_key: 'feedback_one' or 'feedback_two'
    - dict_key: typically 'by_country_respondent' (or 'by_category_respondent' if needed)
    Returns a dictionary {topic_name: usage_count}.
    """
    usage_counts = {}
    for consultation_name, details in yaml_data.items():
        sub_dict = details.get(feedback_key, {}).get(dict_key, {})
        total_responses = sum(sub_dict.values())
        if total_responses > 0:
            topic_field = details.get("main_page", {}).get("topic", "Unknown")
            if topic_field and topic_field != "Unknown":
                topics_list = [t.strip() for t in topic_field.split(",")]
                for t in topics_list:
                    usage_counts[t] = usage_counts.get(t, 0) + 1
    return usage_counts

def count_responses_by_custom_stage(yaml_path):
    """
    Extracts and counts responses based on custom stage and status rules,
    assuming each consultation contains a single (stage, status) tuple.

    Returns:
    - A DataFrame with counts assigned to CALL FOR EVIDENCE, PUBLIC CONSULTATION, and COMMISSION ADOPTION.
    """
    with open(yaml_path, 'r') as f:
        yaml_data = yaml.load(f, Loader=yaml.FullLoader)
    stage_counts = {
        "CALL FOR EVIDENCE": [],
        "PUBLIC CONSULTATION": [],
        "COMMISSION ADOPTION": []
    }
    for _, details in yaml_data.items():
        stage_status_list = details.get("main_page", {}).get("stage_and_feedback_status", [])
        if not stage_status_list:
            continue
        sub_page = details.get("sub_page", {})
        open_val = sub_page.get("amount_feedback_one", np.nan)
        feedback_two = details.get("feedback_two", {})
        q_val = feedback_two.get("amount_feedback_two", np.nan)
        stage_tuple = stage_status_list[0]
        if not isinstance(stage_tuple, tuple) or len(stage_tuple) < 2:
            continue
        stage = stage_tuple[0].strip().upper()
        status = stage_tuple[1].strip().upper()
        if stage == "DRAFT ACT":
            continue
        elif stage in ["CALL FOR EVIDENCE", "PUBLIC CONSULTATION"]:
            stage_counts["CALL FOR EVIDENCE"].append(open_val)
            stage_counts["PUBLIC CONSULTATION"].append(q_val)
        elif stage == "COMMISSION ADOPTION":
            if status == "UPCOMING":
                stage_counts["CALL FOR EVIDENCE"].append(open_val)
                stage_counts["PUBLIC CONSULTATION"].append(q_val)
            elif status in ["OPEN", "CLOSED"]:
                stage_counts["COMMISSION ADOPTION"].append(open_val)
                stage_counts["PUBLIC CONSULTATION"].append(q_val)
    df = pd.DataFrame.from_dict(stage_counts, orient="index").T
    return df

def build_country_topic_respondent_triplets(yaml_data, feedback_key='feedback_one'):
    rows = []
    for _, details in yaml_data.items():
        topics_str = details.get("main_page", {}).get("topic", "")
        if not topics_str:
            continue
        topics = [t.strip() for t in topics_str.split(",") if t.strip()]
        if not topics:
            continue
        by_country = details.get(feedback_key, {}).get('by_country_respondent', {})
        by_respondent = details.get(feedback_key, {}).get('by_category_respondent', {})
        total_country = sum(by_country.values())
        total_respondent = sum(by_respondent.values())
        n_topics = len(topics)
        if total_country == 0 or total_respondent == 0:
            continue
        for country, c_val in by_country.items():
            for respondent, r_val in by_respondent.items():
                for topic in topics:
                    weight = (c_val * r_val) / n_topics
                    rows.append({
                        'country': country,
                        'topic': topic,
                        'respondent': respondent,
                        'weight': weight
                    })
    df = pd.DataFrame(rows)
    return df

def distribute_by_topic(yaml_data, feedback_key, dict_key):
    """
    For a given feedback_key (e.g. 'feedback_one' or 'feedback_two') and dict_key
    (e.g. 'by_country_respondent' or 'by_category_respondent'),
    this function builds a table distributing counts among comma-separated topics.

    Returns a DataFrame with index = row keys from the sub-dict (countries or categories)
    and columns = topics. The counts are distributed evenly if multiple topics exist.
    """
    topic_dict = {}
    for consultation_name, details in yaml_data.items():
        feedback_data = details.get(feedback_key, {})
        sub_dict = feedback_data.get(dict_key, {})
        topic_field = details.get("main_page", {}).get("topic", "Unknown")
        if topic_field and topic_field != "Unknown":
            raw_topics = [t.strip() for t in topic_field.split(",")]
        else:
            raw_topics = ["Unknown"]
        n_topics = len(raw_topics)
        for row_key, val in sub_dict.items():
            for tp in raw_topics:
                if tp not in topic_dict:
                    topic_dict[tp] = {}
                if row_key not in topic_dict[tp]:
                    topic_dict[tp][row_key] = 0
                topic_dict[tp][row_key] += val / n_topics
    df = pd.DataFrame.from_dict(topic_dict, orient='index').fillna(0)
    df = df.T
    return df

def compute_topic_usage(yaml_data):
    """
    Count how many times each topic is introduced across all consultations.
    Returns a dictionary {topic_name: usage_count}.
    """
    topic_counts = {}
    for consultation_name, details in yaml_data.items():
        topic_field = details.get("main_page", {}).get("topic", "Unknown")
        if topic_field and topic_field != "Unknown":
            topics_list = [t.strip() for t in topic_field.split(",")]
            for t in topics_list:
                topic_counts[t] = topic_counts.get(t, 0) + 1
    return topic_counts

def consultations_per_topic(yaml_path):
    """
    Reads YAML from a file path and returns a DataFrame with columns:
    [topic, consultation_count_normalised]

    Each topic's count is divided by the number of topics in its respective consultation.
    """
    with open(yaml_path, 'r') as f:
        yaml_data = yaml.load(f, Loader=yaml.FullLoader)
    topic_counts = {}
    for details in yaml_data.values():
        topic_field = details.get("main_page", {}).get("topic", "")
        if topic_field:
            topics_list = [t.strip() for t in topic_field.split(",") if t.strip()]
            num_topics = len(topics_list)
            if num_topics > 0:
                for t in topics_list:
                    topic_counts[t] = topic_counts.get(t, 0) + (1 / num_topics)
    df = pd.DataFrame(topic_counts.items(), columns=["topic", "consultation_count_normalised"])
    return df

def adjust_topic_dataframe(df_raw, topic_counts):
    """
    Divides each column in df_raw by its usage count from topic_counts.
    If a topic's usage is 0 or missing, default to dividing by 1 (no change).
    """
    df = df_raw.copy()
    for col in df.columns:
        usage = topic_counts.get(col, 0)
        if usage == 0:
            usage = 1
        df[col] = df[col] / usage
    return df

def build_questionnaire_vs_open_df(yaml_data):
    """
    Create a table with columns: [consultation_name, open_feedback, questionnaire_feedback].
    """
    rows = []
    for consultation_name, details in yaml_data.items():
        sub_page = details.get("sub_page", {})
        open_val = sub_page.get("amount_feedback_one", np.nan)
        feedback_two = details.get("feedback_two", {})
        q_val = feedback_two.get("amount_feedback_two", np.nan)
        rows.append({
            "consultation_name": consultation_name,
            "open_feedback": open_val,
            "questionnaire_feedback": q_val
        })
    return pd.DataFrame(rows)

def build_topic_geography_dfs(yaml_data):
    """
    Build two data frames:
      - df_topic_country_open
      - df_topic_country_questionnaire
    distributing the counts among topics if multiple.
    """
    df_open = distribute_by_topic(yaml_data, 'feedback_one', 'by_country_respondent')
    df_questionnaire = distribute_by_topic(yaml_data, 'feedback_two', 'by_country_respondent')
    return df_open, df_questionnaire

def build_topic_response_df(yaml_data, feedback_key='feedback_two'):
    """
    Builds a DataFrame where rows are consultations and columns are topics.
    Values are the total feedback responses (not by country), distributed equally across listed topics.

    Parameters:
    - yaml_data: Parsed YAML dictionary
    - feedback_key: 'feedback_one' (open) or 'feedback_two' (questionnaire)

    Returns:
    - A DataFrame with shape (n_consultations, n_topics)
    """
    rows = []
    for consultation_name, details in yaml_data.items():
        topic_str = details.get("main_page", {}).get("topic", "")
        topics = [t.strip() for t in topic_str.split(",") if t.strip()]
        num_topics = len(topics)
        if num_topics == 0:
            continue
        if feedback_key == 'feedback_one':
            total = details.get("sub_page", {}).get("amount_feedback_one", 0)
        else:
            total = details.get("feedback_two", {}).get("amount_feedback_two", 0)
        if not isinstance(total, (int, float)) or total == 0:
            continue
        scaled = {topic: total / num_topics for topic in topics}
        scaled['consultation'] = consultation_name
        rows.append(scaled)
    df = pd.DataFrame(rows)
    return df.set_index("consultation")

def build_topic_respondent_dfs(yaml_data):
    """
    Build two data frames:
      - df_topic_respondent_open
      - df_topic_respondent_questionnaire
    distributing the counts among topics if multiple.
    """
    df_open = distribute_by_topic(yaml_data, 'feedback_one', 'by_category_respondent')
    df_questionnaire = distribute_by_topic(yaml_data, 'feedback_two', 'by_category_respondent')
    return df_open, df_questionnaire

def collapse_non_matching_rows(df, valid_list, other_label="Excluded Countries"):
    """
    Sums together all rows that are NOT in valid_list into one row labeled `other_label`,
    and removes them from the original DataFrame.

    Parameters:
    - df (pd.DataFrame): DataFrame with countries as index.
    - valid_list (list or set): List of country names to keep.
    - other_label (str): Name for the new 'Other' row.

    Returns:
    - pd.DataFrame with only valid countries + one 'Other' row.
    """
    valid_set = set(valid_list)
    df_valid = df[df.index.isin(valid_set)].copy()
    df_other = df[~df.index.isin(valid_set)]
    if not df_other.empty:
        other_sum = df_other.sum()
        df_valid.loc[other_label] = other_sum
    return df_valid

def build_responses_per_million_df(yaml_data):
    """
    Summation of all responses per country for feedback_one and feedback_two,
    then compute per million using updated population_data_updated.

    Returns a DF with columns:
      [ country, feedback_one_total, feedback_two_total, feedback_one_per_million, feedback_two_per_million ]
    """
    from collections import defaultdict
    f1_sums = defaultdict(float)
    f2_sums = defaultdict(float)
    for consultation_name, details in yaml_data.items():
        f1_data = details.get('feedback_one', {}).get('by_country_respondent', {})
        for c, val in f1_data.items():
            f1_sums[c] += val
        f2_data = details.get('feedback_two', {}).get('by_country_respondent', {})
        for c, val in f2_data.items():
            f2_sums[c] += val
    rows = []
    all_countries = set(list(f1_sums.keys()) + list(f2_sums.keys()))
    for country in sorted(all_countries):
        f1_total = f1_sums[country]
        f2_total = f2_sums[country]
        pop = population_data_updated.get(country, 0)
        if pop > 0:
            f1_pm = f1_total / pop
            f2_pm = f2_total / pop
        else:
            f1_pm = np.nan
            f2_pm = np.nan
        rows.append({
            "country": country,
            "feedback_one_total": f1_total,
            "feedback_two_total": f2_total,
            "feedback_one_per_million": f1_pm,
            "feedback_two_per_million": f2_pm
        })
    df = pd.DataFrame(rows)
    return df

def build_open_timeframe_df(yaml_data):
    """
    For each consultation, parse feedback_period_from / to.
    We'll classify each consultation as Upcoming, Open, or Closed relative to 09 Jan 2025,
    and assign 'normalised_timeframe' accordingly:

      - Upcoming:
          If both feedback counts are 0 => normalised_timeframe = 0
          If at least one feedback count > 0 => normalised_timeframe = 1
            and the other count if 0 => NaN
      - Closed:
          normalised_timeframe = 1
          any feedback count = 0 => NaN
      - Open:
          normalised_timeframe computed as (days elapsed) / (total period).
          feedback counts remain as is.

    Returns a DataFrame with columns:
      [feedback_period_from, feedback_period_to, normalised_timeframe, open_feedback_count, questionnaire_feedback_count]
    """
    results = []
    for consultation_name, details in yaml_data.items():
        from_str = details.get("main_page", {}).get("feedback_period_from", "")
        to_str = details.get("main_page", {}).get("feedback_period_to", "")
        try:
            if from_str:
                start_date = parser.parse(from_str)
            else:
                continue
            if to_str:
                end_date = parser.parse(to_str)
            else:
                continue
        except Exception:
            continue
        if end_date < start_date:
            continue
        open_val = details.get("sub_page", {}).get("amount_feedback_one", 0)
        question_val = details.get("feedback_two", {}).get("amount_feedback_two", 0)
        if ref_date < start_date:
            if open_val == 0 and question_val == 0:
                normalised = 0
            else:
                normalised = 1
                if open_val == 0:
                    open_val = np.nan
                if question_val == 0:
                    question_val = np.nan
        elif ref_date > end_date:
            normalised = 1
            if open_val == 0:
                open_val = np.nan
            if question_val == 0:
                question_val = np.nan
        else:
            total_span = (end_date - start_date).days
            if total_span == 0:
                normalised = 1.0
            else:
                elapsed = (ref_date - start_date).days
                normalised = elapsed / total_span
        results.append({
            "feedback_period_from": from_str,
            "feedback_period_to": to_str,
            "normalised_timeframe": round(normalised, 3),
            "open_feedback_count": open_val,
            "questionnaire_feedback_count": question_val
        })
    df = pd.DataFrame(results)
    return df

def get_all_dataframes(yaml_path):
    """Generates all relevant Dataframes for the Consulation data for further analysis."""
    with open(yaml_path, 'r') as f:
        yaml_data = yaml.load(f, Loader=yaml.FullLoader)
    df_qvso = build_questionnaire_vs_open_df(yaml_data)
    df_tg_open_raw, df_tg_questionnaire_raw = build_topic_geography_dfs(yaml_data)
    topic_usage_counts_open = compute_topic_usage_for_feedback(yaml_data, 'feedback_one', 'by_country_respondent')
    topic_usage_counts_questionnaire = compute_topic_usage_for_feedback(yaml_data, 'feedback_two', 'by_country_respondent')
    topic_counts_open = df_tg_open_raw.sum(axis=0)
    topic_counts_questionnaire = df_tg_questionnaire_raw.sum(axis=0)
    df_consultations_per_topic = consultations_per_topic(yaml_path)
    countries_included = list(population_data_updated.keys())
    df_tg_open_adjusted = adjust_topic_dataframe(df_tg_open_raw, topic_usage_counts_open)
    df_tg_questionnaire_adjusted = adjust_topic_dataframe(df_tg_questionnaire_raw, topic_usage_counts_questionnaire)
    df_tg_open_adjusted = collapse_non_matching_rows(df_tg_open_adjusted, countries_included)
    df_tg_questionnaire_adjusted = collapse_non_matching_rows(df_tg_questionnaire_adjusted, countries_included)
    df_tg_open_adjusted.index.name = "country"
    df_tg_questionnaire_adjusted.index.name = "country"
    df_r_questionnaire = build_topic_response_df(yaml_data, feedback_key='feedback_two').sum(axis=1)
    df_r_open = build_topic_response_df(yaml_data, feedback_key='feedback_one').sum(axis=1)
    df_r_questionnaire_percentages = (df_r_questionnaire / df_r_questionnaire.sum()) * 100
    df_r_open_percentages = (df_r_open / df_r_open.sum()) * 100
    df_tr_open_raw, df_tr_questionnaire_raw = build_topic_respondent_dfs(yaml_data)
    df_tr_open_adjusted = adjust_topic_dataframe(df_tr_open_raw, topic_usage_counts_open)
    df_tr_questionnaire_adjusted = adjust_topic_dataframe(df_tr_questionnaire_raw, topic_usage_counts_questionnaire)
    df_tr_open_adjusted.index.name = "respondent"
    df_tr_questionnaire_adjusted.index.name = "respondent"
    df_topic_responses_questionnaire = build_topic_response_df(yaml_data, feedback_key='feedback_two')
    df_topic_responses_open = build_topic_response_df(yaml_data, feedback_key='feedback_one')
    sankey_triplet = build_country_topic_respondent_triplets(yaml_data)
    df_per_million = build_responses_per_million_df(yaml_data)
    df_open_timeframe = build_open_timeframe_df(yaml_data)
    df_stages = count_responses_by_custom_stage('/Users/ansgarkamratowski/Desktop/Data quality and data wrangling/EU/scrape_your_say.yaml')
    return {
        'questionnaire_vs_open': df_qvso,
        'topic_responses_questionnaire': df_topic_responses_questionnaire,
        'topic_responses_open': df_topic_responses_open,
        'topic_counts_open': topic_counts_open,
        'topic_counts_questionnaire': topic_counts_questionnaire,
        'topic_geography_open': df_tg_open_raw,
        'topic_geography_questionnaire': df_tg_questionnaire_raw,
        'topic_geography_open_adjusted': df_tg_open_adjusted,
        'topic_geography_questionnaire_adjusted': df_tg_questionnaire_adjusted,
        'topic_respondent_open': df_tr_open_raw,
        'topic_respondent_questionnaire': df_tr_questionnaire_raw,
        'respondant_questionnaire': df_r_questionnaire,
        'respondant_open': df_r_open,
        'respondant_questionnaire_percentages': df_r_questionnaire_percentages,
        'respondant_open_percentages': df_r_open_percentages,
        'topic_respondent_open_adjusted': df_tr_open_adjusted,
        'topic_respondent_questionnaire_adjusted': df_tr_questionnaire_adjusted,
        'responses_per_million': df_per_million,
        'open_timeframe': df_open_timeframe,
        'consultations_per_topic': df_consultations_per_topic,
        'stages': df_stages,
        'sankey_diagram_data': sankey_triplet
    }

results = get_all_dataframes('/Users/ansgarkamratowski/Desktop/Data quality and data wrangling/EU/scrape_your_say.yaml')

questionnaire_vs_open = results['questionnaire_vs_open']
topic_responses_questionnaire = results['topic_responses_questionnaire']
topic_responses_open = results['topic_responses_open']
topic_counts_open = results['topic_counts_open']
topic_counts_questionnaire = results['topic_counts_questionnaire']
topic_geography_open = results['topic_geography_open']
topic_geography_questionnaire = results['topic_geography_questionnaire']
topic_geography_open_adjusted = results['topic_geography_open_adjusted']
topic_geography_questionnaire_adjusted = results['topic_geography_questionnaire_adjusted']
topic_respondent_open = results['topic_respondent_open']
topic_respondent_questionnaire = results['topic_respondent_questionnaire']
respondant_questionnaire = results['respondant_questionnaire']
respondant_open = results['respondant_open']
respondant_questionnaire_percentages = results['respondant_questionnaire_percentages']
respondant_open_percentages = results['respondant_open_percentages']
topic_respondent_open_adjusted = results['topic_respondent_open_adjusted']
topic_respondent_questionnaire_adjusted = results['topic_respondent_questionnaire_adjusted']
responses_per_million = results['responses_per_million']
open_timeframe = results['open_timeframe']
consultations_per_topic = results['consultations_per_topic']
stages = results['stages']
sankey_diagram_data = results['sankey_diagram_data']


def summarise_distribution(df, trim_prop=0.1):
    """
    Calculates basic distribution statistics for each numerical column in the DataFrame.
    
    Parameters:
        df (pd.DataFrame): Input DataFrame.
        trim_prop (float): Proportion to trim from each end for the trimmed mean (default is 0.1).
        
    Returns:
        pd.DataFrame: Summary statistics for each numerical column.
    """
    numeric_df = df.select_dtypes(include=[np.number])
    summary = {}
    for col in numeric_df.columns:
        series = numeric_df[col].dropna()
        summary[col] = {
            'mean': series.mean(),
            'median': series.median(),
            'variance': series.var(),
            'std_dev': series.std(),
            'trimmed_mean': trim_mean(series, proportiontocut=trim_prop),
            'skewness': skew(series),
            'kurtosis': kurtosis(series),
            'min': series.min(),
            'max': series.max(),
            'non_null_count': series.count()
        }
    return pd.DataFrame(summary).T

def chi_square_with_cramers_v(df):
    """
    Performs a Chi-square test of homogeneity and computes Cramér's V.
    
    Parameters:
        df (pd.DataFrame): DataFrame where rows represent countries and columns represent topics.
    
    Returns:
        dict: A dictionary with keys:
              - "chi2": The Chi-square statistic.
              - "p_value": The p-value of the test.
              - "dof": Degrees of freedom.
              - "expected": The expected count table under the null hypothesis.
              - "cramers_v": The effect size measure (Cramér's V).
    """
    chi2_stat, p_value, dof, expected_array = chi2_contingency(df.values)
    n = df.values.sum()
    r, c = df.shape
    min_dim = min(r - 1, c - 1)
    if min_dim > 0:
        cramers_v = np.sqrt(chi2_stat / (n * min_dim))
    else:
        cramers_v = np.nan
    return {
        "chi2": chi2_stat,
        "p_value": p_value,
        "dof": dof,
        "expected": expected_array,
        "cramers_v": cramers_v
    }

def two_sample_kolmogorov(data1, data2):
    """
    Performs the two-sample Kolmogorov–Smirnov test between two datasets.
    
    Parameters:
        data1, data2: Arrays or lists of data.
        
    Returns:
        Result from ks_2samp.
    """
    return ks_2samp(data1, data2, nan_policy='omit')

def plot_row_normalised_heatmap(csv_file, title="Row-Normalized Topic–Country Response Distribution", figsize=(12, 8), cmap="Blues"):
    """
    Reads a CSV of shape (countries × topics), computes row-wise proportions, and plots a heatmap.
    
    Parameters:
        csv_file (str): CSV filename with the first column as the index.
        title (str): Title of the plot.
        figsize (tuple): Size of the figure.
        cmap (str): Colormap to use.
    """
    df = pd.read_csv(csv_file, index_col=0)
    row_sums = df.sum(axis=1)
    df_normalised = df.div(row_sums, axis=0).fillna(0)
    plt.figure(figsize=figsize)
    sns.heatmap(df_normalised, cmap=cmap, annot=False, cbar=True, vmin=0.0, vmax=1.0)
    plt.title(title, fontsize=14, pad=12)
    plt.ylabel("Countries")
    plt.xlabel("Topics")
    plt.tight_layout()
    plt.show()

def plot_split_overlay_histograms(df, bins=40, value_range=(0, 400), alpha1=0.8, alpha2=0.5):
    """
    Splits the numeric columns of a DataFrame in half and plots overlay histograms for each matching pair.
    
    Parameters:
        df (pd.DataFrame): DataFrame containing numeric columns.
        bins (int): Number of histogram bins.
        value_range (tuple): Range for the x-axis.
        alpha1 (float): Transparency for the first histogram.
        alpha2 (float): Transparency for the second histogram.
    """
    numeric_cols = df.select_dtypes(include='number').columns.tolist()
    midpoint = len(numeric_cols) // 2
    cols_one = numeric_cols[:midpoint]
    cols_two = numeric_cols[midpoint:midpoint+len(cols_one)]
    for col1, col2 in zip(cols_one, cols_two):
        series1 = df[col1].dropna()
        series2 = df[col2].dropna()
        mean1, median1 = series1.mean(), series1.median()
        mean2, median2 = series2.mean(), series2.median()
        plt.figure(figsize=(10, 6))
        plt.hist(series1, bins=bins, range=value_range, color='#333333', alpha=alpha1, edgecolor='black', density=True, label=f'{col1}')
        plt.hist(series2, bins=bins, range=value_range, color='#17becf', alpha=alpha2, edgecolor='black', density=True, label=f'{col2}')
        plt.axvline(mean1, color='#333333', linestyle='--', linewidth=1.8, label=f'{col1} mean')
        plt.axvline(median1, color='#333333', linestyle='-.', linewidth=1.8, label=f'{col1} median')
        plt.axvline(median2, color='#17becf', linestyle='-.', linewidth=1.8, label=f'{col2} median')
        plt.xlabel("Value")
        plt.ylabel("Density")
        plt.title(f"Histogram: {col1} vs {col2}")
        plt.legend()
        plt.tight_layout()
        plt.show()

def get_matching_row_pairs(df, col1, col2, min_value):
    """
    Returns rows where the values in col1 and col2 are at least min_value and appear more than once.
    
    Parameters:
        df (pd.DataFrame): Input DataFrame.
        col1 (str): First column to examine.
        col2 (str): Second column to examine.
        min_value (numeric): Minimum threshold value.
        
    Returns:
        pd.DataFrame: Rows with duplicated (col1, col2) pairs meeting the threshold.
    """
    filtered = df[(df[col1] >= min_value) & (df[col2] >= min_value)]
    return filtered[filtered.duplicated(subset=[col1, col2], keep=False)]

def plot_violin_responses(df, title="Distribution of Responses by Consultation Stage", log_scale=True, single=False):
    """
    Plots a violin plot for response distributions across consultation stages.
    
    Parameters:
        df (pd.DataFrame): DataFrame with columns representing stages and rows representing response counts.
        title (str): Plot title.
        log_scale (bool): If True, apply log transformation to responses.
        single (bool): If True, assume df is already in long format.
    """
    df_melted = df if single else df.melt(var_name="Stage", value_name="Responses")
    if log_scale:
        df_melted["Responses"] = np.log1p(df_melted["Responses"])
    plt.figure(figsize=(12, 6))
    ax = sns.violinplot(x="Stage", y="Responses", data=df_melted, palette="viridis")
    if log_scale:
        ax.set_ylabel("Number of Responses (log scale)")
        formatter = FuncFormatter(lambda y, _: f'{int(np.expm1(y)):,}')
        ax.yaxis.set_major_formatter(formatter)
        ax.set_ylim(ymin=0)
    else:
        ax.set_ylabel("Number of Responses")
        ax.set_ylim(ymin=0)
    plt.xticks(rotation=45, ha="right")
    plt.xlabel("Consultation Stage")
    plt.title(title)
    plt.tight_layout()
    plt.show()

def plot_general_violin(df, title="Violin Plot of Rows", log_scale=False):
    """
    Plots a horizontal violin plot where each row in the DataFrame is treated as a separate distribution.
    
    Parameters:
        df (pd.DataFrame): DataFrame where each row represents a distribution.
        title (str): Plot title.
        log_scale (bool): If True, apply log transformation to the values.
    """
    df_melted = df.melt(var_name="Row", value_name="Value")
    if log_scale:
        df_melted["Value"] = np.log1p(df_melted["Value"])
    category_order = df_melted.groupby("Row")["Value"].median().sort_values(ascending=False).index.tolist()
    plt.figure(figsize=(10, len(category_order) * 0.3 + 3))
    ax = sns.violinplot(y="Row", x="Value", data=df_melted, palette="viridis", order=category_order)
    if log_scale:
        ax.set_xlabel("Value (log scale)")
        formatter = FuncFormatter(lambda x, _: f'{np.expm1(x):,.0f}')
        ax.xaxis.set_major_formatter(formatter)
        ax.set_xlim(xmin=0)
    else:
        ax.set_xlabel("Value")
        ax.set_xlim(xmin=0)
    ax.set_ylabel("Category")
    plt.title(title)
    plt.tight_layout()
    plt.show()

def plot_comparison_bar_chart(data1, data2, label1="Dataset 1", label2="Dataset 2", title="Comparison of Distributions", y_label='percentages'):
    """
    Plots side-by-side bar charts comparing two categorical distribution datasets.
    
    Parameters:
        data1 (dict): Categories and values for the first dataset.
        data2 (dict): Categories and values for the second dataset.
        label1 (str): Label for the first dataset.
        label2 (str): Label for the second dataset.
        title (str): Plot title.
        y_label (str): Label for the y-axis.
    """
    categories = sorted(set(data1.keys()).union(set(data2.keys())))
    values1 = [data1.get(cat, 0) for cat in categories]
    values2 = [data2.get(cat, 0) for cat in categories]
    bar_width = 0.4
    x = np.arange(len(categories))
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.bar(x - bar_width/2, values1, width=bar_width, label=label1, color="#333333")
    ax.bar(x + bar_width/2, values2, width=bar_width, label=label2, color="#17becf")
    ax.set_xlabel("Categories")
    ax.set_ylabel(y_label)
    ax.set_title(title)
    ax.set_xticks(x)
    ax.set_xticklabels(categories, rotation=45, ha="right")
    ax.legend()
    plt.tight_layout()
    plt.show()

def plot_responses_map(df_per_million, title="Responses Per Million by Country"):
    """
    Plots a choropleth map showing responses per million for open and questionnaire feedback.
    
    Parameters:
        df_per_million (pd.DataFrame): Should include 'country', 'feedback_one_per_million', and 'feedback_two_per_million'.
        title (str): Plot title.
    
    Note:
        The variable 'population_data_updated' must be defined in the namespace.
    """
    world = gpd.read_file(gpd.datasets.get_path('naturalearth_lowres'))
    relevant_countries = set(population_data_updated.keys())
    df_filtered = df_per_million[df_per_million['country'].isin(relevant_countries)]
    world = world.merge(df_filtered, how="left", left_on="name", right_on="country")
    world = world[world['country'].notna()]
    bounds = world.total_bounds
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    world.plot(column="feedback_one_per_million", cmap="Blues", linewidth=0.8, edgecolor="gray",
               legend=True, ax=axes[0], missing_kwds={"color": "lightgray"})
    axes[0].set_title("Open Feedback Per Million")
    axes[0].set_xlim(bounds[0], bounds[2])
    axes[0].set_ylim(bounds[1], bounds[3])
    world.plot(column="feedback_two_per_million", cmap="Blues", linewidth=0.8, edgecolor="gray",
               legend=True, ax=axes[1], missing_kwds={"color": "lightgray"})
    axes[1].set_title("Questionnaire Feedback Per Million")
    for ax in axes:
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_frame_on(False)
    fig.suptitle(title, fontsize=14)
    plt.show()

def plot_alluvial_from_triplets(df, min_weight=1):
    """
    Plots an alluvial (parallel categories) diagram from a DataFrame with columns
    ['country', 'topic', 'respondent', 'weight'].
    
    Parameters:
        df (pd.DataFrame): Input DataFrame.
        min_weight (numeric): Minimum weight to include in the plot.
    """
    df_filtered = df[df['weight'] > min_weight].copy()
    fig = px.parallel_categories(
        df_filtered,
        dimensions=['respondent', 'topic', 'country'],
        color='weight',
        color_continuous_scale='Blues',
        labels={
            'respondent': 'Respondent Type',
            'topic': 'Policy Topic',
            'country': 'Country'
        }
    )
    fig.update_layout(
        title="Alluvial Flow: Respondent → Topic ← Country",
        font_size=10,
        margin=dict(l=50, r=50, t=50, b=50)
    )
    fig.show()

def melt_sankey_matrix(df, index_col):
    """
    Melts a DataFrame to produce an edge list suitable for Sankey diagrams.
    
    Parameters:
        df (pd.DataFrame): Input DataFrame.
        index_col (str): Column name to use as the identifier.
    
    Returns:
        pd.DataFrame: Melted DataFrame with columns [index_col, 'topic', 'weight'].
    """
    return df.reset_index().melt(id_vars=index_col, var_name='topic', value_name='weight')

def generate_bidirectional_sankey_from_matrix_sources(respondent_matrix_df, country_matrix_df, respondent_index='respondent', country_index='country', topic_col='topic', value_col='weight', min_value=1):
    """
    Generates a bidirectional Sankey diagram from respondent and country matrices.
    
    Parameters:
        respondent_matrix_df, country_matrix_df (pd.DataFrame): Input matrices.
        respondent_index (str): Name of the respondent index column.
        country_index (str): Name of the country index column.
        topic_col (str): Name of the topic column.
        value_col (str): Column containing weights.
        min_value (numeric): Minimum weight to include.
    """
    left_df = melt_sankey_matrix(respondent_matrix_df, respondent_index)
    right_df = melt_sankey_matrix(country_matrix_df, country_index)
    left_df = left_df[left_df[value_col] >= min_value]
    right_df = right_df[right_df[value_col] >= min_value]
    left_sources = left_df[respondent_index].unique().tolist()
    right_sources = right_df[country_index].unique().tolist()
    centers = pd.unique(pd.concat([left_df[topic_col], right_df[topic_col]])).tolist()
    right_sources_renamed = [f"{r} [R]" for r in right_sources]
    all_nodes = left_sources + centers + right_sources_renamed
    node_map = {label: idx for idx, label in enumerate(all_nodes)}
    left_df['source'] = left_df[respondent_index].map(node_map)
    left_df['target'] = left_df[topic_col].map(node_map)
    right_df['source'] = right_df[topic_col].map(node_map)
    right_df['target'] = right_df[country_index].apply(lambda x: node_map[f"{x} [R]"])
    all_links = pd.concat([left_df[['source', 'target', value_col]],
                           right_df[['source', 'target', value_col]]], ignore_index=True)
    fig = go.Figure(data=[go.Sankey(
        arrangement="snap",
        node=dict(
            pad=15, thickness=20, line=dict(color="black", width=0.5),
            label=all_nodes
        ),
        link=dict(
            source=all_links['source'],
            target=all_links['target'],
            value=all_links[value_col]
        )
    )])
    fig.update_layout(title_text=f"Bidirectional Sankey Diagram (min ≥ {min_value})", font_size=10)
    fig.show()

def generate_sankey_with_top_nodes(respondent_matrix_df, country_matrix_df,respondent_index='respondent', country_index='country',topic_col='topic', value_col='weight',top_n_left=5, top_n_right=5, top_n_center=10):
    """
    Generates a Sankey diagram grouping lower-value nodes under 'Other'.
    
    Parameters:
        respondent_matrix_df, country_matrix_df (pd.DataFrame): Input matrices.
        respondent_index (str): Name of the respondent index column.
        country_index (str): Name of the country index column.
        topic_col (str): Name of the topic column.
        value_col (str): Column with weight values.
        top_n_left (int): Top N respondents to show.
        top_n_right (int): Top N countries to show.
        top_n_center (int): Top N topics to show.
    """
    left_df = melt_sankey_matrix(respondent_matrix_df, respondent_index)
    right_df = melt_sankey_matrix(country_matrix_df, country_index)
    top_left = left_df.groupby(respondent_index)[value_col].sum().nlargest(top_n_left).index
    top_center = pd.concat([left_df, right_df]).groupby(topic_col)[value_col].sum().nlargest(top_n_center).index
    top_right = right_df.groupby(country_index)[value_col].sum().nlargest(top_n_right).index
    left_df[respondent_index] = left_df[respondent_index].apply(lambda x: x if x in top_left else 'Other (respondents)')
    left_df[topic_col] = left_df[topic_col].apply(lambda x: x if x in top_center else 'Other (topics)')
    right_df[country_index] = right_df[country_index].apply(lambda x: x if x in top_right else 'Other (countries)')
    right_df[topic_col] = right_df[topic_col].apply(lambda x: x if x in top_center else 'Other (topics)')
    left_grouped = left_df.groupby([respondent_index, topic_col], as_index=False)[value_col].sum()
    right_grouped = right_df.groupby([topic_col, country_index], as_index=False)[value_col].sum()
    all_nodes = pd.unique(left_grouped[respondent_index].tolist() +
                            left_grouped[topic_col].tolist() +
                            right_grouped[country_index].apply(lambda x: f"{x} [R]").tolist())
    node_map = {label: idx for idx, label in enumerate(all_nodes)}
    left_grouped['source'] = left_grouped[respondent_index].map(node_map)
    left_grouped['target'] = left_grouped[topic_col].map(node_map)
    right_grouped['source'] = right_grouped[topic_col].map(node_map)
    right_grouped['target'] = right_grouped[country_index].apply(lambda x: node_map[f"{x} [R]"])
    all_links = pd.concat([left_grouped[['source', 'target', value_col]],
                           right_grouped[['source', 'target', value_col]]], ignore_index=True)
    node_labels = list(node_map.keys())

    def rgb_to_hex(rgb_str):
        nums = [int(x) for x in rgb_str.strip("rgb()").split(",")]
        return '#%02x%02x%02x' % tuple(nums)

    def with_opacity(hex_color, alpha=0.5):
        rgba = mcolors.to_rgba(hex_color, alpha)
        return f'rgba({int(rgba[0]*255)},{int(rgba[1]*255)},{int(rgba[2]*255)},{rgba[3]})'

    color_palette = pc.qualitative.Set2
    node_colors = [color_palette[i % len(color_palette)] for i in range(len(node_labels))]
    hex_colors = [rgb_to_hex(c) for c in node_colors]
    index_to_color = {idx: hex_colors[idx] for idx in range(len(node_labels))}
    link_colors = []
    for _, row in all_links.iterrows():
        source_idx = int(row['source'])
        target_idx = int(row['target'])
        if node_labels[target_idx].endswith(" [R]"):
            color_hex = index_to_color[target_idx]
        else:
            color_hex = index_to_color[source_idx]
        link_colors.append(with_opacity(color_hex, 0.5))
    fig = go.Figure(data=[go.Sankey(
        arrangement="freeform",
        node=dict(
            pad=25,
            thickness=20,
            line=dict(color="black", width=0.5),
            label=node_labels,
            color=node_colors
        ),
        link=dict(
            source=all_links['source'],
            target=all_links['target'],
            value=all_links[value_col],
            color=link_colors
        )
    )])
    fig.update_layout(title_text="Sankey with Top Categories (Others Grouped)", font_size=10, width=1000, height=600)
    fig.show()
