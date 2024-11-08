#  Probably should separate

# General Imports
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

def analyze_categorical_features_enhanced(data):
    """
    Enhanced analysis of categorical features with additional metrics and better formatting
    """
    # Define categorical columns based on the actual data
    categorical_columns = [
    'Alternative Dispute Resolution',  # Y/N values
    'Attorney Representative',         # Y/N values
    'Carrier Name',                   # Company names
    'Carrier Type',                   # Types like "1A. PRIVATE"
    'Claim Injury_Type',              # Types like "2. NON-COMP"
    'County of Injury',               # County names
    'COVID_19 Indicator',             # Y/N values
    'District Name',                  # District names
    'Gender',                         # M/F values
    'Industry Code Description',      # Industry descriptions
    'Medical Fee Region',             # Regions I, II, III, IV
    'OIICS_Nature of Injury Description',  # Injury descriptions
    'WCIO Cause of Injury Description',    # Cause descriptions
    'WCIO Nature of Injury Description',   # Nature descriptions
    'WCIO Part Of Body Description',       # Body part descriptions
    'Agreement Reached',              # 0/1 values
    'WCB Decision'                    # Decision descriptions
]

    analysis_results = {}

    for column in categorical_columns:
        if column in data.columns:  # Check if column exists in dataframe
            print(f"\n{'='*70}")
            print(f"DETAILED ANALYSIS FOR: {column}")
            print('='*70)

            # Value counts with percentages
            value_counts = data[column].value_counts()
            value_percentages = data[column].value_counts(normalize=True) * 100

            print("\nTop 3 Values Distribution:")
            print("-"*50)
            for val, count in value_counts.head(3).items():
                percentage = value_percentages[val]
                print(f"{val}: {count:,} occurrences ({percentage:.2f}%)")

            # Unique values analysis
            unique_count = data[column].nunique()
            total_count = len(data)
            uniqueness_ratio = (unique_count / total_count) * 100
            print(f"\nUniqueness Analysis:")
            print("-"*50)
            print(f"Unique values: {unique_count:,}")
            print(f"Uniqueness ratio: {uniqueness_ratio:.2f}% of total records")

            # Numeric statistics if applicable
            if pd.api.types.is_numeric_dtype(data[column]):
                print("\nNumerical Statistics:")
                print("-"*50)
                stats = data[column].describe()
                print(stats)

            # Missing values analysis
            missing_count = data[column].isnull().sum()
            missing_percentage = (missing_count / len(data)) * 100
            print(f"\nMissing Values Analysis:")
            print("-"*50)
            print(f"Missing count: {missing_count:,}")
            print(f"Missing percentage: {missing_percentage:.2f}%")

            # Additional cardinality metrics for non-numeric data
            if not pd.api.types.is_numeric_dtype(data[column]):
                print("\nCardinality Analysis:")
                print("-"*50)
                rare_values = value_counts[value_counts == 1].count()
                print(f"Number of values appearing only once: {rare_values:,}")
                print(f"Percentage of rare values: {(rare_values/unique_count)*100:.2f}%")

            # Store results
            analysis_results[column] = {
                'unique_count': unique_count,
                'missing_count': missing_count,
                'missing_percentage': missing_percentage,
                'value_counts': value_counts,
                'uniqueness_ratio': uniqueness_ratio
            }
        else:
            print(f"\nWarning: Column '{column}' not found in the dataset")

    return analysis_results




def analyze_categorical_features_enhanced(data):
    """
    Enhanced analysis of categorical features with additional metrics and better formatting
    """
    categorical_columns = [
        'Age at Injury',
        'Average Weekly Wage',
        'Birth Year',
        'IME-4 Count',
        'Industry Code',
        'OIICS Nature of Injury Description',
        'WCIO Cause of Injury Code',
        'WCIO Nature of Injury Code',
        'WCIO Part Of Body Code',
        'Number of Dependents'
    ]

    analysis_results = {}

    for column in categorical_columns:
        print(f"\n{'='*70}")
        print(f"DETAILED ANALYSIS FOR: {column}")
        print('='*70)

        # Value counts with percentages
        value_counts = data[column].value_counts()
        value_percentages = data[column].value_counts(normalize=True) * 100

        print("\nTop 10 Values Distribution:")
        print("-"*50)
        for val, count in value_counts.head(3).items():
            percentage = value_percentages[val]
            print(f"{val}: {count:,} occurrences ({percentage:.2f}%)")

        # Unique values analysis
        unique_count = data[column].nunique()
        total_count = len(data)
        uniqueness_ratio = (unique_count / total_count) * 100
        print(f"\nUniqueness Analysis:")
        print("-"*50)
        print(f"Unique values: {unique_count:,}")
        print(f"Uniqueness ratio: {uniqueness_ratio:.2f}% of total records")

        # Numeric statistics if applicable
        if pd.api.types.is_numeric_dtype(data[column]):
            print("\nNumerical Statistics:")
            print("-"*50)
            stats = data[column].describe()


        # Missing values analysis
        missing_count = data[column].isnull().sum()
        missing_percentage = (missing_count / len(data)) * 100
        print(f"\nMissing Values Analysis:")
        print("-"*50)
        print(f"Missing count: {missing_count:,}")
        print(f"Missing percentage: {missing_percentage:.2f}%")

        # Additional cardinality metrics
        if not pd.api.types.is_numeric_dtype(data[column]):
            print("\nCardinality Analysis:")
            print("-"*50)
            rare_values = value_counts[value_counts == 1].count()
            print(f"Number of values appearing only once: {rare_values:,}")
            print(f"Percentage of rare values: {(rare_values/unique_count)*100:.2f}%")

        # Store results
        analysis_results[column] = {
            'unique_count': unique_count,
            'missing_count': missing_count,
            'missing_percentage': missing_percentage,
            'value_counts': value_counts,
            'uniqueness_ratio': uniqueness_ratio
        }

    return analysis_results


def create_comprehensive_boxplots(data):
    """
    Create boxplots for all relevant numerical columns with appropriate scaling
    """
    # Define numerical columns to plot
    numeric_cols = [
    'Age at Injury',                # Continuous
    'Average Weekly Wage',          # Continuous
    'Birth Year',                   # Discrete
    'Industry Code',                # Discrete
    'IME-4 Count',                  # Discrete
    'WCIO Cause of Injury Code',    # Discrete
    'WCIO Nature of Injury Code',   # Discrete
    'WCIO Part Of Body Code',       # Discrete
    'Number of Dependents',         # Discrete
    'Zip Code'                      # Discrete (could be categorical)
    ]

    df = data.copy()

    df.fillna('Missing', inplace=True)
    for col in numeric_cols:
        df[col] = df[col].astype(str)


    # Calculate number of rows and columns for subplot grid
    n_cols = 3
    n_rows = (len(numeric_cols) + n_cols - 1) // n_cols

    # Create subplots
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(20, 6*n_rows))
    fig.suptitle('Distribution of All Numerical Variables', fontsize=16, y=1.02)

    # Flatten axes array for easier iteration
    axes_flat = axes.flatten() if hasattr(axes, 'flatten') else [axes]

    for i, col in enumerate(numeric_cols):
        ax = axes_flat[i]

        # Create boxplot
        sns.boxplot(data=df, y=col, ax=ax, color='steelblue')

        # Calculate statistics
        stats = df[col].describe()
        stats_text = f'Mean: {stats["mean"]:.1f}\nMedian: {stats["50%"]:.1f}\nStd: {stats["std"]:.1f}'

        # Add statistics box
        ax.text(0.95, 0.95, stats_text,
                transform=ax.transAxes,
                verticalalignment='top',
                horizontalalignment='right',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

        # Customize plot based on variable
        ax.set_title(f'Distribution of {col}', pad=20)

        # Add sample size
        ax.set_xlabel(f'n={df[col].notna().sum():,}')

        # Variable-specific customization
        if col == 'Average Weekly Wage':
            ax.set_ylim(0, df[col].quantile(0.99))  # Show up to 99th percentile
            ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x:,.0f}'))
        elif col == 'Birth Year':
            ax.set_ylim(1940, 2020)
        elif col == 'Age at Injury':
            ax.set_ylim(0, 128)
        elif col == 'IME-4 Count':
            ax.set_ylim(0, df[col].quantile(0.99))
        elif col in ['WCIO Cause of Injury Code', 'WCIO Nature of Injury Code', 'WCIO Part Of Body Code']:
            ax.set_ylim(df[col].min(), df[col].max())

    # Remove empty subplots if any
    for j in range(i + 1, len(axes_flat)):
        fig.delaxes(axes_flat[j])

    plt.tight_layout()
    return fig

def print_numerical_summaries(df):
    """
    Print detailed summaries for all numerical columns
    """
    numeric_cols = [
    'Age at Injury',                # Continuous
    'Average Weekly Wage',          # Continuous
    'Birth Year',                   # Discrete
    'Industry Code',                # Discrete
    'IME-4 Count',                  # Discrete
    'WCIO Cause of Injury Code',    # Discrete
    'WCIO Nature of Injury Code',   # Discrete
    'WCIO Part Of Body Code',       # Discrete
    'Number of Dependents',         # Discrete
    'Zip Code'                      # Discrete (could be categorical)
    ]

    for col in numeric_cols:
        print(f"\n=== {col} ===")
        stats = df[col].describe()
        print(stats)
        print(f"Missing values: {df[col].isna().sum():,} ({df[col].isna().mean():.1%})")
        print("-" * 50)



def detect_outliers(series, method='zscore'):
    """
    Detect outliers using various methods

    Parameters:
    series: pandas Series - numerical data to analyze
    method: str - detection method ('zscore' or 'iqr')

    Returns:
    tuple: (outlier mask, lower bound, upper bound)
    """
    # Convert series to numpy array and handle missing values
    data = pd.to_numeric(series, errors='coerce').dropna().values

    if len(data) == 0:
        return np.array([]), np.nan, np.nan

    if method == 'zscore':
        z_scores = np.abs(stats.zscore(data))
        outliers = z_scores > 3
        lower_bound = np.mean(data) - 3 * np.std(data)
        upper_bound = np.mean(data) + 3 * np.std(data)
    else:  # IQR method
        Q1 = np.percentile(data, 25)
        Q3 = np.percentile(data, 75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        outliers = (data < lower_bound) | (data > upper_bound)

    return outliers, lower_bound, upper_bound

def create_histograms(df):
    """
    Create histograms for all numerical variables with appropriate binning, scaling,
    and outlier detection
    """
    # Identify numeric columns
    numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()

    # Calculate number of rows and columns for subplot grid
    n_cols = 3
    n_rows = (len(numeric_cols) + n_cols - 1) // n_cols

    # Create subplots
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(20, 6*n_rows))
    fig.suptitle('Distribution of Numerical Variables with Outlier Analysis', fontsize=16, y=1.02)

    # Flatten axes array for easier iteration
    axes_flat = axes.flatten() if hasattr(axes, 'flatten') else [axes]

    for i, col in enumerate(numeric_cols):
        ax = axes_flat[i]

        # Get non-null values for analysis
        valid_data = pd.to_numeric(df[col], errors='coerce').dropna()

        if len(valid_data) == 0:
            continue

        # Determine outlier detection method based on variable type
        outlier_method = 'iqr' if col in ['Average Weekly Wage', 'Age at Injury'] else 'zscore'

        # Detect outliers
        outliers, lower_bound, upper_bound = detect_outliers(valid_data, method=outlier_method)

        # Custom handling for each variable type
        if 'wage' in col.lower():
            # Plot non-zero wages up to 95th percentile
            non_zero_wages = valid_data[valid_data > 0]
            max_wage = non_zero_wages.quantile(0.95)
            sns.histplot(data=non_zero_wages[non_zero_wages <= max_wage],
                        bins=50, ax=ax)
            ax.set_xlabel(f'${col} (up to 95th percentile)')
            ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x:,.0f}'))

            # Add text about zeros, outliers and anomalies
            zero_pct = (valid_data == 0).mean() * 100
            outlier_pct = np.mean(outliers) * 100
            stats_text = (f'Zero values: {zero_pct:.1f}%\n'
                         f'Outliers: {outlier_pct:.1f}%\n'
                         f'Range: ${lower_bound:,.0f} - ${upper_bound:,.0f}')

        elif 'age' in col.lower():
            sns.histplot(data=valid_data, bins=np.arange(0, 120, 5), ax=ax)
            stats_text = (f'Mean: {valid_data.mean():.1f}\n'
                         f'Median: {valid_data.median():.1f}\n'
                         f'Outliers: {np.mean(outliers):.1%}\n'
                         f'Valid range: {lower_bound:.1f} - {upper_bound:.1f}')

        else:  # For other numerical variables
            sns.histplot(data=valid_data, bins=30, ax=ax)
            stats_text = (f'Mean: {valid_data.mean():.1f}\n'
                         f'Median: {valid_data.median():.1f}\n'
                         f'Outliers: {np.mean(outliers):.1%}\n'
                         f'Valid range: {lower_bound:.1f} - {upper_bound:.1f}')

        # Add statistics
        ax.text(0.95, 0.95, stats_text,
                transform=ax.transAxes,
                verticalalignment='top',
                horizontalalignment='right',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

        # Add title and sample size
        ax.set_title(f'Distribution of {col}', pad=20)
        n_samples = len(valid_data)
        ax.set_xlabel(f'{col} (n={n_samples:,})')

        # Rotate x-axis labels if needed
        if not any(x in col.lower() for x in ['dependent', 'count']):
            ax.tick_params(axis='x', rotation=45)

    # Remove empty subplots if any
    for j in range(i + 1, len(axes_flat)):
        fig.delaxes(axes_flat[j])

    plt.tight_layout()
    return fig

def print_distribution_stats(df):
    """Print detailed distribution statistics including outlier analysis for each numerical variable"""
    # Get numeric columns
    numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns

    for col in numeric_cols:
        print(f"\n=== {col} ===")

        # Convert to numeric and drop invalid values
        valid_data = pd.to_numeric(df[col], errors='coerce')

        # Basic statistics
        stats_df = valid_data.describe()
        print(stats_df)

        # Missing values analysis
        missing_count = valid_data.isna().sum()
        missing_pct = valid_data.isna().mean()
        print(f"Missing values: {missing_count:,} ({missing_pct:.1%})")

        # Zero values analysis (if applicable)
        if 'wage' in col.lower():
            zero_count = (valid_data == 0).sum()
            zero_pct = (valid_data == 0).mean()
            print(f"Zero values: {zero_count:,} ({zero_pct:.1%})")

        # Outlier analysis
        valid_values = valid_data.dropna()
        if len(valid_values) > 0:
            try:
                # Detect outliers using both methods
                zscore_outliers, zscore_lower, zscore_upper = detect_outliers(valid_values, 'zscore')
                iqr_outliers, iqr_lower, iqr_upper = detect_outliers(valid_values, 'iqr')

                print("\nOutlier Analysis:")
                print(f"Z-score method (±3σ):")
                print(f"- Outliers: {np.sum(zscore_outliers):,} ({np.mean(zscore_outliers):.1%})")
                print(f"- Valid range: {zscore_lower:.1f} to {zscore_upper:.1f}")

                print(f"\nIQR method (1.5×IQR):")
                print(f"- Outliers: {np.sum(iqr_outliers):,} ({np.mean(iqr_outliers):.1%})")
                print(f"- Valid range: {iqr_lower:.1f} to {iqr_upper:.1f}")

                # Additional anomaly checks
                skewness = stats.skew(valid_values)
                kurtosis = stats.kurtosis(valid_values)
                print(f"\nDistribution shape:")
                print(f"- Skewness: {skewness:.2f} ({'highly skewed' if abs(skewness) > 1 else 'moderately skewed' if abs(skewness) > 0.5 else 'approximately symmetric'})")
                print(f"- Kurtosis: {kurtosis:.2f} ({'heavy-tailed' if kurtosis > 1 else 'light-tailed' if kurtosis < -1 else 'normal-like'})")

            except Exception as e:
                print(f"Error in outlier analysis: {str(e)}")

        print("-" * 50)


def analyze_date_intervals(df):
    """
    Analyze intervals between different date variables with adjusted scales
    """
    # Convert date columns to datetime
    date_cols = ['Accident Date', 'Assembly Date', 'C-2 Date', 'C-3 Date', 'First Hearing Date']
    date_df = df[date_cols].apply(pd.to_datetime, errors='coerce')

    # Create figure
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    axes_flat = axes.flatten()

    # Define intervals to analyze
    intervals = [
        ('Accident Date', 'Assembly Date', 'Accident to Assembly', 90),  # Show up to 90 days
        ('Accident Date', 'First Hearing Date', 'Accident to First Hearing', 730),  # Show up to 2 years
        ('Assembly Date', 'First Hearing Date', 'Assembly to First Hearing', 730),  # Show up to 2 years
        ('C-2 Date', 'C-3 Date', 'C2 to C3', 60)  # Show up to 60 days
    ]

    for idx, (start_date, end_date, label, max_days) in enumerate(intervals):
        if all(col in date_df.columns for col in [start_date, end_date]):
            # Calculate interval in days
            interval = (date_df[end_date] - date_df[start_date]).dt.days

            # Filter outliers for visualization but keep them for stats
            interval_filtered = interval[(interval >= 0) & (interval <= max_days)]

            # Plot distribution of intervals
            ax = axes_flat[idx]

            # Create histogram with appropriate bin size
            bin_size = max_days // 30  # Adjust number of bins based on range
            sns.histplot(data=interval_filtered, bins=bin_size, ax=ax)

            # Calculate statistics on full data
            stats = interval.describe()
            outliers_pct = (len(interval) - len(interval_filtered)) / len(interval) * 100

            # Add comprehensive statistics
            stats_text = (
                f"Mean: {stats['mean']:.1f} days\n"
                f"Median: {stats['50%']:.1f} days\n"
                f"Std: {stats['std']:.1f} days\n"
                f"Range shown: 0-{max_days} days\n"
                f"Outliers: {outliers_pct:.1f}%\n"
                f"Total cases: {len(interval):,}"
            )

            ax.text(0.95, 0.95, stats_text,
                   transform=ax.transAxes,
                   verticalalignment='top',
                   horizontalalignment='right',
                   bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

            ax.set_title(f'Distribution of Days: {label}\n(Showing range: 0-{max_days} days)')
            ax.set_xlabel('Days')
            ax.set_ylabel('Count')

            # Set x-axis limits
            ax.set_xlim(0, max_days)

            # Add gridlines for better readability
            ax.grid(True, linestyle='--', alpha=0.7)

            # Rotate x-axis labels for better readability
            ax.tick_params(axis='x', rotation=45)

    plt.tight_layout()
    return fig

def print_interval_details(df):
    """
    Print detailed statistics about the intervals
    """
    date_cols = ['Accident Date', 'Assembly Date', 'C-2 Date', 'C-3 Date', 'First Hearing Date']
    date_df = df[date_cols].apply(pd.to_datetime, errors='coerce')

    intervals = [
        ('Accident Date', 'Assembly Date', 'Accident to Assembly'),
        ('Accident Date', 'First Hearing Date', 'Accident to First Hearing'),
        ('Assembly Date', 'First Hearing Date', 'Assembly to First Hearing'),
        ('C-2 Date', 'C-3 Date', 'C2 to C3')
    ]

    print("\nDetailed Interval Analysis:")
    print("="*80)

    for start_date, end_date, label in intervals:
        if all(col in date_df.columns for col in [start_date, end_date]):
            interval = (date_df[end_date] - date_df[start_date]).dt.days

            print(f"\n{label}:")
            print("-"*50)
            print(f"Total cases: {len(interval):,}")
            print(f"Valid intervals: {interval.notna().sum():,}")
            print(f"Missing intervals: {interval.isna().sum():,}")

            # Calculate percentiles
            percentiles = interval.quantile([0.25, 0.5, 0.75, 0.9, 0.95, 0.99])
            print("\nKey percentiles (days):")
            print(f"25th percentile: {percentiles[0.25]:.1f}")
            print(f"Median: {percentiles[0.5]:.1f}")
            print(f"75th percentile: {percentiles[0.75]:.1f}")
            print(f"90th percentile: {percentiles[0.9]:.1f}")
            print(f"95th percentile: {percentiles[0.95]:.1f}")
            print(f"99th percentile: {percentiles[0.99]:.1f}")

            # Calculate negative and extreme positive intervals
            neg_intervals = (interval < 0).sum()
            extreme_intervals = (interval > 365).sum()

            print("\nOutlier analysis:")
            print(f"Negative intervals: {neg_intervals:,} ({neg_intervals/len(interval)*100:.1f}%)")
            print(f"Intervals > 1 year: {extreme_intervals:,} ({extreme_intervals/len(interval)*100:.1f}%)")


def plot_categorical_distributions(data, figsize=(20, 15), max_categories=10):
    """
    Create bar plots for categorical variables with proper formatting and handling of multiple categories
    """
    # Set the style using seaborn
    sns.set_style("whitegrid")

    # Define categorical columns and verify their presence in the data
    categorical_columns = [
    'Alternative Dispute Resolution',  # Y/N values
    'Attorney Representative',         # Y/N values
    'Carrier Name',                   # Company names
    'Carrier Type',                   # Types like "1A. PRIVATE"
    'Claim Injury_Type',              # Types like "2. NON-COMP"
    'County of Injury',               # County names
    'COVID_19 Indicator',             # Y/N values
    'District Name',                  # District names
    'Gender',                         # M/F values
    'Industry Code Description',      # Industry descriptions
    'Medical Fee Region',             # Regions I, II, III, IV
    'OIICS_Nature of Injury Description',  # Injury descriptions
    'WCIO Cause of Injury Description',    # Cause descriptions
    'WCIO Nature of Injury Description',   # Nature descriptions
    'WCIO Part Of Body Description'       # Body part descriptions
    ]


    # Filter to only include columns that exist in the dataframe
    available_columns = [col for col in categorical_columns if col in data.columns]

    if not available_columns:
        raise ValueError("None of the specified categorical columns found in the dataframe")

    # Calculate number of rows needed (3 columns)
    n_cols = 3
    n_rows = (len(available_columns) + n_cols - 1) // n_cols

    # Create figure with a reasonable size
    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
    fig.suptitle('Distribution of Categorical Variables', fontsize=16, y=1.02)

    # Ensure axes is always a 2D array
    if n_rows == 1:
        axes = np.array([axes])
    if n_cols == 1:
        axes = axes.reshape(-1, 1)

    # Flatten axes for iteration
    axes_flat = axes.flatten()

    for idx, column in enumerate(available_columns):
        try:
            ax = axes_flat[idx]

            # Convert to string if necessary and get value counts
            if data[column].dtype == 'object' or pd.api.types.is_categorical_dtype(data[column]):
                series = data[column]
            else:
                series = data[column].astype(str)

            value_counts = series.value_counts()

            # Handle cases with many categories
            if len(value_counts) > max_categories:
                main_categories = value_counts.head(max_categories)
                others_sum = value_counts[max_categories:].sum()
                value_counts = pd.concat([main_categories, pd.Series({'Others': others_sum})])

            # Create bar plot
            bars = sns.barplot(x=value_counts.values, y=value_counts.index, ax=ax, palette='viridis')

            # Add title and labels
            ax.set_title(f'{column}\n(n={series.notna().sum():,}, Missing={series.isna().sum():,})',
                        fontsize=10, pad=10)
            ax.set_xlabel('Count', fontsize=8)

            # Add percentage labels on bars
            total = value_counts.sum()
            for i, v in enumerate(value_counts.values):
                percentage = (v/total) * 100
                if percentage >= 1:  # Only show percentages >= 1%
                    ax.text(v, i, f' {percentage:.1f}%', va='center', fontsize=8)

            # Adjust layout for better readability
            ax.tick_params(axis='y', labelsize=8)

            # Limit label length if too long
            labels = [str(label)[:30] + '...' if len(str(label)) > 30 else str(label)
                     for label in value_counts.index]
            ax.set_yticklabels(labels)

        except Exception as e:
            print(f"Error plotting {column}: {str(e)}")
            ax.text(0.5, 0.5, f"Error plotting {column}", ha='center', va='center')

    # Remove empty subplots if any
    for j in range(len(available_columns), len(axes_flat)):
        fig.delaxes(axes_flat[j])

    plt.tight_layout()
    return fig

def plot_detailed_categorical(data, column, figsize=(12, 6)):
    """
    Create a detailed bar plot for a specific categorical variable
    """
    sns.set_style("whitegrid")

    try:
        plt.figure(figsize=figsize)

        # Convert to string if necessary
        if data[column].dtype != 'object' and not pd.api.types.is_categorical_dtype(data[column]):
            series = data[column].astype(str)
        else:
            series = data[column]

        value_counts = series.value_counts()
        total = len(data)

        # Create bar plot
        ax = sns.barplot(x=value_counts.index, y=value_counts.values, palette='viridis')

        # Add title and labels
        plt.title(f'Distribution of {column}\n(n={series.notna().sum():,}, Missing={series.isna().sum():,})')
        plt.xlabel(column)
        plt.ylabel('Count')

        # Add percentage labels
        for i, v in enumerate(value_counts.values):
            percentage = (v/total) * 100
            if percentage >= 1:  # Only show percentages >= 1%
                ax.text(i, v, f'{percentage:.1f}%', ha='center', va='bottom')

        # Rotate x-labels if needed
        if len(value_counts.index) > 5 or max(len(str(x)) for x in value_counts.index) > 10:
            plt.xticks(rotation=45, ha='right')

        plt.tight_layout()
        return plt.gcf()

    except Exception as e:
        plt.close()
        print(f"Error plotting {column}: {str(e)}")
        return None


def analyze_categorical_anomalies(data, categorical_threshold=0.01):
    """
    Performs comprehensive anomaly analysis on categorical variables.

    Parameters:
    -----------
    data : pandas.DataFrame
        Input dataset to analyze
    categorical_threshold : float
        Minimum frequency threshold for rare categories (default: 0.01 or 1%)

    Returns:
    --------
    dict
        Dictionary containing analysis results
    """
    results = {
        'categorical_anomalies': {},
        'summary': {'total_rare_categories': 0}
    }

    def analyze_categorical_column(series):
        """Helper function to analyze categorical columns"""
        # Remove null values for analysis
        series = series.dropna()

        # Calculate value frequencies
        value_counts = series.value_counts(normalize=True)
        rare_categories = value_counts[value_counts < categorical_threshold]

        # Calculate absolute counts for reporting
        absolute_counts = series.value_counts()

        return {
            'rare_categories': rare_categories.index.tolist(),
            'rare_frequencies': rare_categories.values.tolist(),
            'rare_absolute_counts': [absolute_counts[cat] for cat in rare_categories.index],
            'total_rare_categories': len(rare_categories),
            'most_common': value_counts.index[0],
            'most_common_freq': value_counts.iloc[0],
            'unique_categories': len(value_counts),
            'missing_values': series.isnull().sum(),
            'total_records': len(series)
        }

    # Analyze categorical columns
    categorical_cols = data.select_dtypes(include=['object', 'category']).columns

    for col in categorical_cols:
        if data[col].notna().any():  # Only analyze if column has non-null values
            results['categorical_anomalies'][col] = analyze_categorical_column(data[col])
            results['summary']['total_rare_categories'] += results['categorical_anomalies'][col]['total_rare_categories']

    return results

def plot_category_distributions(data, results, column, figsize=(12, 6)):
    """
    Creates visualization for category distribution with rare categories highlighted.

    Parameters:
    -----------
    data : pandas.DataFrame
        Input dataset
    results : dict
        Results from analyze_categorical_anomalies function
    column : str
        Column name to visualize
    figsize : tuple
        Figure size for plot
    """
    if column not in results['categorical_anomalies']:
        print(f"No analysis results found for column: {column}")
        return None

    plt.figure(figsize=figsize)

    # Get column data
    series = data[column].dropna()
    value_counts = series.value_counts()

    # Create color map (rare categories in red, others in blue)
    rare_cats = set(results['categorical_anomalies'][column]['rare_categories'])
    colors = ['red' if cat in rare_cats else 'blue' for cat in value_counts.index]

    # Create bar plot
    ax = sns.barplot(x=value_counts.index, y=value_counts.values, palette=colors)

    # Add title and labels
    stats = results['categorical_anomalies'][column]
    title = (f'Distribution of {column}\n'
             f'Total Categories: {stats["unique_categories"]}, '
             f'Rare Categories: {stats["total_rare_categories"]}\n'
             f'Missing Values: {stats["missing_values"]} '
             f'({stats["missing_values"]/stats["total_records"]*100:.1f}%)')

    plt.title(title)
    plt.xlabel(column)
    plt.ylabel('Count')

    # Add percentage labels
    total = len(series)
    for i, v in enumerate(value_counts.values):
        percentage = (v/total) * 100
        if percentage >= 1:  # Only show percentages >= 1%
            ax.text(i, v, f'{percentage:.1f}%', ha='center', va='bottom')

    # Rotate x-labels if needed
    if len(value_counts.index) > 5 or max(len(str(x)) for x in value_counts.index) > 10:
        plt.xticks(rotation=45, ha='right')

    plt.tight_layout()
    return plt.gcf()

def generate_categorical_analysis_report(data, categorical_threshold=0.01):
    """
    Generates a comprehensive categorical anomaly analysis report.

    Parameters:
    -----------
    data : pandas.DataFrame
        Input dataset to analyze
    categorical_threshold : float
        Minimum frequency threshold for rare categories

    Returns:
    --------
    tuple
        (analysis_results, summary_text)
    """
    # Perform analysis
    results = analyze_categorical_anomalies(data, categorical_threshold)

    # Generate summary report
    summary = []
    summary.append("=== Categorical Anomaly Analysis Report ===\n")

    # Overall summary
    summary.append(f"Total number of rare categories detected: {results['summary']['total_rare_categories']}")
    summary.append(f"Analysis threshold: Categories occurring in less than {categorical_threshold*100}% of records\n")

    # Detailed categorical analysis
    summary.append("=== Categorical Variables Analysis ===")
    for col, stats in results['categorical_anomalies'].items():
        summary.append(f"\n{col}:")
        summary.append(f"- Total unique categories: {stats['unique_categories']}")
        summary.append(f"- Missing values: {stats['missing_values']} ({stats['missing_values']/stats['total_records']*100:.1f}%)")
        summary.append(f"- Most common value: {stats['most_common']} ({stats['most_common_freq']*100:.1f}%)")
        summary.append(f"- Number of rare categories: {stats['total_rare_categories']}")

        if stats['total_rare_categories'] > 0:
            summary.append("- Rare categories (top 5):")
            for cat, freq, count in zip(
                stats['rare_categories'][:5],
                stats['rare_frequencies'][:5],
                stats['rare_absolute_counts'][:5]
            ):
                summary.append(f"  * {cat}: {count} occurrences ({freq*100:.2f}%)")

    return results, "\n".join(summary)


