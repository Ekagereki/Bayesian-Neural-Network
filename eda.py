import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.stats import chi2_contingency, ttest_ind, f_oneway, pointbiserialr, norm
import warnings
from sklearn.preprocessing import LabelEncoder

warnings.filterwarnings("ignore")


def loading_data():
    url = "https://osf.io/download/nqhtw/"

    try:
        df = pd.read_csv(url)
        print("Data loaded successfully!")
    except Exception as e:
        print(f"An error occurred: {e}")

    return df


def locate_and_remove_outliers(df):

    def locate_item(column_name, item):
        return df[df[column_name] == item].index

    item1_location = locate_item('MSSS_2', 16)
    item2_location = locate_item('GAD_3', 11)
    item3_location = locate_item('EPOCH_Optimism_2', 23)
    item4_location = locate_item('PCS_Academic_04', -7)
    item5_location = locate_item('PCS_Academic_08', 7)

    df.loc[item1_location, 'MSSS_2'] = pd.NA
    df.loc[item2_location, 'GAD_3'] = pd.NA
    df.loc[item3_location, 'EPOCH_Optimism_2'] = pd.NA
    df.loc[item4_location, 'PCS_Academic_04'] = pd.NA
    df.loc[item5_location, 'PCS_Academic_08'] = pd.NA

    return df


def clean_dataframe(df):
    df.loc[df['Parents_Dead'] == 'Both', 'Num_parents_dead'] = 2
    df['Fathers_Education'] = df['Fathers_Education'].replace({"Primary school": "primary", "Secondary school": "secondary"})
    df['Mothers_Education'] = df['Mothers_Education'].replace({"Primary school": "primary", "Secondary school": "secondary"})
    df['Co_Curricular'] = df['Co_Curricular'].replace({"Not involved at all": "Not involved"})
    # Drop rows where all data points are NA
    df_clean = df.iloc[:, 3:].dropna(how='all').copy()

    # Iterate over each column; impute based on data type
    for col in df_clean.columns:
        if df_clean[col].dtype in [np.float64, np.int64]:
            # Compute mean for numeric column for non-missing values
            mean_val = df_clean[col].mean().round(0)
            df_clean[col].fillna(mean_val, inplace=True)
        else:
            # For non-numeric, use mode
            if not df_clean[col].mode().empty:
                mode_val = df_clean[col].mode()[0]
                df_clean[col].fillna(mode_val, inplace=True)
            else:
                # If mode is empty, leave as is
                df_clean[col].fillna('Missing', inplace=True)
    '''
    adding columns msss_total, gad_total, phq_total, gratitude_total, ucla_total,
    pcs_academic_total, epoch_optimism_total, epoch_happiness_total & phq_total
    '''

    df_clean['MSSS_Total'] = df_clean.iloc[:, 0:12].sum(axis=1)
    df_clean['EPOCH_Total'] = df_clean.iloc[:, 29:37].sum(axis=1)
    df_clean['PCS_Academic_Total'] = df_clean.iloc[:, 37:45].sum(axis=1)
    df_clean['UCLA_Total'] = df_clean.iloc[:, 45:53].sum(axis=1)
    df_clean['Gratitude_Total'] = df_clean.iloc[:, 53:59].sum(axis=1)

    df_clean["PHQ_Total"] = df_clean["PHQ_Total"].astype(int)
    df_clean["Depression_Status"] = df_clean["PHQ_Total"].apply(lambda x:
                                                                "Depressed" if x > 10 else "Not depressed")

    new_df = df_clean[['Age', 'Form', 'Gender', 'Financial_Status', 'Home', 'Siblings', 'Religion', 'Num_parents_dead',
                       'Fathers_Education', 'Mothers_Education', 'Co_Curricular', 'Sports', 'Percieved_Academic_Abilities',
                       'GAD_Total', 'MSSS_Total','EPOCH_Total', 'PCS_Academic_Total', 'UCLA_Total', 'Gratitude_Total',
                       'PHQ_Total', 'Depression_Status']]

    return new_df

def import_data():
    df = loading_data()
    df = locate_and_remove_outliers(df)
    df = clean_dataframe(df)

    return df

def plot_categorical(df):
    fig, axes = plt.subplots(2, 4, figsize=(18, 12))
    plt.subplots_adjust(left=0.05, right=0.95, bottom=0.05, top=0.95)
    plt.title("Categorical Plots")

    gender_counts = df['Gender'].value_counts()
    axes[0,2].pie(gender_counts.values, labels=gender_counts.index, autopct='%1.1f%%')
    axes[0,2].set_title('Gender Distribution')

    depression_counts = df['Depression_Status'].value_counts()
    axes[0,1].pie(depression_counts.values, labels=depression_counts.index, autopct='%1.1f%%')
    axes[0,1].set_title('Depression Status')

    curricular_counts = df['Co_Curricular'].value_counts()
    axes[0,0].pie(curricular_counts.values, labels=curricular_counts.index, autopct='%1.1f%%')
    axes[0,0].set_title('Co-Curricular Distribution')

    financial_counts = df['Financial_Status'].value_counts()
    axes[1,2].bar(financial_counts.index, financial_counts.values)
    axes[1,2].set_title('Financial Status')
    axes[1,2].tick_params(axis='x', rotation=19)

    fathers_education_counts = df['Fathers_Education'].value_counts()
    axes[1,1].bar(fathers_education_counts.index, fathers_education_counts.values)
    axes[1,1].set_title('Fathers Education Distribution')
    axes[1,1].tick_params(axis='x', rotation=10)

    mothers_education_counts = df['Mothers_Education'].value_counts()
    axes[1,0].bar(fathers_education_counts.index, fathers_education_counts.values)
    axes[1,0].set_title('Mothers Education Distribution')
    axes[1,0].tick_params(axis='x', rotation=10)

    home_counts = df['Home'].value_counts()
    axes[0,3].pie(home_counts.values, labels=home_counts.index, autopct='%1.1f%%')
    axes[0,3].set_title('Home Distribution')

    form_counts = df['Form'].value_counts()
    axes[1, 3].bar(form_counts.index, form_counts.values)
    axes[1, 3].set_title('Form Distribution')
    axes[1, 3].tick_params(axis='x', rotation=10)

    plt.tight_layout()
    plt.show()

def plot_continuous(df):
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))

    axes[0,0].hist(df['MSSS_Total'], bins=10, edgecolor='black', alpha=0.5)
    axes[0,0].axvline(df['MSSS_Total'].mean(), color='red', linestyle='--', label=f'Mean: {df.MSSS_Total.mean():.2f}')
    axes[0,0].set_title('MSSS Total')
    axes[0,0].set_xlabel('MSSS Total')
    axes[0,0].set_ylabel('Frequency')
    axes[0,0].legend()

    axes[0,1].hist(df['GAD_Total'], bins=10, edgecolor='black', alpha=0.5)
    axes[0,1].axvline(df['GAD_Total'].mean(), color='red', linestyle='--', label=f'Mean: {df.GAD_Total.mean():.2f}')
    axes[0,1].set_title('GAD Total')
    axes[0,1].set_xlabel('GAD Total')
    axes[0,1].set_ylabel('Frequency')
    axes[0,1].legend()

    axes[0,2].hist(df['PHQ_Total'], bins=10, edgecolor='black', alpha=0.5)
    axes[0,2].axvline(df['PHQ_Total'].mean(), color='red', linestyle='--', label=f'Mean: {df.PHQ_Total.mean():.2f}')
    axes[0,2].set_title('PHQ Total')
    axes[0,2].set_xlabel('PHQ Total')
    axes[0,2].set_ylabel('Frequency')
    axes[0,2].legend()

    axes[1,0].hist(df['EPOCH_Total'], bins=10, edgecolor='black', alpha=0.5)
    axes[1,0].axvline(df['EPOCH_Total'].mean(), color='red', linestyle='--', label=f'Mean: {df.EPOCH_Total.mean():.2f}')
    axes[1,0].set_title('EPOCH Total')
    axes[1,0].set_xlabel('EPOCH Total')
    axes[1,0].set_ylabel('Frequency')
    axes[1,0].legend()

    axes[1,1].hist(df['UCLA_Total'], bins=10, edgecolor='black', alpha=0.5)
    axes[1,1].axvline(df['UCLA_Total'].mean(), color='red', linestyle='--', label=f'Mean: {df.UCLA_Total.mean():.2f}')
    axes[1,1].set_title('UCLA Total')
    axes[1,1].set_xlabel('UCLA Total')
    axes[1,1].set_ylabel('Frequency')
    axes[1,1].legend()

    mu, std = norm.fit(df['Gratitude_Total'])
    axes[1,2].hist(df['Gratitude_Total'], bins=10, density=True, edgecolor='black', alpha=0.5)
    axes[1,2].axvline(df['Gratitude_Total'].mean(), color='red', linestyle='--', label=f'Mean: {df.Gratitude_Total.mean():.2f}')
    axes[1,2].set_title('Gratitude Total')
    axes[1,2].set_xlabel('Gratitude Total')
    axes[1,2].set_ylabel('Frequency')
    axes[1,2].legend()

    plt.tight_layout()
    plt.show()

def correlation_matrix(df):
    df['GAD'] = df['GAD_Total']
    df['PHQ'] = df['PHQ_Total']
    df['EPOCH'] = df['EPOCH_Total']
    df['UCLA'] = df['UCLA_Total']
    df['MSSS'] = df['MSSS_Total']
    df['PCS_Academic'] = df['PCS_Academic_Total']
    df['Gratitude'] = df['Gratitude_Total']

    plt.figure(figsize=(18, 18))
    mental_health_corr = df[['Age','GAD', 'MSSS','EPOCH', 'PCS_Academic', 'UCLA', 'Gratitude', 'PHQ']].corr()
    sns.heatmap(mental_health_corr, annot=True, cmap='RdYlGn', fmt=".2f", linewidths=.5)
    plt.title('Correlation Matrix')
    plt.show()

def univariate_analysis_categorical(df, outcome_var, categorical_vars):

    results = []

    for var in categorical_vars:
        if var not in df.columns:
            print(f"Warning: {var} not found in dataset")
            continue

        # Clean the variable - remove missing values in both columns
        temp_df = df[[outcome_var, var]].dropna()

        if temp_df[var].nunique() < 2:
            print(f"Warning: {var} has less than 2 unique values after cleaning")
            continue

        # Create contingency table
        contingency_table = pd.crosstab(temp_df[var], temp_df[outcome_var])

        # Perform Chi-squared test
        chi2, p_value, dof, expected = chi2_contingency(contingency_table)

        # Calculate Cramer's V as effect size for Chi-squared
        n = contingency_table.sum().sum()
        min_dim = min(contingency_table.shape) - 1
        cramers_v = np.sqrt(chi2 / (n * min_dim)) if n * min_dim != 0 else 0

        results.append({
            'Variable': var,
            'Test_Type': 'Chi-squared',
            'P_Value': p_value,
            'Significant': p_value < 0.05,
            'Effect_Size': cramers_v,
            'Contingency_Table': contingency_table.to_dict(),
            'Total_N': len(temp_df)
        })

    return pd.DataFrame(results)

def univariate_analysis_continuous(df, outcome_var, continuous_vars):

    results = []
    # Assuming outcome_var is binary/dichotomous
    outcome_values = df[outcome_var].dropna().unique()
    if len(outcome_values) != 2:
        print(f"Warning: Outcome variable '{outcome_var}' is not dichotomous. Skipping continuous variable analysis.")
        return pd.DataFrame(results) # Return empty dataframe

    # Encode the dichotomous outcome variable numerically for point-biserial correlation
    le = LabelEncoder()
    temp_df_outcome_encoded = df[[outcome_var]].dropna()
    temp_df_outcome_encoded['outcome_encoded'] = le.fit_transform(temp_df_outcome_encoded[outcome_var])
    outcome_map = dict(zip(le.classes_, le.transform(le.classes_)))


    for var in continuous_vars:
        if var not in df.columns:
            print(f"Warning: {var} not found in dataset")
            continue

        # Ensure the variable is numeric and handle non-numeric values by coercing to NaN
        temp_var_data = pd.to_numeric(df[var], errors='coerce')
        temp_df = pd.DataFrame({var: temp_var_data}).dropna().join(temp_df_outcome_encoded)
        temp_df.dropna(inplace=True)


        if len(temp_df) < 10:  # Minimum sample size
            print(f"Warning: {var} has insufficient data after cleaning")
            continue

        # Calculate point-biserial correlation
        correlation, p_value = stats.pointbiserialr(temp_df[var], temp_df['outcome_encoded'])


        results.append({
            'Variable': var,
            'Test_Type': 'Point-biserial Correlation',
            'P_Value': p_value,
            'Significant': p_value < 0.05,
            'Effect_Size': abs(correlation),
            'Correlation': correlation,
            'N': len(temp_df)
        })

    return pd.DataFrame(results)


def plot_significant_factors(df, results_df, outcome_var, plot_type='categorical'):
    """
    Create visualizations for significant factors against a categorical outcome
    """
    significant_results = results_df[results_df['Significant']].copy()

    if len(significant_results) == 0:
        print("No significant factors found for plotting")
        return

    n_plots = min(6, len(significant_results))  # Limit to 6 plots
    n_cols = 2
    n_rows = (n_plots + n_cols - 1) // n_cols  # Calculate rows needed

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5 * n_rows))
    axes = axes.flatten()

    for i, (idx, row) in enumerate(significant_results.head(n_plots).iterrows()):
        if i >= len(axes):
            break

        if plot_type == 'categorical':
            # Create count plot for categorical variables
            sns.countplot(data=df, x=row['Variable'], hue=outcome_var, ax=axes[i])
            axes[i].set_title(f'{row["Variable"]} vs {outcome_var}\n(p={row["P_Value"]:.3f})')
            axes[i].set_xlabel(row['Variable'])
            axes[i].set_ylabel('Count')
            axes[i].tick_params(axis='x', rotation=45)
            axes[i].legend(title=outcome_var)

        else:  # continuous
            # Create boxplot plot for continuous variables against categorical outcome
            sns.boxplot(data=df, x=outcome_var, y=row['Variable'], ax=axes[i])
            axes[i].set_title(f'{row["Variable"]} vs {outcome_var}\n(r={row["Correlation"]:.3f}, p={row["P_Value"]:.3f})')
            axes[i].set_xlabel(outcome_var)
            axes[i].set_ylabel(row['Variable'])


    # Hide unused subplots
    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)

    plt.tight_layout()
    plt.suptitle(f'Significant Factors for {outcome_var}', y=1.02, fontsize=16)
    plt.show()


def comprehensive_univariate_analysis(df, outcome_vars):

    categorical_factors = [
        'Gender', 'Financial_Status', 'Home', 'Religion', 'Co_Curricular', 'Sports', 'Form',
        'Fathers_Education', 'Mothers_Education', 'Percieved_Academic_Abilities', 'Num_parents_dead'
    ]

    continuous_factors = [
        'Age', 'Siblings', 'GAD_Total', 'MSSS_Total', 'EPOCH_Total', 'PCS_Academic_Total', 'UCLA_Total', 'Gratitude_Total'
    ]

    all_results = {}

    for outcome_var in outcome_vars:
        print(f"\n{'='*60}")
        print(f"UNIVARIATE ANALYSIS FOR: {outcome_var}")
        print(f"{'='*60}")

        # Analyze categorical variables
        print(f"\nCategorical Variables Analysis:")
        cat_results = univariate_analysis_categorical(df, outcome_var, categorical_factors)
        if not cat_results.empty:
            cat_results = cat_results.sort_values('P_Value')
            print(cat_results[['Variable', 'Test_Type', 'P_Value', 'Significant', 'Effect_Size', 'Total_N']])

            # Plot significant categorical factors
            plot_significant_factors(df, cat_results, outcome_var, 'categorical')


        # Analyze continuous variables
        print(f"\nContinuous Variables Analysis:")
        cont_results = univariate_analysis_continuous(df, outcome_var, continuous_factors)
        if not cont_results.empty:
            cont_results = cont_results.sort_values('P_Value')
            print(cont_results[['Variable', 'Test_Type', 'P_Value', 'Significant', 'Effect_Size', 'Correlation', 'N']])

            # Plot significant continuous factors
            plot_significant_factors(df, cont_results, outcome_var, 'continuous')


        # Combine results
        all_results[outcome_var] = {
            'categorical': cat_results,
            'continuous': cont_results
        }

    return all_results


def export_results(all_results, filename='univariate_analysis_results.xlsx'):

    with pd.ExcelWriter(filename, engine='openpyxl') as writer:
        for outcome_var, results in all_results.items():
            # Categorical results
            if not results['categorical'].empty:
                results['categorical']['Contingency_Table'] = results['categorical']['Contingency_Table'].apply(str)
                results['categorical'].to_excel(writer, sheet_name=f'{outcome_var}_categorical', index=False)

            # Continuous results
            if not results['continuous'].empty:
                results['continuous'].to_excel(writer, sheet_name=f'{outcome_var}_continuous', index=False)


    print(f"\nResults exported to {filename}")


def main():
    df = import_data()

    outcome_vars = ['Depression_Status']

    plot_categorical(df)
    plot_continuous(df)
    correlation_matrix(df)

    # Perform comprehensive univariate analysis
    all_results = comprehensive_univariate_analysis(df, outcome_vars)

    # Export results
    export_results(all_results)

    # Print summary of significant findings
    print(f"\n{'='*60}")
    print("SUMMARY OF SIGNIFICANT FINDINGS (p < 0.05)")
    print(f"{'='*60}")

    for outcome_var, results in all_results.items():
        print(f"\nFor {outcome_var}:")

        # Significant categorical factors
        sig_cat = results['categorical'][results['categorical']['Significant']]
        if not sig_cat.empty:
            print("  Significant Categorical Factors:")
            for _, row in sig_cat.iterrows():
                print(f"    - {row['Variable']}: p={row['P_Value']:.4f}, Effect Size={row['Effect_Size']:.3f}")

        # Significant continuous factors
        sig_cont = results['continuous'][results['continuous']['Significant']]
        if not sig_cont.empty:
            print("  Significant Continuous Factors:")
            for _, row in sig_cont.iterrows():
                print(f"    - {row['Variable']}: p={row['P_Value']:.4f}, r={row['Correlation']:.3f}")

if __name__ == "__main__":
    main()
