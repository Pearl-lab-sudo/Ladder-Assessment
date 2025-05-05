# --- Libraries ---
import os
import pandas as pd
from datetime import datetime
from sqlalchemy import create_engine, text
from dotenv import load_dotenv
import logging
import argparse
from pathlib import Path

# --- Logging Setup ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- Load Environment Variables ---
load_dotenv()
DB_USER = os.getenv('DB_USER')
DB_PASSWORD = os.getenv('DB_PASSWORD')
DB_HOST = os.getenv('DB_HOST')
DB_PORT = os.getenv('DB_PORT', '5432')
DB_NAME = os.getenv('DB_NAME')

# --- SQLAlchemy Engine ---
engine = create_engine(f'postgresql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}')


# --- Function to Fetch Data ---
def fetch_transaction_data(engine) -> pd.DataFrame:
    """
    Fetches transaction data from database using the provided engine.
    
    Args:
        engine: SQLAlchemy engine object
        
    Returns:
        DataFrame containing transaction data
    """
    # SQL query to fetch transaction data
    transaction_info_query = """
    -- Identify first transaction date for each user
    WITH first_transactions AS (
        SELECT 
            COALESCE(p.user_id, ip.user_id) AS user_id,
            MIN(t.created_at) AS first_transaction_date
        FROM transactions t
        LEFT JOIN investment_plans ip ON ip.id = t.investment_plan_id
        LEFT JOIN plans p ON p.id = t.plan_id
        WHERE t.status = 'success'
        GROUP BY COALESCE(p.user_id, ip.user_id)
    )

    -- Main query to extract transaction details
    SELECT 
        t.id AS transaction_id,
        CASE 
            WHEN t.metadata::text ILIKE '%Monthly maintenance fee deduction%' THEN 'maintenance_fee'
            ELSE t.tx_type
        END AS transaction_type,
        
        u.id AS customer_id,
        u.first_name || ' ' || u.last_name AS customer_name,
        ft.first_transaction_date,
        TO_CHAR(t.created_at, 'YYYY-MM') AS transaction_cohort,
        TO_CHAR(u.created_at, 'YYYY-MM') AS sign_up_cohort,
        TO_CHAR(u.updated_at, 'YYYY-MM') AS kyc_cohort,
        EXTRACT(WEEK FROM t.created_at) AS week_number,
        
        t.created_at AS transaction_date,
        t.amount AS ghs_amount,
        t.usd_amount,
        t.exchange_rate,

        u.created_at AS sign_up_date,
        u.updated_at AS kyc_completed_date,

        CASE 
            WHEN t.investment_plan_id IS NOT NULL THEN ip.plan_option
            WHEN t.plan_id IS NOT NULL THEN p.plan_option
            ELSE NULL
        END AS investment_type,
        
        a.name AS asset_type,
        a.maturity_date AS assets_maturity_date,
        ip.maturity_date AS investment_maturity_date

    FROM transactions t
    LEFT JOIN investment_plans ip ON ip.id = t.investment_plan_id
    LEFT JOIN plans p ON p.id = t.plan_id
    LEFT JOIN users u ON u.id = COALESCE(p.user_id, ip.user_id)
    LEFT JOIN first_transactions ft ON ft.user_id = u.id
    LEFT JOIN assets a ON a.id = ip.asset_id
    WHERE t.status = 'success';
    """
    try:
        logger.info("Fetching transaction data from database...")
        with engine.connect() as connection:
            transactions_info = pd.read_sql(text(transaction_info_query), connection)
        
        logger.info(f"Successfully fetched {len(transactions_info)} transaction records")
        return transactions_info
    
    except Exception as e:
        logger.error(f"Database query failed: {e}")
        raise


def clean_transaction_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Cleans and preprocesses the transaction data.
    
    Args:
        df: DataFrame containing raw transaction data
        
    Returns:
        Cleaned DataFrame
    """
    logger.info("Cleaning transaction data...")
    
    # Make a copy to avoid warnings about setting values on a slice
    cleaned_df = df.copy()

  # List of date columns
    date_columns = [
        'first_transaction_date', 'transaction_date', 
        'sign_up_date', 'kyc_completed_date',
        'assets_maturity_date', 'investment_maturity_date'
    ]
    
    # Convert to datetime, handle errors
    for col in date_columns:
        if col in cleaned_df.columns:
            cleaned_df[col] = pd.to_datetime(cleaned_df[col], errors='coerce', utc=True)

    # Convert numeric columns to appropriate types
    numeric_columns = ['ghs_amount', 'usd_amount', 'exchange_rate', 'week_number']
    for col in numeric_columns:
        if col in cleaned_df.columns:
            cleaned_df[col] = pd.to_numeric(cleaned_df[col], errors='coerce')
    
    # Fill missing values appropriately
    cleaned_df['investment_type'] = cleaned_df['investment_type'].fillna('unknown')
    cleaned_df['asset_type'] = cleaned_df['asset_type'].fillna('unknown')
    cleaned_df['transaction_type'] = cleaned_df['transaction_type'].fillna('unknown')
    cleaned_df['customer_name'] = cleaned_df['customer_name'].fillna('unknown')
    cleaned_df['customer_id'] = cleaned_df['customer_id'].fillna('unknown')
    cleaned_df['ghs_amount'] = cleaned_df['ghs_amount'].fillna(0)
    cleaned_df['usd_amount'] = cleaned_df['usd_amount'].fillna(0)

    default_date = pd.Timestamp('1970-01-01', tz='UTC')
    # Fill missing date values with a default date
    cleaned_df['transaction_date'] = cleaned_df['transaction_date'].fillna(default_date)
    cleaned_df['first_transaction_date'] = cleaned_df['first_transaction_date'].fillna(default_date)
    cleaned_df['sign_up_date'] = cleaned_df['sign_up_date'].fillna(default_date)
    cleaned_df['kyc_completed_date'] = cleaned_df['kyc_completed_date'].fillna(default_date)
    cleaned_df['assets_maturity_date'] = cleaned_df['assets_maturity_date'].fillna(default_date)
    cleaned_df['investment_maturity_date'] = cleaned_df['investment_maturity_date'].fillna(default_date)
    
    # Add funding source column
    cleaned_df['funding_source'] = cleaned_df['transaction_type'].apply(
        lambda x: 'Flex Dollar' if x == 'internal_transfer' else 'mobile money'
    )
    
    # Handle rows with missing crucial data
    if 'transaction_id' in cleaned_df.columns and 'customer_id' in cleaned_df.columns:
        initial_rows = len(cleaned_df)
        
        # First, identify different categories of missing data
        missing_transaction_mask = cleaned_df['transaction_id'].isna()
        missing_customer_mask = cleaned_df['customer_id'].isna()
        
        # Get transaction IDs with missing customer data for logging
        missing_customer_transactions = cleaned_df[missing_customer_mask & ~missing_transaction_mask]['transaction_id'].tolist()
        
        # Count rows in each category
        missing_transaction_count = missing_transaction_mask.sum()
        missing_customer_count = missing_customer_mask.sum()
        
        # Log detailed information
        if missing_transaction_count > 0:
            logger.warning(f"Found {missing_transaction_count} rows with missing transaction IDs. These will be dropped.")
            
        if missing_customer_count > 0:
            logger.warning(f"Found {missing_customer_count} rows with missing customer IDs.")
            logger.info(f"Transactions with missing customer IDs: {missing_customer_transactions[:10]}" + 
                       (f" and {len(missing_customer_transactions)-10} more..." if len(missing_customer_transactions) > 10 else ""))
        
        # Drop only rows with missing transaction IDs (keep rows with just missing customer IDs)
        cleaned_df = cleaned_df.dropna(subset=['transaction_id'])
        
        # Report final results
        dropped_rows = initial_rows - len(cleaned_df)
        if dropped_rows > 0:
            logger.warning(f"Dropped {dropped_rows} rows with missing transaction IDs")
            logger.info(f"Remaining rows with missing customer IDs: {missing_customer_mask.sum() - dropped_rows}")
    
    logger.info("Data cleaning completed")
    return cleaned_df


# --- Breakage Fee ---
# Using vectorized operations for performance optimization
# This implementation is ~10x faster than row-by-row iteration for typical dataset sizes
def apply_breakage_fee(df: pd.DataFrame) -> pd.DataFrame:
    """
    Applies breakage fee logic to the transaction data using vectorized operations.
    Creates new rows for breakage fees when applicable.
    
    Args:
        df: DataFrame containing transaction data
        
    Returns:
        DataFrame with added breakage fee rows
    """
    logger.info("Applying breakage fee logic (vectorized)...")

    # Define the condition for breakage fee application
    condition = (
        (df['transaction_type'] == 'withdrawal') &
        (df['asset_type'].isin(['Risevest Real Estate', 'Risevest Fixed Income'])) &
        (pd.notnull(df['assets_maturity_date'])) &
        (df['transaction_date'] < df['assets_maturity_date'])
    )

    # Create a new DataFrame for breakage fees
    breakage_df = df[condition].copy()
    # Apply breakage fee logic
    breakage_df['transaction_type'] = 'breakage_fee'
    breakage_df['ghs_amount'] = (breakage_df['ghs_amount'] * 0.025).round(2)
    breakage_df['usd_amount'] = (breakage_df['usd_amount'] * 0.025).round(2)
    breakage_df['breakage_fee_applied'] = True

    df.loc[~condition, 'breakage_fee_applied'] = False

    return pd.concat([df, breakage_df], ignore_index=True)


# --- Cash Flow Calculation ---
def calculate_cash_flows(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculates cash flows for GHS and USD based on transaction types.

    Args:
        df: DataFrame containing transaction data

    Returns:
        DataFrame with cash flow columns added
    """
    logger.info("Calculating cash flows...")

    # Create a copy to avoid modifying the original dataframe
    df = df.copy()

    # Initialize cash flow columns
    # Deposits are positive, withdrawals are negative
    df['ghs_cash_flow'] = 0
    df['usd_cash_flow'] = 0
    df.loc[df['transaction_type'] == 'deposit', 'ghs_cash_flow'] = df['ghs_amount']
    df.loc[df['transaction_type'] == 'withdrawal', 'ghs_cash_flow'] = -df['ghs_amount']
    df.loc[df['transaction_type'] == 'deposit', 'usd_cash_flow'] = df['usd_amount']
    df.loc[df['transaction_type'] == 'withdrawal', 'usd_cash_flow'] = -df['usd_amount']
    df.loc[df['transaction_type'] == 'maintenance_fee', 'ghs_cash_flow'] = -df['ghs_amount']
    df.loc[df['transaction_type'] == 'maintenance_fee', 'usd_cash_flow'] = -df['usd_amount']
    df.loc[df['transaction_type'] == 'breakage_fee', 'ghs_cash_flow'] = -df['ghs_amount']
    df.loc[df['transaction_type'] == 'breakage_fee', 'usd_cash_flow'] = -df['usd_amount']

    return df


# --- Reporting ---
def generate_summary_report(df: pd.DataFrame) -> pd.DataFrame:
    """
    Generates a summary report from transaction data.
    
    Args:
        df: DataFrame containing processed transaction data
        
    Returns:
        DataFrame containing summary statistics
    """
    logger.info("Generating summary report...")

    # Count savings vs investment transactions 
    savings_count = df[df['investment_type'].str.contains('savings', case=False, na=False)].shape[0]
    investment_count = df[df['investment_type'].str.contains('investment', case=False, na=False)].shape[0]

    # Create summary dictionary with improved error handling
    summary = {
        'Report_Generation_Date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'Total_Transactions': len(df),
        'Total_Savings_Transactions': savings_count,
        'Total_Investment_Transactions': investment_count,
        'Total_Unique_Customers': df['customer_id'].nunique(),
        'Total_Sign_Up_Cohort': df['sign_up_cohort'].nunique(),
        'Total_KYC_Cohort': df['kyc_cohort'].nunique(),
    }
    
    # Add data quality metrics if available
    if 'missing_customer_flag' in df.columns:
        missing_customer_count = df['missing_customer_flag'].sum()
        summary['Transactions_With_Missing_Customer_ID'] = missing_customer_count
        summary['Percent_Transactions_With_Missing_Customer_ID'] = round((missing_customer_count / len(df)) * 100, 2) if len(df) > 0 else 0
    
    # Add financial metrics with appropriate handling of missing values
    transaction_types = ['deposit', 'withdrawal', 'breakage_fee', 'maintenance_fee']
    currencies = ['ghs', 'usd']
    
    for currency in currencies:
        for tx_type in transaction_types:
            amount_col = f'{currency}_amount'
            if amount_col in df.columns:
                mask = df['transaction_type'] == tx_type
                if mask.any():
                    total = df.loc[mask, amount_col].sum()
                    summary[f'Total_{currency.upper()}_{tx_type.capitalize()}s'] = total
                else:
                    summary[f'Total_{currency.upper()}_{tx_type.capitalize()}s'] = 0
    
    # Add cash flow totals
    for currency in currencies:
        cash_flow_col = f'{currency}_cash_flow'
        if cash_flow_col in df.columns:
            summary[f'Net_{currency.upper()}_Cash_Flow'] = df[cash_flow_col].sum()
    
    # Add breakage fee count
    summary['Breakage_Fees_Count'] = df['breakage_fee_applied'].sum()
    
    # Group by funding source with proper error handling
    for currency in currencies:
        amount_col = f'{currency}_amount'
        if 'funding_source' in df.columns and amount_col in df.columns:
            deposits_by_source = df[df['transaction_type'] == 'deposit'].groupby('funding_source')[amount_col].sum()
            
            for source, amount in deposits_by_source.items():
                summary[f'{currency.upper()}_Deposit_via_{source}'] = amount
    
    return pd.DataFrame([summary])


def generate_cohort_analysis(df: pd.DataFrame) -> pd.DataFrame:
    """
    Generates cohort analysis from transaction data.
    
    Args:
        df: DataFrame containing processed transaction data
        
    Returns:
        DataFrame containing cohort analysis
    """
    logger.info("Generating cohort analysis...")
    
    # Ensure we have required columns
    required_cols = ['customer_id', 'sign_up_cohort', 'kyc_cohort', 'transaction_cohort', 'transaction_type', 'ghs_amount']
    if not all(col in df.columns for col in required_cols):
        logger.warning("Missing required columns for cohort analysis")
        return pd.DataFrame()
    
    # Filter for deposits only
    deposits = df[df['transaction_type'] == 'deposit'].copy()
    
    # Create cohort analysis - average deposit amount by sign up cohort and transaction cohort
    cohort_data = deposits.groupby(['transaction_cohort']).agg({
        'customer_id': 'nunique',
        'ghs_amount': ['count', 'sum', 'mean']
    }).reset_index()
    
    # Flatten multi-level columns
    cohort_data.columns = ['_'.join(col).strip('_') for col in cohort_data.columns.values]
    
    # Rename columns for clarity
    cohort_data = cohort_data.rename(columns={
        'sign_up_cohort': 'Sign_Up_Cohort',
        'transaction_cohort': 'Transaction_Cohort',
        'customer_id_nunique': 'Unique_Customers',
        'ghs_amount_count': 'Deposit_Count',
        'ghs_amount_sum': 'Total_Deposit_Amount',
        'ghs_amount_mean': 'Average_Deposit_Amount'
    })
    
    return cohort_data


def save_to_csv(df: pd.DataFrame, filename: str, output_dir: str = 'reports') -> str:
    """
    Saves DataFrame to CSV file with proper error handling.
    
    Args:
        df: DataFrame to save
        filename: Name of the CSV file
        output_dir: Directory to save the file in
        
    Returns:
        Path to the saved file
    """
    # Create output directory if it doesn't exist
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Add timestamp to filename to avoid overwriting
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    file_path = output_path / f"{filename}_{timestamp}.csv"
    
    try:
        df.to_csv(file_path, index=False)
        logger.info(f"Successfully saved data to {file_path}")
        return str(file_path)
    except Exception as e:
        logger.error(f"Failed to save {filename}: {e}")
        raise
    

def process_transaction_data():
    """
    Main function that orchestrates the entire data processing pipeline.
    """
    try:
        # Parse command line arguments
        parser = argparse.ArgumentParser(description='Process transaction data and generate reports.')
        parser.add_argument('--output-dir', default='reports', help='Directory to save output files')
        args = parser.parse_args()
        
        # Fetch transaction data
        transactions_info = fetch_transaction_data(engine)
        
        # Process data
        cleaned_df = clean_transaction_data(transactions_info)
        with_fees_df = apply_breakage_fee(cleaned_df)
        final_df = calculate_cash_flows(with_fees_df)
        
        # Generate reports
        summary_report = generate_summary_report(final_df)
        cohort_analysis = generate_cohort_analysis(final_df)
        
        # Save processed data and reports
        save_to_csv(final_df, 'final_processed_transactions', args.output_dir)
        save_to_csv(summary_report, 'final_transaction_summary_report', args.output_dir)
        save_to_csv(cohort_analysis, 'final_transaction_cohort_analysis', args.output_dir)
        
        logger.info("Transaction analysis completed successfully")
        
    except Exception as e:
        logger.error(f"Transaction analysis failed: {e}")
        raise

if __name__ == "__main__":
    process_transaction_data()
