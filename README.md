# Ladder Assessment
## Overview
**This project is a data analytics assessment for Ladder. It involves analyzing transactional data from a database, processing and cleaning records, generating summary insights, and performing cohort analysis. All outputs were generated using Python (with Pandas, SQLAlchemy, dotenv, argparse) and PostgreSQL.**

## Assessment Context
The task involved analyzing customer transaction data from a specified database to generate insights for the product and finance teams. Requirements included extracting only successful transactions, applying business rules (e.g., breakage fees, metadata handling), calculating cash flow metrics, and exporting a final proccessed csv file using Python and SQL.

## Project Objectives From Instructions
- Write a comprehensive SQL query to extract relevant transaction data, including these fields: `Date of First Transaction`, `Transaction Cohort`, `USD Amount`, `Week Number of Transaction`, `GHS Amount`, `Sign-Up Date`, `Exchange Rate`, `KYC Completed Date`, `Sign-Up Cohort`, `KYC Cohort`, `Investment Type` (eg. "savings", "investment"), `Maturity Date`, `Transaction Date`, `Customer Name`, `Customer ID`, `Asset Type`, `Transaction Type` & `Transaction ID`.

- Filter the dataset to include only successful transactions and implement conditional logic to reclassify transactions with the metadata description `“Monthly maintenance fee deduction”` as `maintenance_fee`.

- Load the extracted data into Python and process it to prepare it for further analysis.

- Implement breakage fee logic by identifying early withdrawals on selected asset types (e.g., "Risevest Real Estate", "Risevest Fixed Income") and applying a *2.5% fee*. Duplicate applicable transactions and adjust transaction amounts and types to reflect the fee.

- Calculate cash flow metrics for both USD and GHS, including:

   a. Cash Flow (usd_cash_flow, ghs_cash_flow): Positive for deposits, negative for withdrawals.

   b. Deposits (usd_deposit, ghs_deposit)

   c. Withdrawals (usd_withdrawal, ghs_withdrawal)

- Add a new funding_source column:

   > Assign "flex dollar" if the transaction type is internal_transfer

   > Assign "mobile money" for all other transaction types

- Export the final processed data to a CSV file for submission along with the SQL file (.sql) containing query and Python script (.py) for the data processing and analysis.

## Assumptions Made Based On Available Data 
  > NB: Upon reviewing the database, I noted that only 7 tables out of 70 contained records: users, investment_plans, categories, plans, transactions, assets and account_balance.
  
- `Transaction Type`: Assumed the tx_type field in the transactions table represented the transaction type, as there was no column explicitly named “transaction_type”.

- `Customer Name`: Derived the customer name by concatenating first_name and last_name columns from the users table.

- `Customer ID`: Used the user_id field from users table as the customer ID.

- `KYC Completed Date`: Noted the "kyc_completed: true" from the metadata field in the users table. Since there was no direct column for KYC completion date, I assumed updated_at date corresponding to the "kyc_completed: true" might reflect this.

- `Sign-Up Date`: There was no direct sign-up date field, so I assumed the created_at timestamp in the users table represented that. 

- `Investment Type`: Used the plan_option field from both investment_plan and plans tables to infer the investment type since there was no direct column as investment_type and the plan_option from both tables reflected a similar format as example given in instruction.

- Cohorts (Transaction, Sign-Up, KYC): Created cohorts based on the month and year derived from the respective date fields.

- `Transaction Date`: Used `created_at` in the transactions table as the `transaction date` and derived the `week number of transactions` and `date of first transaction` accordingly.

## How to Run the Script
1. Clone the repository:

```python 

git clone https://github.com/Pearl-lab-sudo/Ladder-Assessment.git

cd Ladder-Assessment
```


2. Create a .env file in the root directory:

```python
DB_USER= your_username
DB_PASSWORD= your_password
DB_HOST= your_db_host
DB_PORT= 5432 (default)
DB_NAME= your_database_name
```
3. Install dependencies using pip
```bash
pip install pandas sqlalchemy python-dotenv
```
 > The following libraries are also used, but are part of Python’s standard library and do not require separate installation:

 `datetime`

 `logging`

`argparse`

 `pathlib`

5. Run the script

```python
python index.py
```
## Output Files
- `final_processed_transactions_<timestamp>.csv`
   Cleaned and joined transaction data across multiple tables.

- `final_transaction_summary_report_<timestamp>.csv`
   Summary metrics such as total transaction, Total investments and savings, and Ttal sign_ups.

- `final_transaction_cohort_analysis_<timestamp>.csv`
   Cohort analysis table showing transaction_cohort details.



 
