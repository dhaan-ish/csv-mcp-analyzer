# MCP CSV Analyzer

An MCP (Model Context Protocol) server that provides powerful tools for reading CSV files and executing LLM-generated Python code to extract useful insights from data.

## Features

- **CSV File Loading**: Load CSV files with customizable encoding and separators
- **Data Analysis**: Execute Python code for comprehensive data analysis
- **Statistical Insights**: Generate basic statistical insights automatically
- **Data Visualization**: Create plots and visualizations using matplotlib, seaborn, and plotly
- **Multiple Dataset Support**: Work with multiple datasets simultaneously
- **Safe Code Execution**: Secure execution environment for LLM-generated analysis code
- **Export Capabilities**: Export processed data in various formats (CSV, Excel, JSON)

## Tools Available

### 1. `load_csv`
Load a CSV file into memory for analysis.
- **Parameters**: 
  - `file_path`: Path to the CSV file
  - `encoding`: File encoding (default: utf-8)
  - `separator`: CSV separator (default: comma)

### 2. `list_datasets`
List all currently loaded datasets with basic information.

### 3. `get_dataset_info`
Get detailed information about a specific dataset including:
- Shape and columns
- Data types
- Missing values
- Sample data
- Statistical summaries

### 4. `execute_analysis_code`
Execute Python code for data analysis with access to:
- pandas (`pd`)
- numpy (`np`)
- matplotlib (`plt`)
- seaborn (`sns`)
- plotly (`go`, `px`)
- scipy.stats (`stats`)
- scikit-learn components
- All loaded datasets

### 5. `generate_basic_insights`
Automatically generate statistical insights including:
- Dataset overview
- Numeric column analysis
- Categorical column analysis
- Correlation analysis

### 6. `export_dataset`
Export datasets to files in various formats (CSV, Excel, JSON).

### 7. `clear_datasets`
Clear all loaded datasets from memory.

### 8. `send_email`
Send emails with attachments using Gmail API.
- **Parameters**:
  - `to_email`: Recipient email address (comma-separated for multiple)
  - `subject`: Email subject line
  - `body`: Email body content
  - `body_type`: Type of body content - "plain" or "html" (default: plain)
  - `cc_emails`: CC recipients (comma-separated) [optional]
  - `bcc_emails`: BCC recipients (comma-separated) [optional]
  - `attachments`: List of file paths to attach [optional]
  - `token_file`: Path to the OAuth token file (default: token.json)

## Installation

1. Clone or create the project:
```bash
# If using uv (recommended)
uv init mcp-csv-analyzer
cd mcp-csv-analyzer
```

2. Install dependencies:
```bash
uv sync
```

3. Run the server:
```bash
uv run python main.py
```

## Gmail API Setup

To use the email sending functionality, you need to set up Gmail API credentials:

### 1. Create Google Cloud Project and Enable Gmail API

1. Go to [Google Cloud Console](https://console.cloud.google.com/)
2. Create a new project or select an existing one
3. Enable the Gmail API for your project
4. Create OAuth 2.0 credentials (Desktop application type)
5. Download the credentials JSON file

### 2. Generate OAuth Token

Run the included token generation script:

```bash
# With credentials file
uv run python generate_oauth_token.py credentials.json

# Or enter credentials manually
uv run python generate_oauth_token.py
```

The script will:
- Read your client ID and client secret
- Open a browser for authentication
- Generate and save the OAuth token to `token.json`
- Display token information for verification

### 3. Token Management

- The token is saved to `token.json` by default
- Tokens expire after a period but will auto-refresh if a refresh token is present
- Keep the token file secure as it provides access to send emails from your account
- You can regenerate the token at any time by running the script again

## Usage Examples

### Loading and Analyzing Data

1. **Load a CSV file**:
```python
# Tool call: load_csv
{
    "file_path": "data/sales_data.csv",
    "encoding": "utf-8",
    "separator": ","
}
```

2. **Get dataset information**:
```python
# Tool call: get_dataset_info
{
    "dataset_name": "sales_data"
}
```

3. **Execute analysis code**:
```python
# Tool call: execute_analysis_code
{
    "code": """
# Basic data exploration
print("Dataset shape:", df.shape)
print("\\nColumn info:")
print(df.info())

# Statistical summary
print("\\nStatistical summary:")
print(df.describe())

# Check for missing values
print("\\nMissing values:")
print(df.isnull().sum())

# Create a simple visualization
plt.figure(figsize=(10, 6))
if 'sales' in df.columns:
    plt.hist(df['sales'], bins=30, alpha=0.7)
    plt.title('Sales Distribution')
    plt.xlabel('Sales Amount')
    plt.ylabel('Frequency')
    plt.show()
"""
}
```

4. **Generate automatic insights**:
```python
# Tool call: generate_basic_insights
{
    "dataset_name": "sales_data"
}
```

### Advanced Analysis Examples

**Correlation Analysis**:
```python
# Tool call: execute_analysis_code
{
    "code": """
# Correlation heatmap
import seaborn as sns
plt.figure(figsize=(12, 8))
correlation_matrix = df.select_dtypes(include=[np.number]).corr()
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0)
plt.title('Correlation Heatmap')
plt.tight_layout()
plt.show()
"""
}
```

**Time Series Analysis** (if date column exists):
```python
# Tool call: execute_analysis_code
{
    "code": """
# Convert date column and create time series plot
df['date'] = pd.to_datetime(df['date'])
df_sorted = df.sort_values('date')

plt.figure(figsize=(15, 6))
plt.plot(df_sorted['date'], df_sorted['sales'])
plt.title('Sales Over Time')
plt.xlabel('Date')
plt.ylabel('Sales')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()
"""
}
```

**Machine Learning Analysis**:
```python
# Tool call: execute_analysis_code
{
    "code": """
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# Prepare numeric data for clustering
numeric_cols = df.select_dtypes(include=[np.number]).columns
X = df[numeric_cols].dropna()

# Standardize the data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Perform K-means clustering
kmeans = KMeans(n_clusters=3, random_state=42)
clusters = kmeans.fit_predict(X_scaled)

# Visualize clusters (first two features)
plt.figure(figsize=(10, 6))
plt.scatter(X_scaled[:, 0], X_scaled[:, 1], c=clusters, cmap='viridis')
plt.title('K-means Clustering')
plt.xlabel(f'{numeric_cols[0]} (standardized)')
plt.ylabel(f'{numeric_cols[1]} (standardized)')
plt.colorbar()
plt.show()

print(f"Cluster distribution: {np.bincount(clusters)}")
"""
}
```

### Email Functionality Examples

**Send Analysis Results via Email**:
```python
# Tool call: send_email
{
    "to_email": "recipient@example.com",
    "subject": "Data Analysis Report - Sales Data",
    "body": "Please find attached the analysis results for the sales data.\n\nKey findings:\n- Total revenue: $1.2M\n- Growth rate: 15%\n- Top performing category: Electronics",
    "attachments": ["sales_analysis.pdf", "charts.png"]
}
```

**Send HTML Email with Multiple Recipients**:
```python
# Tool call: send_email
{
    "to_email": "team@example.com",
    "subject": "Weekly Data Insights",
    "body": "<h2>Weekly Report</h2><p>This week's <b>key metrics</b>:</p><ul><li>Orders: 1,234</li><li>Revenue: $45,678</li></ul>",
    "body_type": "html",
    "cc_emails": "manager@example.com",
    "attachments": ["weekly_report.xlsx"]
}
```

**Automated Reporting Workflow**:
```python
# First, generate analysis and export
# Tool call: execute_analysis_code
{
    "code": """
# Generate comprehensive report
report_data = []
for dataset_name, df in loaded_datasets.items():
    summary = {
        'dataset': dataset_name,
        'rows': len(df),
        'columns': len(df.columns),
        'numeric_mean': df.select_dtypes(include=[np.number]).mean().to_dict()
    }
    report_data.append(summary)

# Create visualization
plt.figure(figsize=(12, 6))
# ... create charts ...
plt.savefig('analysis_charts.png')

# Export summary
import json
with open('analysis_summary.json', 'w') as f:
    json.dump(report_data, f, indent=2)
"""
}

# Then send the results
# Tool call: send_email
{
    "to_email": "stakeholders@example.com",
    "subject": "Automated Analysis Report",
    "body": "Attached are the automated analysis results generated on " + datetime.now().strftime('%Y-%m-%d'),
    "attachments": ["analysis_summary.json", "analysis_charts.png"]
}
```

## Dependencies

- **mcp**: Model Context Protocol framework
- **pandas**: Data manipulation and analysis
- **numpy**: Numerical computing
- **matplotlib**: Basic plotting
- **seaborn**: Statistical data visualization
- **plotly**: Interactive plots
- **scipy**: Scientific computing
- **scikit-learn**: Machine learning library
- **openpyxl**: Excel file support

## Security Notes

- Code execution is performed in a controlled environment
- File access is limited to specified paths
- No network access from executed code
- Temporary files are used for plot generation

## License

This project is open source and available under the MIT License.

## Contributing

1. Fork the repository
2. Create a feature branch
3. Add your improvements
4. Submit a pull request

## Support

For issues and questions, please create an issue in the project repository.
