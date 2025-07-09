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
- **Email Integration**: Send analysis results via Gmail API
- **Azure OpenAI Client**: Interactive client powered by Azure OpenAI for natural language interaction

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

## Project Structure

```
mcp-csv-analyzer/
├── mcp_csv_analyzer/       # MCP server implementation
│   ├── __init__.py
│   └── server.py          # Main server with all tools
├── client/                # Azure OpenAI client
│   ├── __init__.py
│   └── app.py            # Interactive client application
├── generate_token/        # Gmail OAuth token generator
│   ├── __init__.py
│   └── generate_oauth_token.py
├── scripts/              # Convenience runner scripts
│   ├── __init__.py
│   ├── run_mcp.py       # Complete system runner
│   ├── run_server.py    # Server runner
│   ├── run_client.py    # Client runner
│   └── run_token_generator.py
├── pyproject.toml        # Project configuration and dependencies
├── test_data.csv         # Sample data file
└── README.md            # This file
```

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

3. Configure Azure OpenAI (for the client):
Create a `.env` file with your Azure OpenAI credentials:
```env
AZURE_OPENAI_ENDPOINT=https://your-resource-name.openai.azure.com/
AZURE_OPENAI_DEPLOYMENT=your-deployment-name
AZURE_OPENAI_API_KEY=your-api-key
AZURE_OPENAI_API_VERSION=2024-02-01
```

## Running the System

### Quick Start (Recommended)

The easiest way to run the complete system is using the integrated runner:

```bash
uv run mcp
```

This command will:
1. Check if Gmail OAuth token exists (`token.json`)
2. If token doesn't exist, offer to run the token generator
3. Start the MCP server in the background
4. Launch the Azure OpenAI-powered client
5. Gracefully shut down the server when you exit

### Available Commands

#### Complete System Runner
```bash
uv run mcp
```
Runs the complete MCP system with automatic token checking and server management.

#### Individual Components

**MCP Server Only**:
```bash
uv run mcp_server
```
Starts just the MCP CSV Analyzer server that provides all the data analysis tools.

**MCP Client Only**:
```bash
uv run mcp_client
```
Starts just the Azure OpenAI-powered client. Requires:
- MCP server to be running separately
- Azure OpenAI credentials configured in `.env`

**Generate OAuth Token**:
```bash
uv run generate_token
```
Runs the OAuth token generator for Gmail integration.

### Running Components Manually

If you prefer to run components separately for development or debugging:

1. **Generate Gmail OAuth token** (one-time setup):
   ```bash
   uv run generate_token
   ```

2. **Start the MCP server**:
   ```bash
   uv run mcp_server
   # Or directly:
   uv run python mcp_csv_analyzer/server.py
   ```

3. **In another terminal, run the client**:
   ```bash
   uv run mcp_client
   # Or directly:
   uv run python client/app.py
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

Run the token generation script:

```bash
uv run generate_token
```

Or with a specific credentials file:
```bash
uv run generate_token credentials.json
```

The script will:
- Read your client ID and client secret (either from file or manual input)
- Open a browser for authentication
- Generate and save the OAuth token to `token.json`
- Display token information for verification

### 3. Token Management

- The token is saved to `token.json` by default
- Tokens expire after a period but will auto-refresh if a refresh token is present
- Keep the token file secure as it provides access to send emails from your account
- You can regenerate the token at any time by running the script again

## Usage Examples

### Using the Interactive Client

When you run `uv run mcp` or `uv run mcp_client`, you'll enter an interactive session where you can use natural language to analyze data:

```
You: Load the test_data.csv file and tell me what columns it has
Assistant: I'll help you load the test_data.csv file and show you its columns...

You: Create a histogram of the numeric columns
Assistant: I'll create histograms for the numeric columns in your dataset...

You: Calculate the correlation between all numeric variables
Assistant: I'll calculate and visualize the correlation matrix for all numeric variables...

You: Send me an email with the analysis results
Assistant: I'll prepare and send an email with the analysis results...
```

The client automatically:
- Detects when to use tools based on your request
- Chains multiple tools together for complex tasks
- Provides clear feedback about what it's doing
- Maintains conversation context for follow-up questions

### Loading and Analyzing Data (Direct Tool Usage)

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

### Core Dependencies
- **mcp**: Model Context Protocol framework
- **pandas**: Data manipulation and analysis
- **numpy**: Numerical computing
- **matplotlib**: Basic plotting
- **seaborn**: Statistical data visualization
- **plotly**: Interactive plots
- **scipy**: Scientific computing
- **scikit-learn**: Machine learning library
- **openpyxl**: Excel file support

### Client Dependencies
- **openai**: Azure OpenAI SDK for the client
- **pydantic-ai**: AI agent framework for tool integration
- **rich**: Terminal formatting for interactive client
- **python-dotenv**: Environment variable management

### Email Integration
- **google-auth**: Google authentication
- **google-auth-oauthlib**: OAuth flow for Google APIs
- **google-api-python-client**: Gmail API client

## Security Notes

- Code execution is performed in a controlled environment
- File access is limited to specified paths
- No network access from executed code
- Temporary files are used for plot generation

## Troubleshooting

### Common Issues

**Import Errors**:
- Make sure you've installed the project with `uv sync`
- Check that all dependencies are properly installed

**Client Connection Issues**:
- Ensure the MCP server is running before starting the client
- Check that your Azure OpenAI credentials are correctly set in `.env`
- Verify the server is running in SSE mode on port 8000

**OAuth Token Issues**:
- Make sure you've completed the token generation process with `uv run generate_token`
- Check that `token.json` exists in the project root or `mcp_csv_analyzer/` directory
- Ensure your Google Cloud project has Gmail API enabled

**Script Not Found**:
- Run `uv sync` to ensure all project scripts are properly installed
- Use `uv run <command>` instead of running scripts directly

**Server Fails to Start**:
- Check if another process is using port 8000
- Ensure you have proper permissions to create temporary files for plots
- Check Python version compatibility (requires Python 3.10+)

## License

This project is open source and available under the MIT License.

## Contributing

1. Fork the repository
2. Create a feature branch
3. Add your improvements
4. Submit a pull request

## Support

For issues and questions, please create an issue in the project repository.
