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
