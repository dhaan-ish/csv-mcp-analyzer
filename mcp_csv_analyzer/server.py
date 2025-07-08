#!/usr/bin/env python3
import io
import json
import logging
import os
import sys
import tempfile
import traceback
from contextlib import redirect_stdout, redirect_stderr
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
from scipy import stats
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

from mcp.server.fastmcp import FastMCP

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NumpyEncoder, self).default(obj)

def safe_json_dumps(obj, **kwargs):
    """JSON dumps with numpy type handling."""
    return json.dumps(obj, cls=NumpyEncoder, **kwargs)

loaded_datasets: Dict[str, pd.DataFrame] = {}
execution_context: Dict[str, Any] = {
    'pd': pd,
    'np': np,
    'plt': plt,
    'sns': sns,
    'go': go,
    'px': px,
    'stats': stats,
    'StandardScaler': StandardScaler,
    'KMeans': KMeans,
    'PCA': PCA,
}

mcp = FastMCP("CSV Analyzer")

@mcp.tool()
def load_csv(file_path: str, encoding: str = "utf-8", separator: str = ",") -> str:
    """
    Load a CSV file into memory for analysis.
    
    Args:
        file_path: Path to the CSV file
        encoding: File encoding (default: utf-8)
        separator: CSV separator (default: comma)
    
    Returns:
        Summary information about the loaded dataset
    """
    try:
        if not os.path.exists(file_path):
            return f"Error: File '{file_path}' not found."
        
        df = pd.read_csv(file_path, encoding=encoding, sep=separator)
        
        dataset_name = Path(file_path).stem
        loaded_datasets[dataset_name] = df
        execution_context[dataset_name] = df
    
        summary = {
            "dataset_name": dataset_name,
            "shape": df.shape,
            "columns": list(df.columns),
            "data_types": df.dtypes.astype(str).to_dict(),
            "missing_values": df.isnull().sum().to_dict(),
            "memory_usage": f"{df.memory_usage(deep=True).sum() / 1024 / 1024:.2f} MB"
        }
        
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 0:
            summary["numeric_summary"] = df[numeric_cols].describe().to_dict()
        
        return f"Successfully loaded dataset '{dataset_name}':\n{safe_json_dumps(summary, indent=2)}"
        
    except Exception as e:
        return f"Error loading CSV file: {str(e)}"

@mcp.tool()
def list_datasets() -> str:
    """
    List all currently loaded datasets.
    
    Returns:
        Information about all loaded datasets
    """
    if not loaded_datasets:
        return "No datasets currently loaded."
    
    info = []
    for name, df in loaded_datasets.items():
        info.append({
            "name": name,
            "shape": df.shape,
            "columns": len(df.columns),
            "memory_usage": f"{df.memory_usage(deep=True).sum() / 1024 / 1024:.2f} MB"
        })
    
    return f"Loaded datasets:\n{safe_json_dumps(info, indent=2)}"

@mcp.tool()
def get_dataset_info(dataset_name: str) -> str:
    """
    Get detailed information about a specific dataset.
    
    Args:
        dataset_name: Name of the dataset to inspect
    
    Returns:
        Detailed information about the dataset
    """
    if dataset_name not in loaded_datasets:
        return f"Dataset '{dataset_name}' not found. Available datasets: {list(loaded_datasets.keys())}"
    
    df = loaded_datasets[dataset_name]
    
    info = {
        "name": dataset_name,
        "shape": df.shape,
        "columns": list(df.columns),
        "data_types": df.dtypes.astype(str).to_dict(),
        "missing_values": df.isnull().sum().to_dict(),
        "duplicate_rows": df.duplicated().sum(),
        "memory_usage": f"{df.memory_usage(deep=True).sum() / 1024 / 1024:.2f} MB"
    }
    
    info["head"] = df.head().to_dict('records')
    
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    if len(numeric_cols) > 0:
        info["numeric_summary"] = df[numeric_cols].describe().to_dict()
    
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns
    if len(categorical_cols) > 0:
        info["categorical_summary"] = {}
        for col in categorical_cols:
            value_counts = df[col].value_counts().head(10).to_dict()
            info["categorical_summary"][col] = value_counts
    
    return safe_json_dumps(info, indent=2)

@mcp.tool()
def execute_analysis_code(code: str, dataset_name: Optional[str] = None) -> str:
    """
    Execute Python code for data analysis on the loaded datasets.
    
    Args:
        code: Python code to execute
        dataset_name: Optional specific dataset to work with
    
    Returns:
        Results of the code execution including output and any plots generated
    """
    try:
        context = execution_context.copy()
        
        if dataset_name:
            if dataset_name not in loaded_datasets:
                return f"Dataset '{dataset_name}' not found. Available datasets: {list(loaded_datasets.keys())}"
            context['df'] = loaded_datasets[dataset_name]
        elif len(loaded_datasets) == 1:
            context['df'] = list(loaded_datasets.values())[0]
        
        context.update(loaded_datasets)
        
        stdout_capture = io.StringIO()
        stderr_capture = io.StringIO()
    
        with redirect_stdout(stdout_capture), redirect_stderr(stderr_capture):
            exec(code, context)
        
        stdout_output = stdout_capture.getvalue()
        stderr_output = stderr_capture.getvalue()
        
        result = {
            "execution_status": "success",
            "stdout": stdout_output,
            "stderr": stderr_output if stderr_output else None
        }
        
        if plt.get_fignums():
            plot_info = []
            for fig_num in plt.get_fignums():
                fig = plt.figure(fig_num)
                
                with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp_file:
                    fig.savefig(tmp_file.name, dpi=150, bbox_inches='tight')
                    plot_info.append({
                        "figure_number": fig_num,
                        "file_path": tmp_file.name
                    })
                
                plt.close(fig)
            
            result["plots_generated"] = len(plot_info)
            result["plot_files"] = plot_info
        
        return safe_json_dumps(result, indent=2)
        
    except Exception as e:
        error_info = {
            "execution_status": "error",
            "error_type": type(e).__name__,
            "error_message": str(e),
            "traceback": traceback.format_exc()
        }
        return safe_json_dumps(error_info, indent=2)

@mcp.tool()
def generate_basic_insights(dataset_name: str) -> str:
    """
    Generate basic statistical insights for a dataset.
    
    Args:
        dataset_name: Name of the dataset to analyze
    
    Returns:
        Basic insights and statistics about the dataset
    """
    if dataset_name not in loaded_datasets:
        return f"Dataset '{dataset_name}' not found. Available datasets: {list(loaded_datasets.keys())}"
    
    df = loaded_datasets[dataset_name]
    
    insights = {
        "dataset_overview": {
            "total_rows": len(df),
            "total_columns": len(df.columns),
            "missing_data_percentage": (df.isnull().sum().sum() / (len(df) * len(df.columns))) * 100,
            "duplicate_rows": df.duplicated().sum()
        }
    }
    
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    if len(numeric_cols) > 0:
        insights["numeric_analysis"] = {}
        for col in numeric_cols:
            col_data = df[col].dropna()
            insights["numeric_analysis"][col] = {
                "mean": col_data.mean(),
                "median": col_data.median(),
                "std": col_data.std(),
                "min": col_data.min(),
                "max": col_data.max(),
                "missing_count": df[col].isnull().sum(),
                "outliers_iqr": len(col_data[
                    (col_data < col_data.quantile(0.25) - 1.5 * (col_data.quantile(0.75) - col_data.quantile(0.25))) |
                    (col_data > col_data.quantile(0.75) + 1.5 * (col_data.quantile(0.75) - col_data.quantile(0.25)))
                ])
            }
    
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns
    if len(categorical_cols) > 0:
        insights["categorical_analysis"] = {}
        for col in categorical_cols:
            insights["categorical_analysis"][col] = {
                "unique_values": df[col].nunique(),
                "most_frequent": df[col].mode().iloc[0] if not df[col].mode().empty else None,
                "missing_count": df[col].isnull().sum(),
                "top_5_values": df[col].value_counts().head(5).to_dict()
            }
    
    if len(numeric_cols) > 1:
        corr_matrix = df[numeric_cols].corr()
        strong_correlations = []
        for i in range(len(corr_matrix.columns)):
            for j in range(i + 1, len(corr_matrix.columns)):
                corr_val = corr_matrix.iloc[i, j]
                if abs(corr_val) > 0.7:
                    strong_correlations.append({
                        "column1": corr_matrix.columns[i],
                        "column2": corr_matrix.columns[j],
                        "correlation": corr_val
                    })
        
        insights["correlation_analysis"] = {
            "strong_correlations": strong_correlations
        }
    
    return safe_json_dumps(insights, indent=2)

@mcp.tool()
def export_dataset(dataset_name: str, file_path: str, format: str = "csv") -> str:
    """
    Export a dataset to a file.
    
    Args:
        dataset_name: Name of the dataset to export
        file_path: Path where to save the file
        format: Export format (csv, excel, json)
    
    Returns:
        Status of the export operation
    """
    if dataset_name not in loaded_datasets:
        return f"Dataset '{dataset_name}' not found. Available datasets: {list(loaded_datasets.keys())}"
    
    df = loaded_datasets[dataset_name]
    
    try:
        if format.lower() == "csv":
            df.to_csv(file_path, index=False)
        elif format.lower() == "excel":
            df.to_excel(file_path, index=False)
        elif format.lower() == "json":
            df.to_json(file_path, orient='records', indent=2)
        else:
            return f"Unsupported format: {format}. Supported formats: csv, excel, json"
        
        return f"Successfully exported dataset '{dataset_name}' to '{file_path}' in {format} format."
        
    except Exception as e:
        return f"Error exporting dataset: {str(e)}"

@mcp.tool()
def clear_datasets() -> str:
    """
    Clear all loaded datasets from memory.
    
    Returns:
        Confirmation message
    """
    global loaded_datasets
    
    count = len(loaded_datasets)
    loaded_datasets.clear()
    keys_to_remove = [k for k in execution_context.keys() if k not in ['pd', 'np', 'plt', 'sns', 'go', 'px', 'stats', 'StandardScaler', 'KMeans', 'PCA']]
    for key in keys_to_remove:
        execution_context.pop(key, None)
    
    return f"Cleared {count} dataset(s) from memory."

def main():
    """Main entry point for the MCP server."""
    print("SErver started")
    mcp.run(transport="stdio")

if __name__ == "__main__":
    main() 