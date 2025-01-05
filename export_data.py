import pandas as pd

def export_data(dataframe, file_format="csv", filename="exported_data"):
    """
    Export a DataFrame to a specified file format (CSV or Excel).
    
    Parameters:
        dataframe (pd.DataFrame): The data to export.
        file_format (str): The format to export to ("csv" or "excel").
        filename (str): The name of the output file (without extension).
        
    Returns:
        str: The path of the exported file.
    """
    try:
        # Define the output file path
        file_path = f"{filename}.{file_format}"
        
        if file_format.lower() == "csv":
            dataframe.to_csv(file_path, index=False)
        elif file_format.lower() == "excel":
            dataframe.to_excel(file_path, index=False, engine="openpyxl")
        else:
            return "Invalid file format! Please choose 'csv' or 'excel'."
        
        return f"File successfully exported to: {file_path}"
    except Exception as e:
        return f"Error exporting file: {e}"
