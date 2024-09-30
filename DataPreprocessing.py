import pandas as pd
from bs4 import BeautifulSoup
import os

def preprocess_csv(file_paths):
    """
    Preprocess multiple CSV files and combine them into one DataFrame.

    :param file_paths: list of str, paths to the CSV files
    :return: pandas.DataFrame, combined data
    """
    df_list = []
    for file_path in file_paths:
        try:
            # Read each CSV file and append it to the list
            df = pd.read_csv(file_path, encoding='ISO-8859-1')
            df_list.append(df)
        except UnicodeDecodeError as e:
            print(f"Error reading {file_path}: {e}")
    
    # Combine all DataFrames in the list into one
    combined_df = pd.concat(df_list, ignore_index=True)
    return combined_df

# Example usage: List of CSV file paths
csv_files = [
    r"C:/Users/eshaa/OneDrive/Documents/MFP/Business_Operations.csv",
    r"C:/Users/eshaa/OneDrive/Documents/MFP/customer_segmentation_data.csv",
    r"C:/Users/eshaa/OneDrive/Documents/MFP/Finance_data.csv",
    r"C:/Users/eshaa/OneDrive/Documents/MFP/marketing_data.csv"
]

# Call the function to combine the CSVs
combined_data = preprocess_csv(csv_files)

# Display the first few rows of the combined DataFrame
print(combined_data.head())



#def preprocess_html(file_path):
#    """
#    Convert an HTML file into plain text using BeautifulSoup.
#
#    :param file_path: str, path to the HTML file
#    :return: str, extracted text from HTML
#    """
#    with open(file_path, 'r', encoding='utf-8') as file:
#        soup = BeautifulSoup(file, 'html.parser')
#    return soup.get_text()

def preprocess_xml(file_paths):
    """
        Convert multiple XML files into plain text by parsing their content.

        :param file_paths: list of str, paths to the XML files
        :return: str, combined extracted text from all XML files
    """
    combined_text = ""

    for file_path in file_paths:
        try:
            with open(file_path, 'r', encoding='utf-8') as file:
                soup = BeautifulSoup(file, 'xml')
                combined_text += soup.get_text() + "\n"  # Append text from each file
        except Exception as e:
            print(f"Error processing {file_path}: {e}")
    
    return combined_text

# Example usage: List of XML file paths
xml_files = [
    r"C:/Users/eshaa/OneDrive/Documents/MFP/questions_answers.xml",
]

# Call the function to combine the XMLs
combined_xml_text = preprocess_xml(xml_files)

# Output the combined text from all XML files
print(combined_xml_text)


