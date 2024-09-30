import pandas as pd
from bs4 import BeautifulSoup
import os


def preprocess_csv("C:/Users/eshaa/OneDrive/Documents/Master's Final Project/customer_segmentation_data.csv","C:/Users/eshaa/OneDrive/Documents/Master's Final Project/Finance_data.csv"):
    """
    Preprocess CSV files by reading the data and performing basic cleaning.

    :param file_path: str, path to the CSV file
    :return: pandas.DataFrame, cleaned data
    """
    df = pd.read_csv("C:/Users/eshaa/OneDrive/Documents/Master's Final Project/customer_segmentation_data.csv","C:/Users/eshaa/OneDrive/Documents/Master's Final Project/Finance_data.csv"h)
    # Example: Clean data by removing null values, duplicates, etc.
    df.dropna(inplace=True)
    return df
    """
    Preprocess CSV files by reading the data and performing basic cleaning.

    :param file_path: str, path to the CSV file
    :return: pandas.DataFrame, cleaned data
    """
    df = pd.read_csv(file_path)
    # Example: Clean data by removing null values, duplicates, etc.
    df.dropna(inplace=True)
    return df

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

def preprocess_xml("C:/Users/eshaa/OneDrive/Documents/Master's Final Project/questions_answers.xml"):
    """
    Convert an XML file into plain text by parsing its content.

    :param file_path: str, path to the XML file
    :return: str, extracted text from XML
    """
    with open(file_path, 'r', encoding='utf-8') as file:
        soup = BeautifulSoup(file, 'xml')
    return soup.get_text()


