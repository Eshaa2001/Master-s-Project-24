import requests
from bs4 import BeautifulSoup
import xml.etree.ElementTree as ET

# Create the root of the XML document
root = ET.Element('questions')

# Define the URL of the website
url = "https://economics.stackexchange.com/"

# Send a GET request to fetch the webpage content
response = requests.get(url)

# Check if the request was successful
if response.status_code == 200:
    # Parse the homepage content using BeautifulSoup
    soup = BeautifulSoup(response.text, 'html.parser')
    
    # Find all the questions on the homepage
    questions = soup.find_all('div', class_='s-post-summary--content')

    # Loop through each question and extract the title and link
    for question in questions:
        title = question.find('a', class_='s-link').get_text()
        link = "https://economics.stackexchange.com/" + question.find('a', class_='s-link')['href']
        
        # Create an XML element for each question
        question_element = ET.SubElement(root, 'question')
        title_element = ET.SubElement(question_element, 'title')
        title_element.text = title
        link_element = ET.SubElement(question_element, 'link')
        link_element.text = link
        
        # Now, fetch the content from the individual question page
        question_response = requests.get(link)
        
        if question_response.status_code == 200:
            # Parse the question page content
            question_soup = BeautifulSoup(question_response.text, 'html.parser')
            
            # Extract the question body/content
            question_content = question_soup.find('div', class_='s-prose')
            if question_content:
                # Add question content to XML
                content_element = ET.SubElement(question_element, 'content')
                content_element.text = question_content.get_text(separator="\n").strip()
            else:
                print(f"No question content found for {title}.")
            
            # Extract answers
            answers = question_soup.find_all('div', class_='answercell')
            
            if answers:
                answers_element = ET.SubElement(question_element, 'answers')
                for idx, answer in enumerate(answers, 1):
                    answer_content = answer.find('div', class_='s-prose')
                    if answer_content:
                        # Create XML element for each answer
                        answer_element = ET.SubElement(answers_element, 'answer')
                        answer_element.text = answer_content.get_text(separator="\n").strip()
            else:
                print(f"No answers found for {title}.")
            
        else:
            print(f"Failed to retrieve question content for {link}. Status code: {question_response.status_code}")

# Write the XML tree to a file
tree = ET.ElementTree(root)
with open("questions_answers.xml", "wb") as f:
    tree.write(f, encoding='utf-8', xml_declaration=True)

print("Data saved to questions_answers.xml")
