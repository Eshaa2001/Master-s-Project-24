# Master-s-Project-24

**Initial Idea:**  

Building a Business Operations Assistant driven by AI.

**Objective**

Tailoring our LLM for a specific domain while also exploring different ways to fine-tune them.

**Approach:**

Constructing a Retrieval Augmented Generation (RAG) pipeline integrated or powered with LLM to create an interface wherein the end user can ask business-operations related queries and get answers with relevant contextual information and supporting visualizations for data distributions. Python offers a rich set of libraries including matplotlib, seaborn, plotly which can be utilized for the sake of creating visualizations on queries by user.
Instead of relying solely on the LLM to generate responses, the system will retrieve the most relevant information from a pre-defined set of business documents, and the LLM will generate a contextually accurate response.The system will convert the query into a vector and search a pre-processed database of business documents including case studies, business reports, QAs from various data sources using a vector search.These retrieved documents will be passed as context to the GPT model.
GPT will process both the user query and the retrieved business documents to generate a response. 
When it comes to data analysis and visualization, we can analyze the types of questions users ask and how often each category (finance, marketing, supply chain etc.) is queried. Distribution of data across various sub-domains for instance such as finance, marketing etc. 

Performance evaluation of the system will be done using certain metrics such as precision, recall, F-1 score and accuracy. These metrics will help assess how well the system is retrieving relevant documents and generating accurate responses based on the business operations use case. For instance, precision will help in evaluating how many of the retrieved documents (or generated answers) are actually relevant to the user’s query. On the other hand, Recall will help evaluate how well the system is covering all the relevant information available in the database. Higher accuracy metrics such as F-1 score will help in checking overall performance of model. Higher the F-1 score, better is the overall performance. Plotting precision vs. recall before and after fine-tuning. This curve will show how the tradeoff between precision and recall changes, highlighting the model’s improvements. Tracking the improvements in precision, recall, F1-score, and accuracy before and after fine-tuning will give a clear picture of how effectively the system retrieves relevant business documents and generates factually accurate responses.

Coming onto the step of augmentation, the system will retrieve and rank relevant documents, such as articles or reports about supply chain optimization, to provide relevant context. 
Right from feeding documents into the LLM to delivering response to the user, it's going to involve several intermediate steps. Let's assume, once the most relevant documents are retrieved from the database, they will be passed to the LLM (e.g., GPT-4). The system will ensure that these documents provide the necessary context for the LLM to generate an informed response. The LLM will then process the user's query in conjunction with the retrieved documents. It will analyze both the query and the contextual information to better understand the user's needs and the specific domain knowledge required to formulate a relevant answer.Based on the retrieved documents and the user query, the LLM will generate an accurate and contextually enriched answer. The response will be tailored to the query, providing a coherent and comprehensive output that draws directly from the most relevant business operations documents. After processing, it will generate a final response that will be delivered to the user. This answer will be both precise and contextually tailored, offering the user a well-rounded, accurate response to their business-related inquiry.
The entire bot once ready can be hosted on streamlit so that users can have a clickable local host web url to access the interface.

**Data Source:**
Historical Data in .csv files related to Business Operations, Finanace, Supply-chain, Marketing

https://www.kaggle.com/datasets/danish1212/business-operations
https://www.kaggle.com/datasets/adhoppin/financial-data
https://www.kaggle.com/datasets/nitindatta/finance-data
https://www.kaggle.com/datasets/dorothyjoel/us-regional-sales
https://www.kaggle.com/datasets/bytadit/ecommerce-order-dataset
https://www.kaggle.com/datasets/sahilnbajaj/marketing-campaigns-data-set
https://www.kaggle.com/datasets/fahmidachowdhury/customer-segmentation-data-for-marketing-analysis

Such datasets will be useful to extract trends in financial operations, business and marketing strategies. To analyze how companies respond to economic shifts and have revised their marketing strategies accordingly.

**FAQs Data**
https://quant.stackexchange.com/
https://economics.stackexchange.com/
https://or.stackexchange.com/
https://quant.meta.stackexchange.com/
These websites are rich sources of FAQs related to business, operations research, finance. We can generate XML files for these which can then be parsed 
Here are some other data sources on "What do the bid trends say each Fiscal Year from 2016-20", they can be downloaded in .csv format as well.
https://catalog.data.gov/dataset/fy16-bid-trends-report-data
https://catalog.data.gov/dataset/fy18-bid-trends-report-data
https://catalog.data.gov/dataset/fy19-bid-trends-report-data
https://catalog.data.gov/dataset/fy20-bid-trends-report-data

We can utilize these datases for building a comprehensive knowledge base on business related FAQs, perfect for retrieval-based question answering.

**Case Studies**
Harvard Busines Review (HBR) has a rich repository of business case studies focusing on real-world company operations, decisions, and strategies across various industries.
https://store.hbr.org/case-studies/?srsltid=AfmBOop1zIiE6Dmk_UX70MH1apQyBiatwiptu3UnvsY7F1DWsLG50-TI#/filter:custom_subject:Sales$2520$2526$2520Marketing

CB Reports contain market trends, company case studies, and investment patterns in various industries.
Use Case: Utilize the business case studies and trends to inform more dynamic answers around business growth, market predictions, and operational strategies.
https://www.cbinsights.com/research/report/


These case studies as supporting documents for answering specific business operation queries, especially for retrieval purposes.

**World Bank Open Data**
https://data.worldbank.org/topic/financial-sector (This is an example link for financial sector dataset -  downloadable in .csv, .xlsx format)
https://data.worldbank.org/indicator/CM.MKT.LCAP.CD (This is an example link for market capitalization dataset -  downloadable in .csv, .xlsx format)

These contain economic indicators, financial statistics, and business data for companies and countries worldwide.
Use Case: Can be used for providing context and analysis on global business trends, which helps in creating background knowledge for more informed responses.




**Other**
https://fred.stlouisfed.org/series/ISRATIO (multiple datasets available in .xlsx, .csv on various domains including Business Operations, Finance, Marketing, Supply-chain)



**Data Format and How it is to be handled**

For question-answer pairs (like Stack Exchange Data Dump), the data usually comes in easy-to-read XML or JSON files. Each entry will have a question, its answer, and some extra details like topic **tags** or user ratings. We can organize this data into a clear structure to train models that retrieve relevant answers. This will play an integral role in training the model. 

For text documents (like research papers or reports), the data might be in PDF or HTML formats. You’ll need to convert these into plain text using tools like PyPDF2 or BeautifulSoup so the system can use them for finding the right information.

For structured business data, such as datasets from places like the UCI Repository, World Bank, or Yelp, these usually come in CSV files. You can use these for different types of data analysis, like forecasting or classifying information.

These different types of data will together form the knowledge base for your RAG system, making it easier for the model to answer questions related to business operations.


**Research Motivation**
Following were some research papers from google scholar which were quite insightful:
1. https://www.researchgate.net/profile/Reshmi-Sankar/publication/323451431_EMPOWERING_CHATBOTS_WITH_BUSINESS_INTELLIGENCE_BY_BIG_DATA_INTEGRATION/links/5b9351b4299bf14739257a86/EMPOWERING-CHATBOTS-WITH-BUSINESS-INTELLIGENCE-BY-BIG-DATA-INTEGRATION.pdf
2. https://ieeexplore.ieee.org/abstract/document/9500127
3. https://dl.acm.org/doi/abs/10.1145/3640794.3665538?casa_token=pv4OYavJppUAAAAA:p-RlA7CnS3fhXRSkI8na57Lh2jI178MNCQtJtUbdkPhqEWJXX5fRVFzXgB0gkTYwn1gF_gR-IQwGgw
4. https://arxiv.org/abs/2406.16937
5. https://github.com/Revanth980727/DataAnalysis-Chatbot



