# Master-s-Project-24

Initial Idea -  Building a Business Operations Assistant driven by AI.

Approach:
Constructing a Retrieval Augmented Generation (RAG) pipeline integrated or powered with LLM to create an interface wherein the end user can ask business-operations related queries and get answers with relevant contextual information and supporting visualizations for data distributions. Python offers a rich set of libraries including matplotlib, seaborn, plotly which can be utilized for the sake of creating visualizations on queries by user.
Instead of relying solely on the LLM to generate responses, the system will retrieve the most relevant information from a pre-defined set of business documents, and the LLM will generate a contextually accurate response.The system will convert the query into a vector and search a pre-processed database of business documents including case studies, business reports, QAs from various data sources using a vector search.These retrieved documents will be passed as context to the GPT model.
GPT will process both the user query and the retrieved business documents to generate a response. 
When it comes to data analysis and visualization, we can analyze the types of questions users ask and how often each category (finance, marketing, supply chain etc.) is queried. Distribution of data across various sub-domains for instance such as finance, marketing etc. 
Performance evaluation of the system will be done using certain metrics such as precision, recall, F-1 score and accuracy. These metrics will help assess how well the system is retrieving relevant documents and generating accurate responses based on the business operations use case. For instance, precision will help in evaluating how many of the retrieved documents (or generated answers) are actually relevant to the user’s query. On the other hand, Recall will help evaluate how well the system is covering all the relevant information available in the database. Higher accuracy metrics such as F-1 score will help in checking overall performance of model. Higher the F-1 score, better is the overall performance. Plotting precision vs. recall before and after fine-tuning. This curve will show how the tradeoff between precision and recall changes, highlighting the model’s improvements. Tracking the improvements in precision, recall, F1-score, and accuracy before and after fine-tuning will give a clear picture of how effectively the system retrieves relevant business documents and generates factually accurate responses.
Coming onto the step of augmentation, the system will retrieve and rank relevant documents, such as articles or reports about supply chain optimization, to provide relevant context. 
Right from feeding documents into the LLM to delivering response to the user, it's going to involve several intermediate steps. Let's assume, once the most relevant documents are retrieved from the database, they will be passed to the LLM (e.g., GPT-4). The system will ensure that these documents provide the necessary context for the LLM to generate an informed response. The LLM will then process the user's query in conjunction with the retrieved documents. It will analyze both the query and the contextual information to better understand the user's needs and the specific domain knowledge required to formulate a relevant answer.Based on the retrieved documents and the user query, the LLM will generate an accurate and contextually enriched answer. The response will be tailored to the query, providing a coherent and comprehensive output that draws directly from the most relevant business operations documents. After processing, it will generate a final response that will be delivered to the user. This answer will be both precise and contextually tailored, offering the user a well-rounded, accurate response to their business-related inquiry.
The entire bot once ready can be hosted on streamlit so that users can have a clickable local host web url to access the interface.

Data Source:


