Initial Idea:

Building a Business Operations Assistant driven by AI.

Objective

Tailoring our LLM for a specific domain while also exploring different ways to fine-tune them.

Approach:

Constructing a Retrieval Augmented Generation (RAG) pipeline integrated or powered with LLM to create an interface wherein the end user can ask business-operations related queries and get answers with relevant contextual information and supporting visualizations for data distributions. Python offers a rich set of libraries including matplotlib, seaborn, plotly which can be utilized for the sake of creating visualizations on queries by user. Instead of relying solely on the LLM to generate responses, the system will retrieve the most relevant information from a pre-defined set of business documents, and the LLM will generate a contextually accurate response.The system will convert the query into a vector and search a pre-processed database of combining Wikipedia and business blogs as its knowledge base. The chatbot will be designed to efficiently retrieve relevant information and generate accurate and insightful responses to user queries, with a particular focus on improving the quality and specificity of answers within the business domain. These retrieved documents will be passed as context to the GPT model. GPT will process both the user query and the retrieved business documents to generate a response. For this project, a suitable Large Language Model (LLM) such as GPT-4 will be used as the foundation for generating human-like text. However, instead of relying solely on the pre-trained model, we will employ the RAG architecture to enhance the model’s ability to retrieve relevant business-related information from external sources such as Wikipedia and business blogs. It will utilize text embedding models to transform text data into numerical vectors. These embeddings will be stored in a vector database (FAISS) to enable fast and accurate retrieval of relevant documents during chatbot interactions. The retrieval process will involve breaking down large documents (e.g., Wikipedia articles or business blog entries) into smaller, manageable chunks. A chunking strategy (e.g., 300-500 word chunks) will be employed to ensure that the system can efficiently locate and retrieve relevant sections of these documents, maximizing response relevance.

To effectively evaluate the performance of a RAG-based chatbot, it’s crucial to employ robust evaluation metrics that quantify both the accuracy and the quality of the generated responses. The primary goal of these metrics is to ensure that the chatbot’s output is not only factually correct but also relevant and understandable from a human perspective. Two widely recognized metrics—ROUGE and BLEU—are essential in this evaluation process, each focusing on different aspects of response quality.

ROUGE (Recall-Oriented Understudy for Gisting Evaluation) ROUGE is a set of metrics designed to evaluate how well machine-generated summaries or responses overlap with reference answers (or "gold-standard" answers) provided by humans. It is commonly used in tasks that involve text generation, such as summarization, machine translation, and chatbot evaluations. In the context of the RAG-based chatbot, ROUGE will help assess the relevance of the chatbot’s responses. Key Aspects of ROUGE:
1.1 ROUGE-N: This variant looks at the overlap of n-grams (sequences of words) between the generated and reference responses. For instance, ROUGE-1 evaluates the overlap of single words (unigrams), while ROUGE-2 looks at the overlap of two consecutive words (bigrams). Higher overlap indicates a better response, as it captures important words and phrases that should be present in a relevant answer.

1.2 ROUGE-L: This measures the longest common subsequence (LCS) between the generated response and the reference answer. It focuses on the sequential order of words, ensuring that not only are the correct words used, but that they appear in a natural, coherent order that reflects human language patterns.

1.3 Precision and Recall: ROUGE provides a balance between precision (how much of the generated response is relevant) and recall (how much of the relevant information from the reference answer is captured in the generated response). This balance is crucial in ensuring that the chatbot produces answers that are both informative and succinct.

By using ROUGE, we can measure the extent to which the chatbot's generated responses align with human-generated gold-standard answers in terms of content and structure. Higher ROUGE scores imply that the chatbot is effectively capturing the key points and relevant details from its knowledge sources (Wikipedia, blogs) while maintaining a natural flow.

BLEU (Bilingual Evaluation Understudy) BLEU is another popular metric designed to evaluate machine-generated text, particularly in translation and natural language generation tasks. BLEU measures how well the chatbot’s generated responses match human-level quality, with a focus on fluency and grammatical correctness.
Key Aspects of BLEU:

2.1 N-gram Matching: Similar to ROUGE, BLEU also evaluates the overlap of n-grams between the generated and reference answers. However, BLEU places greater emphasis on how well these n-grams match, rewarding both precision (the correct use of phrases) and fluency (natural, human-like sentence structure).

2.2 Weighted N-grams: BLEU scores are calculated by comparing not just single words, but also longer sequences of words (e.g., bigrams, trigrams). This is important for ensuring that the generated text follows logical and grammatical patterns, rather than just using the right keywords. BLEU will penalize the chatbot for missing key multi-word expressions, resulting in a lower score if the response is disjointed or lacks coherence.

2.3 Brevity Penalty: BLEU includes a brevity penalty, which discourages the chatbot from generating overly short answers that omit essential details. This ensures that responses are sufficiently detailed while still being concise.

By using BLEU, we can assess the linguistic quality of the chatbot’s answers, ensuring they are not only factually accurate but also sound natural and are grammatically correct. BLEU, when combined with ROUGE, provides a comprehensive evaluation of both the relevance of content (ROUGE) and the fluency or quality of the language used (BLEU).

Coming onto the step of augmentation, the system will retrieve and rank relevant documents, such as articles or reports about supply chain optimization, to provide relevant context. Right from feeding documents into the LLM to delivering response to the user, it's going to involve several intermediate steps. Let's assume, once the most relevant documents are retrieved from the database, they will be passed to the LLM (e.g., GPT-4). The system will ensure that these documents provide the necessary context for the LLM to generate an informed response. The LLM will then process the user's query in conjunction with the retrieved documents. It will analyze both the query and the contextual information to better understand the user's needs and the specific domain knowledge required to formulate a relevant answer.Based on the retrieved documents and the user query, the LLM will generate an accurate and contextually enriched answer. The response will be tailored to the query, providing a coherent and comprehensive output that draws directly from the most relevant business operations documents. After processing, it will generate a final response that will be delivered to the user. This answer will be both precise and contextually tailored, offering the user a well-rounded, accurate response to their business-related inquiry. The entire bot once ready can be hosted on streamlit so that users can have a clickable local host web url to access the interface.

Hallucinations in LLMS

The term "Hallucination" here involves generating false or misleading information by well-trained LLMs, often due to biases in training data, leading to overconfident and inaccurate outputs. This overconfidence is closely linked to an overreliance on accuracyoriented training. To address these issues, it’s crucial to understand the key considerations and criteria in LLM evaluation.

Dealing with Hallucinations in LLMS

To address LLM hallucinations in the Retrieval Augmented Generation (RAG) pipeline, the system will incorporate a combination of the following practical methods that can be implemented effectively:

Directly grounding answers in retrieved documents: The LLM will be instructed to generate responses only based on the retrieved content, not on any inferred or additional knowledge it might generate. This is might be an approach which limits the model’s ability to "hallucinate" facts that don’t exist in the provided documents.
Answer verification with sources: The system can display retrieved documents or snippets alongside the answer. The user will have the ability to review the source, ensuring transparency.
Implement confidence-based filtering where the system only delivers answers if the model is above a certain confidence level regarding the information's correctness. This can be achieved through calibrating the model and setting thresholds based on evaluation metrics like precision or recall on the training data.

Data Sources:
https://www.bworldonline.com/

Research Motivation Following were some research papers from google scholar which were quite insightful:

https://www.researchgate.net/profile/Reshmi-Sankar/publication/323451431_EMPOWERING_CHATBOTS_WITH_BUSINESS_INTELLIGENCE_BY_BIG_DATA_INTEGRATION/links/5b9351b4299bf14739257a86/EMPOWERING-CHATBOTS-WITH-BUSINESS-INTELLIGENCE-BY-BIG-DATA-INTEGRATION.pdf
https://ieeexplore.ieee.org/abstract/document/9500127
https://dl.acm.org/doi/abs/10.1145/3640794.3665538?casa_token=pv4OYavJppUAAAAA:p-RlA7CnS3fhXRSkI8na57Lh2jI178MNCQtJtUbdkPhqEWJXX5fRVFzXgB0gkTYwn1gF_gR-IQwGgw
https://arxiv.org/abs/2406.16937
https://github.com/Revanth980727/DataAnalysis-Chatbot