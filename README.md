# AI-Powered-SMS-Spam-Filter Project Overview and Deployment 
Project Overview

The AI-Powered SMS Spam Filter is a machine learning-based solution designed to automatically classify incoming SMS messages into categories such as Spam or Transactional/Legitimate. The project leverages Natural Language Processing (NLP) techniques, specifically TF-IDF (Term Frequency–Inverse Document Frequency) vectorization, combined with a Logistic Regression classifier, to detect and filter unwanted promotional or fraudulent messages.

To make the system scalable and easy to integrate, the model is deployed as a REST API using FastAPI, containerized with Docker, and can be queried via simple HTTP requests.

Key features include:

1. Lightweight and Fast: Optimized for real-time SMS classification with minimal latency.
2. Customizable Whitelist Mechanism: Users can whitelist trusted keywords, senders, or patterns to prevent false positives.
3. REST API with Swagger UI: Easy-to-use interface for testing and integration at http://127.0.0.1:8000/docs#/default/predict_get_predict_get.
4. Containerized Deployment: Seamless portability and deployment across environments using Docker.

This project is particularly useful for businesses, telecom companies, and applications that need to reduce spam messages, improve user trust, and ensure important transactional messages (e.g., OTPs, alerts, confirmations) are never blocked.

## Steps to Run the Project 
## Docker Deployment

1. Clone the Repository
git clone https://github.com/YashMmodi-7504/AI-Powered-SMS-Spam-Filter
cd sms-spam-classifier

2. Build the Docker Image
docker build -t sms-filter:latest .

3. Run the Docker Container :
docker run -d --name sms-filter -p 8000:8000 sms-filter:latest

4. Access the API :
Open http://127.0.0.1:8000/docs#/default/predict_get_predict_get
 in your browser.
You will see an interactive Swagger UI to test predictions.

5. Send a Sample Request
curl -X POST "http://127.0.0.1:8000/predict" \
-H "Content-Type: application/json" \
-d '{"message": "Congratulations! You have won a free voucher"}'

   Response Example:
  {
    "message": "Congratulations! You have won a free voucher",
    "prediction": "Spam"
  }

## Render Deployment
Base URL: https://ai-powered-sms-spam-filter-for-a2p-sms.onrender.com

Docs: https://ai-powered-sms-spam-filter-for-a2p-sms.onrender.com/docs#/default/predict_get_predict_get

## Streamlit Deployment
App URL: https://ai-powered-sms-spam-filter.streamlit.app/

## Demo Loom Video
Loom Link : https://www.loom.com/share/3863e653122241f290a130e4f6438f9d?sid=23b4f17e-73ef-4232-80c7-7b68b38ccc0e
