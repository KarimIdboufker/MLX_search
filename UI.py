import streamlit as st
import requests

# Set the URL of the FastAPI server
FASTAPI_URL = "http://37.27.223.55:8000/"  # Update if hosted on a different URL or port

# Streamlit UI
st.title("Document Recommendation System")
st.write("Enter a query to receive top document recommendations.")

# Input box for the query
query = st.text_input("Query")

# Submit button
if st.button("Get Recommendations"):
    if query:
        # Send request to FastAPI server
        response = requests.post(FASTAPI_URL, json={"query": query})
        
        if response.status_code == 200:
            # Display top recommended documents
            recommendations = response.json().get("top_documents", [])
            st.write("Top Recommended Documents:")
            for doc_id in recommendations:
                st.write(f"- Document ID: {doc_id}")
        else:
            st.error("Error retrieving recommendations. Please try again.")
    else:
        st.warning("Please enter a query.")