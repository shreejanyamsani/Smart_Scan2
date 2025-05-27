import streamlit as st

class StreamlitLogger:
    """Custom logger to capture print statements and display them in Streamlit."""
    def __init__(self):
        self.text = []
    
    def write(self, message):
        self.text.append(message)
        st.write(message)
    
    def flush(self):
        pass
    
    def get_logs(self):
        return "\n".join(self.text)