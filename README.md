# MultiAgent Nexus 2: LangGraph-Based Multi-Agent Workflow System

This project is a LangGraph-based application that implements a multi-agent workflow system. It leverages LangGraph, LangChain, and Groq's language models to create a dynamic and interactive workflow where different agents (Prompt Enhancer, Researcher, Coder, and Validator) collaborate to process user queries efficiently. The system is designed to handle tasks ranging from query clarification and information gathering to technical problem-solving and validation.

---

## Features

- **Multi-Agent Workflow**: Utilizes a team of specialized agents (Enhancer, Researcher, Coder, and Validator) to process user queries.
- **Dynamic Routing**: The Supervisor agent dynamically routes tasks to the most appropriate agent based on the query's requirements.
- **Integration with External Tools**:
  - **Tavily Search**: For gathering information from the web.
  - **Riza Code Interpreter**: For executing Python code and solving technical problems.
- **Streamlit Interface**: Provides an interactive web interface for users to input queries and view results.
- **Structured Outputs**: Uses Pydantic models to ensure structured and consistent outputs from the language model.

---

## Installation

1. **Set up a virtual environment**:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows use `venv\Scripts\activate`
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Set up environment variables**:
   Create a `.env` file in the root directory and add your API keys:
   ```plaintext
   GROQ_API_KEY=your_groq_api_key
   TAVILY_API_KEY=your_tavily_api_key
   RIZA_API_KEY=your_riza_api_key
   ```

---

## Running the Application

To start the Streamlit application, run the following command:

```bash
streamlit run app.py
```

The application will be available at `http://localhost:8501`.

---

## Workflow Overview

1. **Supervisor**: Evaluates the user query and decides which agent should handle it next.
2. **Enhancer**: Refines vague or incomplete user queries to make them more precise.
3. **Researcher**: Gathers information using the Tavily Search tool.
4. **Coder**: Executes Python code or solves technical problems using the Riza Code Interpreter.
5. **Validator**: Reviews the final output to ensure it addresses the user's query satisfactorily.

---

## Example Usage

1. Open the Streamlit app in your browser.
2. Enter a query in the input box (e.g., "What is the capital of France?" or "Calculate the square root of 144").
3. Click the "Process Query" button.
4. View the step-by-step outputs from each agent and the final answer.

---

## Code Structure

- **app.py**: The main Streamlit application script.
- **Supervisor, Validator, and Agent Nodes**: Defined using Pydantic models and LangGraph nodes.
- **Tools**:
  - `TavilySearchResults`: For web searches.
  - `ExecPython`: For executing Python code.
- **Streamlit Interface**: Handles user input and displays outputs.

---

## API Keys

- **Groq API Key**: Required for interacting with Groq's language models (https://console.groq.com/keys).
- **Tavily API Key**: Required for using the Tavily Search tool (https://app.tavily.com/home).
- **Riza API Key**: Required for using the Riza Code Interpreter (https://riza.io/playground).

---

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

---

## Acknowledgments

- [Streamlit](https://streamlit.io/)
- [LangChain](https://www.langchain.com/)
- [LangGraph](https://langchain.com/langgraph)
- [Groq](https://groq.com/)
- [Tavily](https://tavily.com/)
- [Riza](https://riza.io/)

---

This README provides a comprehensive guide to setting up, running, and using the MultiAgent Nexus 2 application. For further details, refer to the code comments and documentation links provided.
