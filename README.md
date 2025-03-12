# Legal Assistant API

This project is a REST API for a legal assistant application that interacts with the Indian Kanoon API to provide legal information and assistance. The API allows users to submit legal queries and receive relevant case summaries and analyses.

## Project Structure

```
legal-assistant-api
├── src
│   ├── legal_assistant
│   │   ├── __init__.py
│   │   ├── legal_assistant.py
│   │   └── api.py
│   └── app.py
├── requirements.txt
├── .env
└── README.md
```

## Setup Instructions

1. **Clone the repository:**
   ```
   git clone <repository-url>
   cd legal-assistant-api
   ```

2. **Create a virtual environment:**
   ```
   python -m venv venv
   source venv/bin/activate  # On Windows use `venv\Scripts\activate`
   ```

3. **Install dependencies:**
   ```
   pip install -r requirements.txt
   ```

4. **Set up environment variables:**
   Create a `.env` file in the root directory and add the necessary API keys and configuration settings. Example:
   ```
   CONVEX_URL=<your_convex_url>
   GEMINI_API_KEY=<your_gemini_api_key>
   INDIAN_KANOON_API_KEY=<your_indiankanoon_api_key>
   ```

## Usage

1. **Run the application:**
   ```
   python src/app.py
   ```

2. **Access the API:**
   The API will be available at `http://localhost:8000` (or the port specified in your app configuration). You can use tools like Postman or curl to interact with the API.

3. **Example Request:**
   To submit a legal query, send a POST request to the `/query` endpoint with the following JSON body:
   ```json
   {
       "query": "What is the procedure for filing a divorce?"
   }
   ```

4. **Example Response:**
   The API will respond with a JSON object containing the analysis and relevant case summaries.

## Contributing

Contributions are welcome! Please open an issue or submit a pull request for any improvements or bug fixes.

## License

This project is licensed under the MIT License. See the LICENSE file for more details.