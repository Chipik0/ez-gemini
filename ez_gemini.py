import os
import nltk
import time
import requests
import asyncio
import aiohttp
from collections import defaultdict
import google.generativeai as genai

if __name__ == "__main__": os.chdir(os.path.dirname(__file__))

class Color:
    PURPLE = '\033[95m'
    CYAN = '\033[96m'
    DARKCYAN = '\033[36m'
    BLUE = '\033[94m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'
    END = '\033[0m'

class GenerationError(Exception):
    def __init__(self, message, code):
        self.message = message
        self.code = code
        super().__init__(message)

class GenerativeAI:
    def __init__(self, api_key, proxies=None):
        self.api_key = api_key
        self.proxies = proxies
        self.base_url = "https://generativelanguage.googleapis.com/v1beta/models/{}/:generateContent?key={}"
        genai.configure(api_key=api_key)
    
    def count_tokens(self, history):
        total_tokens = 0
        for message in history:
            try:
                total_tokens += len(nltk.tokenize.word_tokenize(message.get('content', '')))
            
            except LookupError:
                nltk.download('punkt_tab')
                total_tokens += len(nltk.tokenize.word_tokenize(message.get('content', '')))
        
        return total_tokens
    
    def _upload_file(self, filename):
        """
        Uploads a file to the Generative AI model and returns the dict with URL and MIME type.
        """
        
        uploaded = genai.upload_file(filename)
        return {"fileUri": uploaded.uri, "mimeType": uploaded.mime_type}
    
    def _disable_security(self, history: dict):
        history['safetySettings'] = [
            {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_NONE"},
            {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_NONE"},
            {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_NONE"},
            {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_NONE"}
        ]
        
        return history

    def _transform_messages(self, history):
        result = {'contents': []}
        for entry in history:
            transformed_entry = {'role': entry['role'], 'parts': []}

            if 'content' in entry:
                transformed_entry['parts'].append({'text': entry['content']})

            if 'media' in entry:
                for media in entry['media']:
                    if isinstance(media, str):
                        media = self._upload_file(media)
                    transformed_entry['parts'].append({'fileData': media})

            result['contents'].append(transformed_entry)

        return result
    
    async def generate_answer_async(self, prompt, model='gemini-1.5-flash', system_prompt: str = None, proxy: dict = None):
        """
        Generates a response using the Google Generative AI API asynchronously.

        ### Parameters:
        - prompt (list): List of dictionaries with 'role' and 'content'.
        - model (str, optional): Model name for generation. Defaults to 'gemini-1.5-flash'.
        - system_prompt (str, optional): System prompt. Defaults to None.
        - proxy (dict, optional): Proxy configuration for the request.

        ### Returns:
        - dict: The generated response as a JSON.
        """

        history = self._transform_messages(prompt)
        history = self._disable_security(history)
        
        if system_prompt:
            history["system_instruction"] = {
                "parts": { "text": system_prompt}
            }

        async def make_request():
            async with aiohttp.request(
                method='POST',
                url=self.base_url.format(model, self.api_key),
                json=history,
                headers={'Content-Type': 'application/json'},
                proxy=proxy
            ) as response:
                
                return response
        
        response = await make_request()

        if response.status == 200:
            return await response.json()

        elif response.status in (429, 500):
            await asyncio.sleep(2)
            retry_response = await make_request()

            if retry_response.status == 200:
                return await retry_response.json()
                
            else:
                raise GenerationError(await retry_response.text(), retry_response.status)
        
        else:
            raise GenerationError(await response.text(), response.status)

    def generate_answer(self, prompt, model = 'gemini-1.5-flash', system_prompt: str = None, proxy: dict = None):
        """
        Generates a response using the Google Generative AI API.

        ### Parameters:
        - prompt (list): List of dictionaries with 'role' and 'content'.
        - model (str, optional): Model name for generation. Defaults to 'gemini-1.5-flash'.
        - system_prompt (str, optional): System prompt. Defaults to None.
        - proxy (dict, optional): Proxy configuration for the request.

        ### Returns:
        - dict: The generated response as a JSON.
        """

        history = self._transform_messages(prompt)
        history = self._disable_security(history)
        
        if system_prompt:
            history["system_instruction"] = {
                "parts": { "text": system_prompt}
            }
        
        def make_request():
            return requests.post(
                self.base_url.format(model, self.api_key),
                json=history,
                headers={'Content-Type': 'application/json'},
                proxies=self.proxies if not proxy else proxy
            )
        
        response = make_request()
        
        if response.status_code == 200:
            return response.json()
        
        elif response.status_code in (429, 500):
            time.sleep(2)
            
            response = make_request()
            
            if response.status_code == 200:
                return response.json()
            
            else:
                raise GenerationError(response.text, response.status_code)
        else:
            raise GenerationError(response.text, response.status_code)
    
    def test_proxy(self, proxy, prompt = None):
        """
        Tests if a proxy is working by making a request to a test URL.

        ### Parameters:
        - proxy (str): Proxy address in the format 'IP:PORT'.
        For example:
        "0.0.0.0:0
        1.1.1.1:1
        ..."
        - prompt (list, optional): List of messages for the test.

        ### Returns:
        - tuple: Status code and the response text if successful; otherwise, None.
        """
        if prompt is None:
            prompt = [
                {"role": "user", "content": "I love flowers"},
                {"role": "user", "content": "What am I love?"}
            ]

        proxies = {"http": f"http://{proxy}", "https": f"https://{proxy}"}
        response = None

        try:
            genai_instance = GenerativeAI(self.api_key, proxies)
            response = genai_instance.generate_answer(prompt)
            if 'candidates' in response.json():
                return response.status_code, response.json()['candidates'][0]['content']['parts'][0]['text']
        
        except Exception:
            pass
        
        return response.status_code if response else None, None
    
    def test_proxies(self, proxies, prompt = None):
        """
        Tests a list of proxies and returns the working ones.

        Parameters:
        - proxies (list): List of proxies to test.
        - prompt (list, optional): List of messages for the test.

        Returns:
        - list: List of working proxies.
        """
        if prompt is None:
            prompt = [
                {"role": "user", "content": "I love flowers"},
                {"role": "user", "content": "What am I love?"}
            ]

        working_proxies = []
        
        print(f'{Color.BLUE}{Color.BOLD}--- PROXY TEST TOOL ---{Color.END}')
        print(f'Testing with prompt: {prompt}')
        print(f'{Color.BLUE}{len(proxies)} proxies to test...\n\n')

        for i, proxy in enumerate(proxies):
            address, port = proxy.split(':')
            print(f'{Color.BLUE}{Color.BOLD}Testing {i + 1} proxy: {Color.GREEN}{Color.BOLD}{address}:{port}{Color.END}...')

            def check():
                status, answer = self.test_proxy(proxy, prompt)
                if status == 200:
                    print(f"{Color.GREEN}{Color.BOLD}--- OK. Code {status}{Color.END}. Proxy is working. Response: {answer}\n")
                    working_proxies.append(proxy)
                    return True
                
                elif status == 429:
                    print(f'{Color.YELLOW}{Color.BOLD}Rate-limited! Retrying in 10 seconds...\n')
                    time.sleep(10)
                    return check()
                
                else:
                    print(f'{Color.RED}{Color.BOLD}--- Failed. Code {status}{Color.END}. Proxy is not working.\n')
                    return True

            check()

        print(f'{Color.GREEN}{Color.BOLD}--- TEST COMPLETED ---{Color.END}')
        print(f'{Color.GREEN}{Color.BOLD}Working proxies: {len(working_proxies)} out of {len(proxies)} tested.{Color.END}')
        
        print(working_proxies)
        return working_proxies