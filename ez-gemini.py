import requests, time
from collections import defaultdict
import google.generativeai as genai
import os

if __name__ == "__main__": os.chdir(os.path.dirname(__file__))

class color:
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
    def __init__(self, api_key, proxies = None):
        self.api_key = api_key
        self.proxies = proxies
        self.base_url = "https://generativelanguage.googleapis.com/v1beta/models/{}/:generateContent?key={}"
        genai.configure(api_key = api_key)

    def transform_messages(self, messages):
        """
        Transform a list of messages into a specific format required for the Generative AI API.

        Parameters:
        - messages (list): A list of dictionaries, where each dictionary represents a message. Each message dictionary must have 'role' and 'content' keys.

        Returns:
        - dict: A dictionary with a single key 'contents', which maps to a list of dictionaries. Each dictionary in this list represents a message and has 'role' and 'parts' keys. The 'role' key maps to a string representing the role of the message sender, and the 'parts' key maps to a list containing a dictionary with a 'text' key representing the content of the message. The function also modifies the 'role' key to match the expected format for the Generative AI API.
        """
        transformed_dict = defaultdict(list)
        
        for msg in messages:
            role = msg["role"]
            content = msg["content"]
            
            transformed_dict["contents"].append({
                "role": role,
                "parts": [{"text": content}]
            })
        
        for item in transformed_dict["contents"]:
            if item["role"] == "assistant":
                item["role"] = "model"
            
            elif item["role"] == "system":
                item["role"] = "user"
        
        return dict(transformed_dict)

    def generate_answer(self, prompt, model = 'gemini-1.5-pro', files = [], proxy = None):
        """
        Generate a response using the Google Generative AI API.

        Parameters:
        - prompt (list): A list of dictionaries, where each dictionary represents a message. Each message dictionary must have 'role' and 'content' keys.
        - model (str, optional): The model name to use for generation. Defaults to 'gemini-1.5-pro'.
        - files (list, optional): A list of file paths to be included in the prompt. Defaults to [].
        - temperature (float, optional): A value between 0 and 1 that controls the creativity of the generated response. Defaults to 0.7.
        - max_tokens (int, optional): The maximum number of tokens to generate in the response. Defaults to 1000.

        Returns:
        - str: The generated response as a string.
        """
        
        
        gemini_urls = []
        gemini_mimes = []
        
        for file in files:
            file = genai.upload_file(file)
            gemini_urls.append(file.uri)
            gemini_mimes.append(file.mime_type)
        
        history = self.transform_messages(prompt)
        
        for uri, mime in zip(gemini_urls, gemini_mimes):
            history['contents'][0]['parts'].append(
                {
                    "fileData": {
                        "mimeType": mime,
                        "fileUri": uri
                    }
                }
            )

        response = requests.post(
            self.base_url.format(model, self.api_key),
            json=history,
            headers={'Content-Type': 'application/json'},
            proxies=self.proxies if not proxy else proxy
        )
        
        if response.status_code == 200:
            return response
        elif response.status_code == 429 or response.status_code == 500:
            response = requests.post(
            self.base_url.format(model, self.api_key),
            json=history,
            headers={'Content-Type': 'application/json'},
            proxies=self.proxies if not proxy else proxy
            )
            
            return response
        else:
            raise GenerationError(response.text, response.status_code)
    
    def test_proxy(self, proxy, prompt = [{"role": "user", "content": "I love flowers"}, {"role": "user", "content": "What am i love?"}]):
        """
        Test if a proxy is working by making a request to a test URL.

        Parameters:
        - proxy (dict): A dictionary with the proxy configuration.

        Returns:
        - bool: True if the proxy works, False otherwise.
        """
        proxies = {"http": f"http://{proxy}", "https": f"https://{proxy}"}
    
        for protocol, proxy_url in proxies.items():
            response = None
            try:
                GenerativeAI(
                    self.api_key,
                    {protocol: proxy_url}
                )
                
                response = self.generate_answer(prompt)
                
                if response.json().get('candidates', False):
                    return response.status_code, response.json()['candidates'][0]['content']['parts'][0]['text']
                else: return response.status_code, response.json()
            
            except Exception:
                if response:
                    return response.status_code, None
                else: return None, None
    
    def test_proxies(
        self, 
        proxies: list, 
        prompt = [
            {"role": "user", "content": "I love flowers"}, 
            {"role": "user", "content": "What am i love?"}
        ]):
        
        working = []
        
        print(f'{color.BLUE}{color.BOLD}--- PROXY TEST TOOL ---{color.END}')
        print(f'Testing on {prompt} prompt')
        print(f'{color.BLUE}{len(proxies)}{color.END} to test...\n\n\n')
        
        for i, proxy in enumerate(proxies):
            address, port = proxy.split(':')
            
            print(f'{color.BLUE}{color.BOLD}Testing{color.END} {i + 1}rd {color.GREEN}{color.BOLD}{address}:{port}{color.END} proxy...')
            
            def check():
                status, answer = self.test_proxy(address, prompt)
                
                if status == 200:
                    print(f"{color.GREEN}{color.BOLD}--- OK. Code {status}{color.END}. This proxy is suitable for Gemini free API. Gemini Response: {answer}\n")
                    working.append(proxy)
                    return True
                elif status == 500:
                    return check()
                elif status == 429:
                    print(f'{color.YELLOW}{color.BOLD}Oh, we are rate-limited! Retrying in 10 seconds...\n\n')
                    return False
                else:
                    print(f'{color.RED}{color.BOLD}--- Failed. Code {status}{color.END}. Proxy is not suitable for Gemini free API.\n\n')
                    return True
            
            if not check(): time.sleep(10); check()
        
        print(f'{color.GREEN}{color.BOLD}--- TEST COMPLETED ---{color.END}')
        print(f'{color.GREEN}{color.BOLD}Working {len(working)} proxies from all {len(proxies)}. Tool returned a list of working proxies.')
        
        return working