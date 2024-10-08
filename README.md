# EZ - Gemini
#### The most convenient API handler for interacting with Gemini. This library will allow you to use Gemini GenerativeAI absolutely for free, in any country, thanks to the proxy. (You need to specify the proxy address yourself in code.)

## Setup
Just download `ez_gemini.py` and use it as library.

**You need to get free API Key from Google Cloud**
Go to [AI Google Studio](https://aistudio.google.com/app/apikey) and generate an API key, if you do not have a Google Cloud account or project you must first create one.<br/><br/>
! - **Note that you will not be billed unless you click “Set up billing” even if billing is enabled in Google Cloud.**<br/>
! - **Make sure to never share your API key and ensure that it is excluded from your public repository in GitHub.**<br/>

## Examples
- Use this code in the begin of your code.
```
from ez_gemini import GenerativeAI

model = GenerativeAI('API_KEY')
```
##
- How do i just ask Gemini something?
Just use `model.generate_answer()` with a standard history system like in OpenAI.
```
result = model.generate_answer([
    {"role": "user", "content": "Hello"},
    {"role": "user", "content": "Do you love chips?"}
])

print(result['candidates'][0]['content']['parts'][0]['text'])
```
##
- How do i add file(-s) to prompt?
Also use `model.generate_answer()`, but with `files: list` argument.

```
result = model.generate_answer([
    {"role": "user", "content": "Hello"},
    {
        "role": "user", "content": "What's on this files?", 
        "media": ["path/to/file1", "path/to/file2"]
    }
    ]
)

print(result['candidates'][0]['content']['parts'][0]['text'])
```
##
Also you can change model using `model` argument in `model.generate_answer()`<br/>
**Current list of available models:**
- gemini-1.0-pro-latest
- gemini-1.0-pro
- gemini-pro
- gemini-1.0-pro-001
- gemini-1.0-pro-vision-latest
- gemini-pro-vision
- gemini-1.5-pro-latest
- gemini-1.5-pro-001
- gemini-1.5-pro
- gemini-1.5-pro-exp-0801
- gemini-1.5-flash-latest
- gemini-1.5-flash-001
- gemini-1.5-flash
- gemini-1.5-flash-001-tuning

## ToDo
- The ability to install 2 API keys to increase the rate limit.
- Repeating the request in case of an error. - DONE
- API Key Tutorial. - DONE