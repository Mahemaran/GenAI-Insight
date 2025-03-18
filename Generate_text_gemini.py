from google import genai

client = genai.Client(api_key="AIzaSyAHdrg4G4Ves-qh8GcNfwh0FiPIGRCsGic")
failure = ""
AC = ""

prompt = f'''

No more explanation
'''
response = client.models.generate_content(model="gemini-2.0-flash", contents=prompt)
print(response.text)