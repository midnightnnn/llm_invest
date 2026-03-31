import os
from google import genai
from dotenv import load_dotenv

load_dotenv()

location = os.getenv("GOOGLE_CLOUD_LOCATION", "global")
if location == "global":
    location = "asia-northeast3" # embeddings may not be available in global

client = genai.Client(
    vertexai=True,
    project=os.getenv("GOOGLE_CLOUD_PROJECT"),
    location=location,
)

response = client.models.embed_content(
    model="text-embedding-004",
    contents="Apple earnings are out tomorrow.",
)

print("Embedding dimension:", len(response.embeddings[0].values))
print("First 5 values:", response.embeddings[0].values[:5])

