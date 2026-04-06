import asyncio
from services.llm.gemini import GeminiProvider

async def test():
    provider = GeminiProvider()
    
    with open("test_image.png", "rb") as f:
        image_bytes = f.read()
    
    print("Sending to Gemini Vision...")
    result = await provider.analyze_image(image_bytes, "zero_shot")
    
    # Save full response to file
    with open("gemini_response.txt", "w") as f:
        f.write(result.raw_text)
    
    print("Full response saved to gemini_response.txt")
    print("Response length:", len(result.raw_text))
    print("Parsed JSON:", result.parsed_json)

asyncio.run(test())