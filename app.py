import os
import time
import gradio as gr
import base64
import requests
import json

from openai import OpenAI

api_key = os.getenv('OPENAI_API_KEY')

ebay_price_validation_client = OpenAI(api_key=api_key)

if not api_key:
    raise EnvironmentError("API key for OpenAI not found in environment variables.")

def analyse_image(base64_image):
    response = ebay_price_validation_client.chat.completions.create(
    model="gpt-4-vision-preview",
    messages=[
        {
        "role": "user",
        "content": [
            {"type": "text", "text": "Provide me with a list of all products displayed in the image and suggest search terms I can use to find these products on eBay."},
            {
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/png;base64,{base64_image}"
                },
            },
        ],
        }
    ],
    max_tokens=300,
    )
    return response.choices[0].message.content



def get_products(query):
    api_key = os.getenv("SERPAPI")
    if not api_key:
        print("SERPAPI environment variable is not set.")
        return

    base_url = "https://serpapi.com/search.json"
    params = {
        "engine": "ebay",
        "_nkw": query,
        "ebay_domain": "ebay.com",
        "api_key": api_key,
    }

    try:
        response = requests.get(base_url, params=params)
        response.raise_for_status()

        data = response.json()

        top_3_results = data.get("organic_results", [])[:3]

        markdown_output = ""
        for result in top_3_results:
            title = result.get("title")
            price = result.get("price", {}).get("raw", "N/A")
            link = result.get("link")
            image = result.get("thumbnail")
            markdown_output += f"- **Title:** [{title}]({link})\n"
            markdown_output += f"  - **Price:** {price}\n"
            markdown_output += f"  - **Link:** [View on eBay]({link})\n"
            markdown_output += f"  - **Image:** ![Image]({image})\n\n"
        return markdown_output
    except requests.RequestException as e:
        print(f"Request failed: {e}")
        return

ebay_price_validation_assistant = ebay_price_validation_client.beta.assistants.create(
    name="eBay Price Validator",
    instructions="As an eBay Price Validator, you assist users in estimating the market value of items by finding similar products listed on eBay. You analyze images provided by users to identify products, then search eBay to present similar items and their prices. This helps users gauge how much something is roughly worth. Provide concise, relevant information about similar listings, including price ranges, conditions, and direct links to the listings on eBay.",
    model="gpt-4-1106-preview",
    tools=[{
        "type": "function",
        "function": {
            "name": "search_ebay",
            "description": "Retrieve eBay search results for a given query.",
            "parameters": {
                "type": "object",
                "properties": {
                "query": {
                    "type": "string",
                    "description": "The search query for finding products on eBay."
                }
                },
                "required": ["query"]
            }
        }
    }]
)

conversation_thread = ebay_price_validation_client.beta.threads.create()

def process_query(user_query, interaction_history=[]):
    user_message = ebay_price_validation_client.beta.threads.messages.create(
        thread_id=conversation_thread.id,
        role="user",
        content=user_query
    )

    assistant_response = ebay_price_validation_client.beta.threads.runs.create(
        thread_id=conversation_thread.id,
        assistant_id=ebay_price_validation_assistant.id,
        instructions="The user needs help validating the price of an item on eBay."
    )

    while True:
        time.sleep(0.5)  # Brief pause to allow processing

        response_status = ebay_price_validation_client.beta.threads.runs.retrieve(
            thread_id=conversation_thread.id,
            run_id=assistant_response.id
        )

        if response_status.status == 'requires_action':
            call_functions(response_status.required_action.submit_tool_outputs.model_dump(), assistant_response.id)
        
        if response_status.status == 'completed':
            response_messages = ebay_price_validation_client.beta.threads.messages.list(
                thread_id=conversation_thread.id
            )
            
            data = response_messages.data
            final_response = data[0]
            content = final_response.content
            response = content[0].text.value
            return response

        else:
            continue

def call_functions(required_actions, run_id):
    tool_outputs = []

    for action in required_actions["tool_calls"]:
        func_name = action['function']['name']
        arguments = json.loads(action['function']['arguments'])

        if func_name == "search_ebay":
            output = get_products(arguments['query'])
            tool_outputs.append({
                "tool_call_id": action['id'],
                "output": output 
            })
        else:
            raise ValueError(f"Unknown function: {func_name}")

    ebay_price_validation_client.beta.threads.runs.submit_tool_outputs(
        thread_id=conversation_thread.id,
        run_id=run_id,
        tool_outputs=tool_outputs
    )

def upload_file(file):
    file_path = file.name
    with open(file_path, 'rb') as file:
        file_content = file.read()
    
    encoded_content = base64.b64encode(file_content)
    product_list = analyse_image(encoded_content.decode('utf-8'))
    new_value = "The following similar products were found on eBay:" + product_list

    return new_value, file.name

css = """
#chat {
    height:500px
}
#upload .unpadded_box {
    min-height: 50px;
}
"""

textbox = gr.Textbox(placeholder="Enter your query or upload an image below", lines=2, max_lines=5, label="Query")

with gr.Blocks(css=css, title="eBay Price Validation Tool") as demo:
    with gr.Row(elem_id = "upload"):
        with gr.Column():
            file_output = gr.File()
            upload_button = gr.UploadButton("Upload an Image to Validate Price on eBay", file_types=["image"], file_count="single", scale=0)
            upload_button.upload(upload_file, upload_button, outputs=[textbox, file_output])

    with gr.Row(elem_id = "chat"):
        with gr.Column():
            chat = gr.ChatInterface(
                textbox=textbox,
                fn=process_query,
                undo_btn = None,
                stop_btn = None,
                retry_btn = None,
                clear_btn = None,
            )


if __name__ == "__main__":
    demo.launch()