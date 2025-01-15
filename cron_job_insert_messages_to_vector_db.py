from pymongo import MongoClient
import pandas as pd
from datetime import datetime, timedelta
from bson import ObjectId
from pinecone import Pinecone
from tqdm import tqdm
import numpy as np
import openai
import os
import asyncio
from dotenv import load_dotenv

load_dotenv()

# MongoDB connection
client = MongoClient(os.getenv('MONGO_DB_URL'))
db = client[os.getenv('DB_NAME')]
pinecone = Pinecone(api_key=os.getenv('PINECONE_API_KEY'))
openai.api_key = os.getenv('OPENAI_API_KEY')
index = pinecone.Index(os.getenv('PINECONE_INDEX'))

# Assuming you have these collections available
messages_collection = db["messages"]
users_collection = db["users"]
channels_collection = db["channels"]
conversations_collection = db["conversations"]
workspaces_collection = db["workspaces"]

# Function to chunk messages
def chunk_messages(messages, chunk_size=100):
    chunk = []
    for message in messages:
        chunk.append(message)
        if len(chunk) >= chunk_size:
            yield chunk
            chunk = []
    if chunk:
        yield chunk

async def vectorize_message(content: str):
    try:
        response = await openai.Embedding.acreate(
            model="text-embedding-3-large",
            input=content,
        )
        return response["data"][0]["embedding"]
    except Exception as e:
        print(f"Error in vectorization: {e}")
        return []

async def get_user_details(user_id):
    user = users_collection.find_one({"_id": user_id})
    if user:
        return {
            "senderId": str(user["_id"]),
            "senderUsername": user["username"],
            "senderDisplayName": user.get("displayName"),
            "senderProfile": {
                "tone": user.get("profile", {}).get("tone", "neutral"),
                "formality": user.get("profile", {}).get("formality", "neutral")
            }
        }
    return None

async def get_container_details(message):
    container_info = {
        "containerId": "",
        "containerType": "",
        "containerName": "",
        "workspaceId": ""
    }

    if message.get("threadId"):
        # This is a thread message - get parent message to find container
        parent_message = messages_collection.find_one({"_id": message["threadId"]})
        if parent_message:
            # Recursively get container details from parent message
            return await get_container_details(parent_message)
    
    if message.get("channelId"):
        channel = channels_collection.find_one({"_id": message["channelId"]})
        if channel:
            container_info.update({
                "containerId": str(channel["_id"]),
                "containerType": "channel",
                "containerName": channel["name"],
                "workspaceId": str(channel["workspaceId"])
            })
    elif message.get("conversationId"):
        conversation = conversations_collection.find_one({"_id": message["conversationId"]})
        if conversation:
            # Get workspaces for both participants
            participants = conversation["participants"]
            participant_workspaces = set()
            
            for participant_id in participants:
                user = users_collection.find_one(
                    {"_id": participant_id},
                    {"workspaces": 1}
                )
                if user and user.get("workspaces"):
                    participant_workspaces.update(user["workspaces"])
            
            # If participants share workspaces, use the first shared workspace
            # You might want to adjust this logic based on your specific needs
            shared_workspace = next(iter(participant_workspaces)) if participant_workspaces else None
            
            container_info.update({
                "containerId": str(conversation["_id"]),
                "containerType": "conversation",
                "workspaceId": str(shared_workspace) if shared_workspace else None
            })
    
    return container_info

async def get_message_context(message_id):
    # Find previous and next messages in the same container
    current_message = messages_collection.find_one({"_id": message_id})
    if not current_message:
        return None, None

    previous_message = messages_collection.find_one({
        "$or": [
            {"channelId": current_message.get("channelId")},
            {"conversationId": current_message.get("conversationId")}
        ],
        "createdAt": {"$lt": current_message["createdAt"]}
    }, sort=[("createdAt", -1)])

    if previous_message and '_id' not in previous_message:  # Check if _id field is missing
        breakpoint()

    next_message = messages_collection.find_one({
        "$or": [
            {"channelId": current_message.get("channelId")},
            {"conversationId": current_message.get("conversationId")}
        ],
        "createdAt": {"$gt": current_message["createdAt"]}
    }, sort=[("createdAt", 1)])

    return (str(previous_message["_id"]) if previous_message else "",
        str(next_message["_id"]) if next_message else "")

def extract_mentioned_users(content):
    # Simple @username extraction - you might want to make this more sophisticated
    mentioned = []
    words = content.split()
    for word in words:
        if word.startswith("@"):
            mentioned.append(word[1:])
    return mentioned

async def process_messages(messages, message_type):
    for chunk in chunk_messages(messages):
        for message in chunk:
            print(f"Processing message: {str(message['_id'])}")
            
            # Get vector for individual message
            vector = await vectorize_message(message["content"])
            print(vector)
            
            # Get additional context
            user_details = await get_user_details(message["user"])
            container_details = await get_container_details(message)
            prev_msg_id, next_msg_id = await get_message_context(message["_id"])
            
            try:
                index.upsert(
                    vectors=[{
                        "id": str(message["_id"]),
                        "values": vector,
                        "metadata": {
                            # Message Core
                            "messageId": str(message["_id"]),
                            "content": message["content"],
                            "timestamp": message.get("createdAt", datetime.now()).isoformat(),
                            "type": message["type"],
                            "edited": message.get("edited", False),
                            
                            
                            # Thread Context
                            "isThreadMessage": bool(message.get("threadId")),
                            "threadId": str(message["threadId"]) if message.get("threadId") else "",
                            "parentMessageId": str(message["threadId"]) if message.get("threadId") else "",  # Same as threadId for clarity
                            # User Context
                            # **user_details,
                            # User Context
                            "senderId": user_details.get("senderId", ""),
                            "senderUsername": user_details.get("senderUsername", ""),
                            "senderDisplayName": user_details.get("senderDisplayName", ""),
                            "senderProfileTone": user_details.get("senderProfile", {}).get("tone", "neutral"),
                            "senderProfileFormality": user_details.get("senderProfile", {}).get("formality", "neutral"),

                            # Container Context
                            "containerId": container_details.get("containerId", ""),
                            "containerType": container_details.get("containerType", ""),
                            "containerName": container_details.get("containerName", ""),
                            "workspaceId": container_details.get("workspaceId", ""),
                            
                            
                            # Additional Context
                            "previousMessageId": prev_msg_id,
                            "nextMessageId": next_msg_id,
                            "mentionedUsers": extract_mentioned_users(message["content"]),
                            "attachmentTypes": [att["type"] for att in message.get("attachments", [])]
                        }
                    }]
                )
            except Exception as e:
                print(f"Error upserting message: {e}")

# Rest of your code remains the same...

# Fetch messages grouped by type
def fetch_messages_by_type():
    # Calculate timestamp for 24 hours ago
    twenty_four_hours_ago = datetime.now() - timedelta(hours=24)
    
    # Add timestamp filter to queries
    conversation_messages = messages_collection.find({
        "type": "conversation",
        "createdAt": {"$gte": twenty_four_hours_ago}
    })
    
    channel_messages = messages_collection.find({
        "type": "channel",
        "createdAt": {"$gte": twenty_four_hours_ago}
    })
    
    return conversation_messages, channel_messages


# Main function to process all messages
async def process_all_messages():
    conversation_messages, channel_messages = fetch_messages_by_type()
    
    # Convert cursors to lists and get counts
    conversation_list = list(conversation_messages)
    channel_list = list(channel_messages)
    
    print(f"Found {len(conversation_list)} conversation messages from last 24 hours")
    print(f"Found {len(channel_list)} channel messages from last 24 hours")

    print("Processing conversation messages...")
    await process_messages(conversation_list, "conversation")

    print("Processing channel messages...")
    await process_messages(channel_list, "channel")


# Run the script
if __name__ == "__main__":
    asyncio.run(process_all_messages())