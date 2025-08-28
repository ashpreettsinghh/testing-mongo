from pymongo import MongoClient
import requests

MONGO_URI = "mongodb+srv://:@..mongodb.net/?retryWrites=true&w=majority&appName=JobSleuth"
# 1. Connect to MongoDB Atlas
client = MongoClient(MONGO_URI)  # Replace with your MongoDB Atlas URI
db = client["devjobs"]
collection = db["jobs"]

# 2. Fetch all documents
# docs = collection.find({}).sort("_id", 1)
#
# # 3. Loop through and update each document
# for doc in docs:
#     description = doc['description']
#     json_data = {
#         "modelName": "Qwen/Qwen3-Embedding-8B",
#         "apiKey": "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",
#         "inputText": f"{description}"
#     }
#
#     response = requests.post(url="https://ai--iypm.onrender.com/vector-embedding/get-embedding",json=json_data)
#     # Example: adding a new field 'status' with value 'active'
#     vector = response.json()['Vector']
#     collection.update_one(
#         {"_id": doc["_id"]},  # find document by its unique _id
#         {"$set": {"embedding": f"{vector[0]}"}}  # add/update the new field
#     )
#     print(doc)
#
# print("✅ All documents updated successfully!")
#


# from pymongo.operations import SearchIndexModel
# # Create your index model, then create the search index
# search_index_model = SearchIndexModel(
#   definition = {
#     "fields": [
#       {
#         "type": "vector",
#         "path": "embedding",
#         "similarity": "cosine",
#         "numDimensions": 4096
#       }
#     ]
#   },
#   name="vector_index",
#   type="vectorSearch"
# )
# collection.create_search_index(model=search_index_model)


# step1
desc = "Coins is looking for a Senior Project Manager to lead and manage key projects, ensuring successful delivery on time and with excellent quality. They will collaborate with cross-functional teams and drive process improvements to contribute to the overall success of the product and services."
json_data = {
        "modelName": "Qwen/Qwen3-Embedding-8B",
        "apiKey": "",
        "inputText": f"{desc}"
    }

response = requests.post(url="https://ai--iypm.onrender.com/vector-embedding/get-embedding",json=json_data)
# Example: adding a new field 'status' with value 'active'
vector = response.json()['Vector']

# 2. Your query embedding (get from API)
query_embedding = vector[0]  # replace with your API call

# 3. Run vector similarity search
pipeline = [
    {
        "$vectorSearch": {
            "queryVector": query_embedding,
            "path": "embedding",
            "numCandidates": 300,
            "limit": 5,
            "index": "vector_index"
        }
    },
    {
        "$project": {
            "description": 1,
            "score": {"$meta": "vectorSearchScore"}
        }
    }
]

results = list(collection.aggregate(pipeline))

# 4. Print results
for r in results:
    print(f"Doc ID: {r['_id']}, Score: {r['score']:.4f}, Desc: {r['description']}")


