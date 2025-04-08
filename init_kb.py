import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
from openai import OpenAI
import json
import logging
import argparse
from pymilvus import (
    connections,
    Collection,
    utility,
    FieldSchema,  
    CollectionSchema, 
    DataType 
)

key = 'sk-bOBUTf38jTU2TDza2nz1cAR8QETryCcI2W1vGTNmZybyUZjL'

client = OpenAI(
    base_url="https://api2.aigcbest.top/v1",
    api_key=key
)

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

def init_knowledge_base(milvus_host: str, milvus_port: int, knowledge_file: str):
    try:
        # 连接 Milvus
        logging.info(f"Connecting to Milvus at {milvus_host}:{milvus_port}")
        connections.connect(host=milvus_host, port=milvus_port)

        # 读取知识库文件
        logging.info(f"Loading knowledge file: {knowledge_file}")
        with open(knowledge_file, 'r', encoding='utf-8') as f:
            data = json.load(f)['knowledge_base']

        # 创建集合
        collection_name = "ros_knowledge"

        
        if utility.has_collection(collection_name):
            utility.drop_collection(collection_name)

        if not utility.has_collection(collection_name):
            # 定义字段模式
            fields = [
                FieldSchema(name="id", dtype=DataType.INT64, is_primary=True),
                FieldSchema(name="title", dtype=DataType.VARCHAR, max_length=200),
                FieldSchema(name="content", dtype=DataType.VARCHAR, max_length=2000),
                FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=1536)
            ]
            
            # 创建集合模式
            schema = CollectionSchema(
                fields,
                description="ROS2 Knowledge Base"
            )
            
            # 创建集合
            Collection(name=collection_name, schema=schema)
            logging.info(f"Collection {collection_name} created")


        # 准备数据
        collection = Collection(collection_name)
        entities = []
        for item in data:
            combined_text = f"{item['title']} {item['content']}"
            
           
            try:
                response = client.embeddings.create(
                    input=combined_text,
                    model="text-embedding-3-small"
                )
                embedding = response.data[0].embedding
            except Exception as e:
                logging.error(f"Failed to generate embedding: {str(e)}")
                continue
            
            entities.append({
                "id": item['id'],
                "title": item['title'],
                "content": item['content'],
                "embedding": embedding
            })

        # 插入数据
        logging.info(f"Inserting {len(entities)} entries...")
        if entities:
            collection.insert(entities)
            collection.flush()
        
        # 创建索引
        index_params = {
            "metric_type": "L2",
            "index_type": "IVF_FLAT",
            "params": {"nlist": 128}
        }
        collection.create_index("embedding", index_params)
        
        logging.info("Knowledge base initialized successfully!")

    except Exception as e:
        logging.error(f"Initialization failed: {str(e)}")
        raise

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--milvus-host', default='standalone', help='Milvus服务地址')
    parser.add_argument('--milvus-port', type=int, default=19530, help='Milvus端口')
    parser.add_argument('--knowledge-file', default='knowledge.json', help='知识库文件路径')
    args = parser.parse_args()
    
    init_knowledge_base(args.milvus_host, args.milvus_port, args.knowledge_file)