import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

import json
import logging
import argparse
from pymilvus import connections, Collection, utility
from sentence_transformers import SentenceTransformer


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

        # 初始化编码模型
        encoder = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')

        # 读取知识库文件
        logging.info(f"Loading knowledge file: {knowledge_file}")
        with open(knowledge_file, 'r', encoding='utf-8') as f:
            data = json.load(f)['knowledge_base']

        # 创建集合
        collection_name = "ros_knowledge"
        if not utility.has_collection(collection_name):
            schema = {
                "fields": [
                    {"name": "id", "type": "INT64", "is_primary": True},
                    {"name": "title", "type": "VARCHAR", "max_length": 200},
                    {"name": "content", "type": "VARCHAR", "max_length": 2000},
                    {"name": "embedding", "type": "FLOAT_VECTOR", "dim": 384}
                ],
                "description": "ROS2 Knowledge Base"
            }
            Collection.create(collection_name, schema)
            logging.info(f"Collection {collection_name} created")

        # 准备数据
        collection = Collection(collection_name)
        entities = []
        for item in data:
            combined_text = f"{item['title']} {item['content']}"
            embedding = encoder.encode(combined_text).tolist()
            
            entities.append({
                "id": item['id'],
                "title": item['title'],
                "content": item['content'],
                "embedding": embedding
            })

        # 插入数据
        logging.info(f"Inserting {len(entities)} entries...")
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