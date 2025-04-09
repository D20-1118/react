import os
import json
import logging
import uvicorn
from fastapi import FastAPI, HTTPException
from pymilvus import connections, Collection
from openai import OpenAI  
from dashscope import Generation
from fastapi.middleware.cors import CORSMiddleware 

# 初始化组件
app = FastAPI()

# 允许所有来源的跨域请求
app.add_middleware(
    CORSMiddleware,
    # 允许所有来源的跨域请求，你也可以设置为具体的域名来限制请求来源
    allow_origins=["*"],
    # 参数设置为True表示允许携带身份凭证，如cookies
    allow_credentials=True,
    # 表示允许所有HTTP方法的请求
    allow_methods=["*"],
    # 表示允许所有请求头
    allow_headers=["*"]
)

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)


# OpenAI客户端配置
key = 'sk-bOBUTf38jTU2TDza2nz1cAR8QETryCcI2W1vGTNmZybyUZjL'
QWEN_API_KEY = 'sk-0f7f1821377c4de3ae1bf166175673b6'

client = OpenAI(
    base_url="https://api2.aigcbest.top/v1",
    api_key=key
)

# 连接 Milvus
milvus_host = os.getenv("MILVUS_HOST", "localhost")  
connections.connect(host=milvus_host, port=19530)
collection = Collection("ros_knowledge")

# 新增加载集合的代码
def load_collection():
    try:
        if not collection.has_index():
            raise ValueError("Collection has no index")
        collection.load()
        logging.info("Collection loaded successfully")
    except Exception as e:
        logging.error(f"Failed to load collection: {str(e)}")
        raise

load_collection()  # 启动时加载集合

def get_embedding(text: str) -> list:
    """使用OpenAI生成文本嵌入"""
    try:
        response = client.embeddings.create(
            input=text,
            model="text-embedding-3-small"
        )
        return response.data[0].embedding
    except Exception as e:
        logging.error(f"Embedding生成失败: {str(e)}")
        raise

def search_knowledge(prompt: str, top_k: int = 3) -> list:
    """语义检索增强"""
    try:
        # 生成查询向量
        query_vec = get_embedding(prompt)
        
        # Milvus检索（注意metric_type应与索引类型匹配）
        search_params = {"metric_type": "L2", "params": {"nprobe": 10}}
 
        results = collection.search(
            data=[query_vec],
            anns_field="embedding",
            param=search_params,
            limit=top_k,
            output_fields=["title", "content"]
        )
        
        # 构建上下文
        return [
            f"【{hit.entity.get('title')}】{hit.entity.get('content')}"
            for hit in results[0]
        ]
    except Exception as e:
        logging.error(f"Knowledge search failed: {str(e)}")
        raise


@app.post("/chat")
async def chat_endpoint(prompt_data: dict):
    try:
        prompt = prompt_data.get("prompt", "")
        if not prompt:
            raise HTTPException(status_code=400, detail="Empty prompt")
        
        # 语义检索
        context = search_knowledge(prompt)
        
        # 构建增强prompt
        enhanced_prompt = (
            "你是一个ROS2专家,但如果我没有提到ros相关信息,你可以忽略这个身份,正常回答,若涉及ros信息,请严格根据以下知识回答：/n"
            f"{'/n'.join(context)}/n/n"
            f"问题：{prompt}/n"
            "回答要求：/n"
            "1. 若在 Milvus 中未能检索到合适的ROS2相关知识,就忽略以下要求/n"
            "2. 使用中文/n"
            "3. 包含具体命令示例/n"
            "4. 标注知识来源ID(如【来源1】)"
        )
        
        # 调用Qwen-Turbo
        response = Generation.call(
            model="qwen-turbo",
            prompt=enhanced_prompt,
            api_key=QWEN_API_KEY,
            stream=False
        )
        
        return {
            "text": response.output.text,
            "context": context
        }
        
    except Exception as e:
        logging.error(f"API error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    
    uvicorn.run(app, host="0.0.0.0", port=8000)