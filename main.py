import os
import json
import logging
import uvicorn
from fastapi import FastAPI, HTTPException
from pymilvus import connections, Collection
from sentence_transformers import SentenceTransformer
from dashscope import Generation

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# 初始化组件
app = FastAPI()

# 连接 Milvus
milvus_host = os.getenv("MILVUS_HOST", "standalone")  # 使用容器名称
connections.connect(host=milvus_host, port=19530)
collection = Collection("ros_knowledge")
encoder = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')

# 加载Qwen-Turbo配置
QWEN_API_KEY = "sk-0f7f1821377c4de3ae1bf166175673b6" 

def search_knowledge(prompt: str, top_k: int = 3) -> list:
    """语义检索增强"""
    try:
        # 生成查询向量
        query_vec = encoder.encode(prompt).tolist()
        
        # Milvus检索
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
            "你是一个ROS2专家,请严格根据以下知识回答：\n"
            f"{'\n'.join(context)}\n\n"
            f"问题：{prompt}\n"
            "回答要求：\n"
            "1. 使用中文\n"
            "2. 包含具体命令示例\n"
            "3. 标注知识来源ID(如【来源1】)"
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