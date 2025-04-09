import os
import json
import logging
import uvicorn
from fastapi import FastAPI, HTTPException
from pymilvus import connections, Collection
from openai import OpenAI
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

# 允许跨域请求
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# 
OPENAI_API_KEY = 'sk-bOBUTf38jTU2TDza2nz1cAR8QETryCcI2W1vGTNmZybyUZjL'
QWEN_API_KEY = 'sk-0f7f1821377c4de3ae1bf166175673b6'


# 初始化两个客户端
openai_client = OpenAI(
    base_url="https://api2.aigcbest.top/v1",
    api_key=OPENAI_API_KEY
    )  

qwen_client = OpenAI(
    api_key=QWEN_API_KEY,
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"
)

# 连接Milvus
connections.connect(host="localhost", port=19530)
collection = Collection("ros_knowledge")
collection.load()

# 工具函数定义
def get_embedding(text: str) -> list:
    """使用OpenAI生成文本嵌入"""
    response = openai_client.embeddings.create(
        input=text,
        model="text-embedding-3-small"
    )
    return response.data[0].embedding

def search_knowledge(query: str, top_k: int = 3) -> list:
    """语义搜索实现"""
    query_vec = get_embedding(query)
    
    search_params = {"metric_type": "L2", "params": {"nprocaBe": 10}}
    results = collection.search(
        data=[query_vec],
        anns_field="embedding",
        param=search_params,
        limit=top_k,
        output_fields=["title", "content"]
    )
    
    return [
        {"title": hit.entity.get('title'), 
         "content": hit.entity.get('content')}
        for hit in results[0]
    ]


tools = [
    {
        "type": "function",
        "function": {
            "name": "search_ros_knowledge",
            "description": "当用户询问ROS2相关技术问题时搜索知识库",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "用于知识库搜索的关键词或问题，用中文表述"
                    },
                    "top_k": {
                        "type": "integer",
                        "description": "返回结果数量",
                        "default": 3
                    }
                },
                "required": ["query"]
            }
        }
    }
]

def process_function_call(response):
    """处理函数调用"""
    if response.choices[0].message.tool_calls:
        tool_call = response.choices[0].message.tool_calls[0]
        if tool_call.function.name == "search_ros_knowledge":
            args = json.loads(tool_call.function.arguments)
            results = search_knowledge(**args)
            
            context = "\n".join(
                [f"【{item['title']}】{item['content']}" 
                 for item in results]
            )
            return context
    return None

@app.post("/chat")
async def chat_endpoint(prompt_data: dict):
    try:
        prompt = prompt_data.get("prompt", "")
        if not prompt:
            raise HTTPException(status_code=400, detail="Empty prompt")

        # 第一轮：判断是否需要搜索
        response = qwen_client.chat.completions.create(
            model="qwen-turbo",
            messages=[{"role": "user", "content": prompt}],
            tools=tools,
            tool_choice="auto"
        )

        # 处理函数调用
        context = ""
        if response.choices[0].message.tool_calls:
            context = process_function_call(response)

            # 第二轮：带上下文生成最终回答
            second_response = qwen_client.chat.completions.create(
                model="qwen-turbo",
                messages=[
                    {"role": "user", "content": prompt},
                    {
                        "role": "tool",
                        "content": context,
                        "tool_call_id": response.choices[0].message.tool_calls[0].id
                    }
                ]
            )
            answer = second_response.choices[0].message.content
        else:
            answer = response.choices[0].message.content

        return {
            "text": answer,
            "context": context.split("\n") if context else []
        }

    except Exception as e:
        logging.error(f"API error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)