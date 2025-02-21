from flask import Flask, request, jsonify
from flask_cors import CORS
import redis
import json

from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder

from langchain_community.tools import DuckDuckGoSearchRun
from langgraph.prebuilt import create_react_agent

app = Flask(__name__)
CORS(app)

# 连接 Redis 数据库
redis_client = redis.StrictRedis(host='localhost', port=6379, db=0, decode_responses=True)

# 设置模板
prompt_template = ChatPromptTemplate.from_messages(
    [
        ('system', '你是一个智能助手，请完成用户布置的任务'),
        MessagesPlaceholder(variable_name='chat_history'),
        ('human', '{new_message}'),
    ]
)

# 设置模型
model = ChatOpenAI(
    model="gpt-4o-mini",
    openai_api_key='输入你的API',
    temperature=0.9
)

# 连接模板以及模型
chain = prompt_template | model

# 创建 React Agent
tools = [DuckDuckGoSearchRun()]
agent = create_react_agent(model, tools)

# 获取用户 IP 地址
def get_user_ip():
    if request.headers.get('X-Forwarded-For'):
        ip = request.headers.getlist('X-Forwarded-For')[0]
    else:
        ip = request.remote_addr
    return ip

def needs_search(query):
    search_keywords = ["股票", "今天", "现在", "新闻", "实时", "查询", "天气"]
    return any(keyword in query for keyword in search_keywords)

# 获取回复
def get_chatgpt_response(user_ip, user_msg):
    #初始化/读取记忆
    chat_history = json.loads(redis_client.get(user_ip) or '[]')
    #选择是否使用agent
    if needs_search(user_msg):
        response = agent.invoke(
            {'messages':user_msg}
        )
        response = response['messages'][3]
        print('agent')
    else:
        chat_history = json.loads(redis_client.get(user_ip) or '[]')
        response = chain.invoke(
            {'chat_history': chat_history, 'new_message': user_msg}
        )
        print('chain')

    # 更新记忆
    chat_history.append(('human', user_msg))
    chat_history.append(('ai', response.content))
    redis_client.set(user_ip, json.dumps(chat_history))

    return response.content

# 总结记忆
def summarize_chat_history(chat_history):
    summary_response = chain.invoke(
        {'chat_history': chat_history, 'new_message': '请用最少的文字概括对话，只保留对话流程以及对话的大概内容即可。'}
    )
    return summary_response.content

# 更新主题
def get_chat_topic(summary):
    topic_response = chain.invoke(
        {'chat_history': [('system', summary)], 'new_message': '请总结以上对话并生成一个简短的聊天主题,限制在5个字左右,只需要内容即可，不需要类似“主题：”这样的前缀。起题目的语言风格可以略微幽默一些'}
    )
    return topic_response.content

@app.route("/ask", methods=["POST"])
def ask_ai():
    """处理用户问题并调用 AI"""
    user_ip = get_user_ip()  # 获取用户 IP 地址
    
    data = request.get_json()  # 解析前端传来的 JSON
    user_input = data.get("question", "").strip()

    if not user_input:
        return jsonify({"error": "请输入问题"}), 400

    # 调用 OpenAI 生成 AI 回复
    response = get_chatgpt_response(user_ip, user_input)

    chat_topic = None
    # 总结主题
    chat_history = json.loads(redis_client.get(user_ip) or '[]')
    if len(chat_history) > 4: 
        summary = summarize_chat_history(chat_history)
        redis_client.set(user_ip, json.dumps([('system', summary)]))
        # 让 AI 生成当前聊天主题
        chat_topic = get_chat_topic(summary)
        
    return jsonify({"answer": response, "topic": chat_topic})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
