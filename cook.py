from flask import Flask, request, jsonify
from flask_cors import CORS

from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder

app = Flask(__name__)
CORS(app)

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
    openai_api_key='设置你的API_KEY',
    temperature=0.9
)

# 连接模板以及模型
chain = prompt_template | model
user_histories = []

# 获取回复
def get_chatgpt_response(user_msg):
    global user_histories  # 声明 user_histories 是一个全局变量
    chat_history = user_histories
    response = chain.invoke(
        {'chat_history': chat_history, 'new_message': user_msg}
    )
    chat_history.append(('human', user_msg))
    chat_history.append(('ai', response.content))
    return response.content

def summarize_chat_history(chat_history):
    summary_response = chain.invoke(
        {'chat_history': chat_history, 'new_message': '请用最少的文字概括对话，只保留对话流程以及对话的大概内容即可。'}
    )
    return summary_response.content

def get_chat_topic(summary):
    topic_response = chain.invoke(
        {'chat_history': [('system', summary)], 'new_message': '请总结以上对话并生成一个简短的聊天主题,限制在5个字左右,只需要内容即可，不需要类似“主题：”这样的前缀。起题目的语言风格可以略微幽默一些'}
    )
    return topic_response.content

@app.route("/ask", methods=["POST"])
def ask_ai():
    """处理用户问题并调用 AI"""
    global user_histories  # 声明 user_histories 是一个全局变量
    
    data = request.get_json()  # 解析前端传来的 JSON
    user_input = data.get("question", "").strip()

    if not user_input:
        return jsonify({"error": "请输入问题"}), 400

    # 调用 OpenAI 生成 AI 回复
    response = get_chatgpt_response(user_input)

    chat_topic = None
    # 记忆总结
    if len(user_histories) > 4: 
        summary = summarize_chat_history(user_histories)
        user_histories = [('system', summary)]
        # 让 AI 生成当前聊天主题
        chat_topic = get_chat_topic(summary)
        
    
    return jsonify({"answer": response, "topic": chat_topic})

if __name__ == "__main__":
    app.run(debug=True)
