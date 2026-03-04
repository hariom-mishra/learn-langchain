from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, ToolMessage
from dotenv import load_dotenv

load_dotenv()

@tool
def multiply(a: int, b: int) -> int:
    """Multiply two numbers"""
    return a * b

llm = ChatOpenAI()
llm_with_tool = llm.bind_tools([multiply])

# Step 1: Human message
query = HumanMessage(content="What is 5 multiplied by 7?")
messages = [query]

# Step 2: LLM decides to call tool
res = llm_with_tool.invoke(messages)
messages.append(res)

# Step 3: Execute tool
tool_call = res.tool_calls[0]
tool_result = multiply.invoke(tool_call["args"])

# Step 4: Wrap result in ToolMessage (IMPORTANT)
tool_message = ToolMessage(
    content=str(tool_result),
    tool_call_id=tool_call["id"],
)

messages.append(tool_message)

# Step 5: Final LLM response
llm_response = llm_with_tool.invoke(messages)

print("Final Answer:")
print(llm_response.content)