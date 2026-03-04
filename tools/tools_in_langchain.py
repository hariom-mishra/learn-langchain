from langchain_core.tools import tool

#create a function
@tool
def add_numbers(a: int, b: int) -> int:
    """
    Adds two numbers.
    """
    return a + b

result = add_numbers.invoke({"a": 5, "b": 7})

print(result)