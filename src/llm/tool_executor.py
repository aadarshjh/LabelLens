from src.llm.tools import lookup_ingredient, lookup_regulatory_info, lookup_source

ALLOWED_TOOLS = {
    "lookup_ingredient": lookup_ingredient,
    "lookup_regulatory_info": lookup_regulatory_info,
    "lookup_source": lookup_source,
}

def execute_tool(tool_name: str, arguments: dict) -> dict:
    """
    Safely executes a permitted tool using the provided arguments.
    Strictly validates arguments against expected schema.
    """
    if tool_name not in ALLOWED_TOOLS:
        return {
            "success": False,
            "error": f"Tool '{tool_name}' not allowed. Must be one of: {list(ALLOWED_TOOLS.keys())}"
        }

    if not isinstance(arguments, dict):
        return {
            "success": False,
            "error": "Arguments must be a dictionary."
        }

    # Argument validation based on specific tool schema
    if tool_name in ["lookup_ingredient", "lookup_regulatory_info"]:
        expected_arg = "ingredient_name"
    else:
        expected_arg = "source_identifier"

    if expected_arg not in arguments:
        return {
            "success": False,
            "error": f"Missing required argument: '{expected_arg}'"
        }

    if len(arguments) > 1:
        return {
            "success": False,
            "error": "Unexpected additional arguments provided."
        }

    arg_value = arguments[expected_arg]
    if not isinstance(arg_value, str) or not arg_value.strip():
        return {
            "success": False,
            "error": f"Argument '{expected_arg}' must be a non-empty string."
        }

    func = ALLOWED_TOOLS[tool_name]

    try:
        # Execute the underlying controlled python function
        result_data = func(**arguments)
        # Note: Tools now return structured dicts directly,
        # so we just return it to the LLM Client.
        return result_data
    except Exception as e:
        # Catch any unexpected application errors gracefully
        return {
            "success": False,
            "error": f"Tool execution failed: {str(e)}"
        }
