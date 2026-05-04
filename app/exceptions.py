class DataSemAgentError(Exception):
    """Base exception for all DataSemAgent errors"""

class TokenLimitExceeded(DataSemAgentError):
    """Exception raised when the token limit is exceeded"""

class MCPConnectionError(DataSemAgentError):
    """Exception raised when the MCP connect fails"""

class QueryNotAllowedError(DataSemAgentError):
    """Exception raised when a non-query SQL statement is executed"""