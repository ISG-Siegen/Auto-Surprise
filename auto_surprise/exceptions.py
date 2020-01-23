class Error(Exception):
   """Base class for other exceptions"""
   pass

class ValidationError(Error):
    """
    Exceptions raised for validation errors
    """

    def __init__(self, expression, message):
        self.expression = expression
        self.message = message
