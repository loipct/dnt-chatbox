from pydantic import BaseModel
from typing import List, Optional, Union

class Resource(BaseModel):
    topic : str  #Union[str, None]
    title : str
    principle :str