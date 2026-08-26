import os
from dotenv import load_dotenv

load_dotenv()

var = os.get_env("MY_VARIABLE")
print(var)