import logging
import os
from datetime import datetime 
                                                             #.log will each log file extension
log_file = f"{datetime.now().strftime('%m-%d-%Y %I-%M-%S %p')}.log"  # Use hyphens and dashes instead of slashes
log_path = os.path.join(os.getcwd(), "logs")  # store log info where call it in any file,  
os.makedirs(log_path, exist_ok=True)# make logs folder,if already exist(exist_ok=Ture), then left 
# cwd= currnet working directory

log_file_path = os.path.join(log_path, log_file)

logging.basicConfig( # now add  necessory details in this funciton(which build-in)
    filename=log_file_path,
    format="[%(asctime)s] %(lineno)d %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO,
)
