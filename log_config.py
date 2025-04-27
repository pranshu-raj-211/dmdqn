import logging

logger = logging.getLogger("dmdqn_logger")
logger.setLevel(logging.INFO)

file_handler = logging.FileHandler("replay_buffer.log", mode='w')
file_handler.setLevel(logging.INFO)

formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s [%(filename)s:%(lineno)d] - %(message)s')
file_handler.setFormatter(formatter)

logger.addHandler(file_handler)