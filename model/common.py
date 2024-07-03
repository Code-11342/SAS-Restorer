import logging

def log(msg):
    logger=logging.getLogger("train")
    logger.info(msg)
    print(msg)