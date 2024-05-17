import logging

logging.basicConfig(filename='example1.log', level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

i=0
while i<5:
    i+=1
    logging.info(f"Action {i} Started : {i}")
    print("Done")