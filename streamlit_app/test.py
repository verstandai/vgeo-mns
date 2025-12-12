import os

if __name__ == "__main__":
    print(__file__)
    print(os.path.dirname(__file__))
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    print(os.path.join(os.path.dirname(BASE_DIR), "mns_demo_enriched.csv"))