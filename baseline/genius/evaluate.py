import runpy
import sys


if __name__ == "__main__":
    sys.argv = ["pro.evaluate", "--config", "configs/genius_multimodal.yaml"] + sys.argv[1:]
    runpy.run_module("pro.evaluate", run_name="__main__")
