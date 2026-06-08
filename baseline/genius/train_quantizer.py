import runpy
import sys


if __name__ == "__main__":
    sys.argv = ["pro.train_quantizer", "--config", "configs/genius_multimodal.yaml"] + sys.argv[1:]
    runpy.run_module("pro.train_quantizer", run_name="__main__")
