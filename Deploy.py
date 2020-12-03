from data import load_folder,Preproc_Elements_Load,Tf_Convert
import argparse

#Args Parse related
parser = argparse.ArgumentParser()
parser.add_argument("model_path",default="./model_path")
parser.add_argument("pad_len",default=45)
args = parser.parse_args()
Model_Path = args.model_path
Pad = args.pad_len

def start():
    Vocab,Model = load_folder(Model_Path)
    return Vocab,Model
if __name__ == "__main__":
    Input = [["...","..."]] #Always should 
    vocab,model = start()
    test_x = Tf_Convert(Preproc_Elements_Load(Input,vocab,Pad_Len=Pad))
    print(test_x)
    print(model.predict(test_x))
