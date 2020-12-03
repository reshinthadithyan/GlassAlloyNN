import re
import numpy as np
import pandas as pd
import pickle
def Check_Hyb(Element):
    return "".join([(" "+i if i.isupper() else i) for i in Element]).strip().split()
def Create_Vocab(Alloys_List):
    """Given a List of Alloys returns the Metal Vocabulary"""
    Metal_List = ["PAD"]
    for Alloy in Alloys_List:
        Alloy = Parse_Alloy(Alloy)
        for Metal in Alloy:
            if Metal not in Metal_List and Metal != "": #and "(" not in Metal and "/" not in Metal:
                if len(Check_Hyb(Metal)) == 1:
                    Metal_List.append(Metal)
                else:
                    for Element in Check_Hyb(Metal):
                        if Element not in Metal_List:
                            #print(Element)
                            Metal_List.append(Element)
    Metal_Dictionary_Vocab = {Metal_List[i]:i for i in range(len(Metal_List))}
    return Metal_Dictionary_Vocab


def Vocabulize_Alloy(Alloy_List,Metal_Dict):
    """Given a List of Alloys Vocabulizes the Alloys present"""
    Output_Num =  []
    for Alloy in Alloy_List:
        Alloy = Parse_Alloy(Alloy)
        Need = [Metal_Dict[i] for i in Alloy if i!=""]
        Output_Num.append(Need)
    return Output_Num


def Parse_Alloy(Alloy_String):
    """Parsing an Alloy to elementary blocks."""
    Output = re.sub(r'[0-9\(+\)+\.+\/]', ' ', Alloy_String)
    Output_Entities = [Check_Hyb(i) for i in Output.split(" ")if i != " "]
    return sum(Output_Entities,[])

def Pad_Data(Dataset):
    """Given a List of List Pad based on the max length of the Sublist"""
    Pad_Dataset = []
    Pad_Length = max(len(i) for i in Dataset)
    for i in Dataset:
        if len(i) < Pad_Length:
            Need = i + [0]*(Pad_Length-len(i))
            Pad_Dataset.append(Need)
        else:
            Pad_Dataset.append(i)
    return Pad_Dataset

def Preproc_Elements(DataFrame,key):
    """Given DataFrame and the key column of Alloys, it numericalizes and pads it with the maximum seq length"""
    Alloy_List = DataFrame[key].values.tolist()
    Vocabulary = Create_Vocab(Alloy_List)
    Alloy_Vec = Vocabulize_Alloy(Alloy_List,Vocabulary)
    Padded_Alloy_Vec = Pad_Data(Alloy_Vec) #X
    return Padded_Alloy_Vec,Vocabulary

def Tf_Convert(Inp_List):
    """Converts a List of List to a Numpy Array"""
    Output = np.array([np.array(i) for i in Inp_List])
    return Output     
def Save_Mod(Vocab,Model,Folder_Path):
    """Saves Model in a directory
    Args : Vocab - Variable containing the Vocabulary
           Model - Variable containing the trained Model
           Folder_Path - Folder_Path in which Mod.h5 and vocab.pkl will be saved."""
    Model.save(Folder_Path+"Mod.h5")
    with open(Folder_Path+"vocab.pkl", 'wb') as handle:
        pickle.dump(Vocab, handle, protocol=pickle.HIGHEST_PROTOCOL)
    return "Sucessfully Saved Model and Vocab in"+Folder_Path
if __name__ == "__main__":
    DF = pd.read_csv("G:\Work Related\Mettalurgy\Data\EM Dataset.csv")
    Output,Vocab = Preproc_Elements(DF,"Metallic glasses (at. %)")
    Np_Output = Tf_Convert(Output)    
    
