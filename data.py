import re
import numpy as np
import pandas as pd
def Create_Vocab(Alloys_List):
    """Given a List of Alloys returns the Metal Vocabulary"""
    Metal_List = ["PAD"]
    for Alloy in Alloys_List:
        for Metal in Alloy:
            if Metal not in Metal_List:
                Metal_List.append(Metal)
    Metal_Dictionary_Vocab = {Metal_List[i]:i for i in range(len(Metal_List))}
    return Metal_Dictionary_Vocab


def Vocabulize_Alloy(Alloy_List,Metal_Dict):
    """Given a List of Alloys Vocabulizes the Alloys present"""
    Output_Num =  []
    for Alloy in Alloy_List:
        Need = [Metal_Dict[i] for i in Alloy]
        Output_Num.append(Need)
    return Output_Num


def Parse_Alloy(Alloy_String):
    """Parsing an Alloy to elementary blocks."""
    Output = re.sub('[0-9]', ' ', Alloy_String)
    Output_Entities = [i for i in Output.split(" ")if i != ""]
    return Output_Entities

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
    return Padded_Alloy_Vec

def Tf_Convert(Inp_List):
    """Converts a List of List to a Numpy Array"""
    Output = np.array([np.array(i) for i in Inp_List])
    return Output        
if __name__ == "__main__":
    DF = pd.read_csv("G:\Work Related\Mettalurgy\Data\EM Dataset.csv")
    Output = Preproc_Elements(DF,"Metallic glasses (at. %)")
    Np_Output = Tf_Convert(Output)
    
    
