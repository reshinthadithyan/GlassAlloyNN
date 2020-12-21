from tensorflow import keras
from tensorflow.keras.layers import Input,Dense,Flatten,Embedding,Dropout

def Return_Model(inp1,inp2,input_length,embd_dim,vocab_size):
  #Alloy 
  x1 = Embedding(vocab_size,embd_dim,input_length=input_length)(inp1)
  x1 = keras.layers.Conv1D(64,4,strides=2,activation="relu")(x1)
  x1 = keras.layers.MaxPool1D()(x1)
  x1 = Flatten()(x1)
  #x1=Dense(28,'relu')(x1)
  x1=Dense(128,'relu')(x1)
  x1 = Dropout(0.2)(x1)
  x1=Dense(256,'relu')(x1)
  x1 = Dropout(0.2)(x1)
  x1=Dense(46,'relu')(x1)
  x1 = Dropout(0.2)(x1)

#   #x1=Flatten()(x1)
#   #Composition
  #x2=Dense(28,'relu')(inp2)
  x2=Dense(128,'relu')(inp2)
  x2 = Dropout(0.2)(x2)
  x2=Dense(256,'relu')(x2)
  x2 = Dropout(0.2)(x2)
  x2=Dense(46,'relu')(x2)
  x2 = Dropout(0.2)(x2)
  #x2=Flatten()(x2)
  merge=keras.layers.concatenate([x1,x2])
  #merge = keras.layers.Dot(axes=1)([x1, x2])
  z1=Dense(40,'relu')(merge)
  z1 = Dropout(0.2)(z1)
  outp=Dense(1,'relu')(z1)
  model=keras.Model(inputs=[inp1,inp2],outputs=outp)
  return model
if __name__ == "__main__":
    from data import Preproc_Elements,Tf_Convert
    import pandas as pd
    import numpy as np
    Path = "/home/reshinth-adith/reshinth/work/glass_alloy/GlassAlloyNN/Data/Em_data_corr.csv"
    DF = pd.read_csv(Path)
    Output, Vocab,num_list = Preproc_Elements(DF, "Metallic glasses (at. %)")
    Np_Output = Tf_Convert(Output)
    num_list = Tf_Convert(num_list)
    Np_shear= np.array([float(i) for i in DF["Shear Modulus (GPa)"].values.tolist()])
    print(Np_Output.shape,num_list.shape,Np_shear.shape)
    Model_Func = Return_Model(Input(shape=[7]),Input(shape=[14]),7,3,len(Vocab))
    Model_Func.compile(keras.optimizers.Adam(0.01),'mse','mae')
    print("Model Starting to Execute")
    Model_Func.fit([Np_Output[:200],num_list[:200]],Np_shear[:200],batch_size=32,epochs=100,validation_split=0.1)