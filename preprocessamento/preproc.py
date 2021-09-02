import pandas as pd

data = pd.read_csv("/caminhocsv.csv", header = None)
data2 = pd.read_csv("/caminhocsv.csv", header = None)

list_data_malicious = [] #criando lista vazia para separar cada classe em um dataframe
list_data_benign = []


for x in range (5): #tirando a quíntupla                       
  data=data.drop(data.columns[0], axis=1)
  data2=data2.drop(data2.columns[0], axis=1)
#print("label da primeira linha, que é normal\n")
#print(data.values[0][40]) #printa a label da primeira linha, que é normal
#print("label da sexta linha, que é maliciosa\n")
#print(data.values[5][40]) #printa label da sexta linha, que é maliciosa
#print(data.head(20))



for x in range (4): # tirando 4 colunas totalmente zeradas. deixei 3 para ter 36 colunas.
  data=data.drop(data.columns[24], axis=1)
  data2=data2.drop(data2.columns[24], axis=1)


#organizando os pixels
#[0,1,2,14,18,21,22,23,25,26,27,33]
data[data.columns[6]],data[data.columns[7]],data[data.columns[8]],data[data.columns[9]],data[data.columns[10]],data[data.columns[11]],data[data.columns[24]],data[data.columns[28]],data[data.columns[29]]= data2[data2.columns[0]],data2[data2.columns[1]],data2[data2.columns[2]],data2[data2.columns[14]],data2[data2.columns[18]],data2[data2.columns[21]],data2[data2.columns[33]],data2[data2.columns[22]],data2[data2.columns[23]]


data[data.columns[0]],data[data.columns[1]],data[data.columns[2]],data[data.columns[14]],data[data.columns[18]],data[data.columns[21]],data[data.columns[33]],data[data.columns[22]],data[data.columns[23]]= data2[data2.columns[6]],data2[data2.columns[7]],data2[data2.columns[8]],data2[data2.columns[9]],data2[data2.columns[10]],data2[data2.columns[11]],data2[data2.columns[24]],data2[data2.columns[28]],data2[data2.columns[29]]

#não vai ser mais util, posso apagar
del data2

for i in range (2000): ## MUDAR DE ACORDO COM O DATASET, atualmente para 2k
  if data.values[i][36] == 0: # separa os benignos
    list_data_benign.append(data.values[i]) # data.values[i] é uma amostra, nesse caso, benigna
  else:
    list_data_malicious.append(data.values[i])


#Transformando as listas em dataframes
data_benign = pd.DataFrame(list_data_benign)
data_malicious = pd.DataFrame(list_data_malicious)

"""**Continuando o tratamento dos dados**"""
#tirando o label
data_benign=data_benign.drop(data_benign.columns[36], axis=1)
data_malicious=data_malicious.drop(data_malicious.columns[36], axis=1)


data_benign2=data_benign.to_numpy()
data_malicious2=data_malicious.to_numpy()


#transformando em matriz 6x6
data_benign2 = data_benign2.reshape((data_benign2.shape[0], 6, 6)).astype('float32') #data_benign.shape[0] é a quantidade de imagens
data_malicious2 = data_malicious2.reshape((data_malicious2.shape[0], 6, 6)).astype('float32')


from matplotlib import pyplot as plt
plt.gray() #colocando a cor padrão como tons de cinza

for x in range(data_benign2.shape[0]):
  plt.imsave('caminhodataset/benign/'+str(x)+'.tiff',data_benign2[x]) #.tiff usado para facilitar leitura no hdfs
  plt.imsave('caminhodataset/malicious/'+str(x)+'.tiff',data_malicious2[x])

print("préprocessamento realizado com sucesso")
