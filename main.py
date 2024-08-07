from typing import Union
from fastapi import FastAPI, Request
import os
# İgnore Warnings
import warnings
warnings.filterwarnings("ignore")
import numpy as np
from pydantic import BaseModel

# Load environment variables from .env file
import pickle
cwd = os.getcwd()
file_name_njoku = cwd+"/model_svm_eninjoku.pkl"
file_name_makama = cwd+"/model_svm_makama.pkl"
file_name_mariere = cwd+"/model_svm_mariere.pkl"
file_name_mth = cwd+"/model_svm_mth.pkl"
file_name_sodeinde = cwd+"/model_svm_sodeinde.pkl"

app = FastAPI()

model_njoku = pickle.load(open(file_name_njoku, "rb"))
model_makama = pickle.load(open(file_name_makama, "rb"))
model_mariere = pickle.load(open(file_name_mariere, "rb"))
model_mth = pickle.load(open(file_name_mth, "rb"))
model_sodeinde = pickle.load(open(file_name_sodeinde, "rb"))

# scaler_njoku = model_njoku.named_steps['standardscaler']
# scaler_makama = model_makama.named_steps['standardscaler']
# scaler_mariere = model_mariere.named_steps['standardscaler']
# scaler_mth = model_mth.named_steps['standardscaler']
# scaler_sodeinde = model_sodeinde.named_steps['standardscaler']


location = {  "eninjoku": model_njoku, "makama": model_makama, "mariere": model_mariere, "mth": model_mth, "sodeinde": model_sodeinde}


def predict_next_hours(hours, model,former):
  final = []
  for hour in range(hours//2):
    reshaped_input = np.array(former).reshape(1, -1)
    next_prediction = model.named_steps['svr'].predict(reshaped_input).flatten()[0] #model.predict(np.array(former))
    final.append(next_prediction)
    former.append(next_prediction)
    former = former[1:]
  return model.named_steps['standardscaler'].inverse_transform(np.array(final).reshape(1, -1)).flatten()




@app.get("/")
def read_root():
    
    return {"Hello": "World"}


@app.post("/forecast")
async def get_prediction(request: Request):
    form = {  "eninjoku": [-1.57097799, -1.50370353, -1.43642908, -1.36915462, -1.30188016,
       -1.26824293, -1.20096847, -1.13369401, -1.06641956, -0.9991451 ,
       -0.93187064, -0.86459618, -0.83095895, -0.76368449, -0.69641003,
       -0.62913558, -0.56186112, -0.49458666, -0.4273122 , -0.39367497,
       -0.32640051, -0.25912606, -0.1918516 , -0.12457714], 
       "makama": [-1.57428492, -1.50739993, -1.44051493, -1.37362994, -1.30674494,
       -1.27330244, -1.20641745, -1.13953245, -1.07264746, -1.00576247,
       -0.93887747, -0.87199248, -0.83854998, -0.77166498, -0.70477999,
       -0.63789499, -0.57101   , -0.504125  , -0.43724001, -0.40379751,
       -0.33691252, -0.27002752, -0.20314253, -0.13625753], 
       "mariere": [-0.88006657, -0.77945559, -0.67884462, -0.57823364, -0.47762266,
       -0.41054868, -0.3099377 , -0.20932673, -0.10871575, -0.04164177,
        0.05896921,  0.15958019,  0.26019116,  0.36080214,  0.42787612,
        0.5284871 ,  0.62909808,  0.72970905,  0.79678304,  0.89739401,
        0.99800499,  1.09861597,  1.19922694,  1.26630093], 
        "mth": [ 0.81407187,  0.84772387,  0.88137586,  0.94867985,  0.98233184,
        1.04963583,  1.08328783,  1.15059182,  1.18424381,  1.2515478 ,
        1.31885179,  1.38615578,  1.45345977,  1.52076376,  1.55441575,
        1.58806775,  1.62171974,  1.65537174,  1.68902373, -1.67617573,
       -1.60887174, -1.57521975, -1.50791576, -1.44061177], 
       "sodeinde": [ 0.97703352,  1.01046372,  1.04389391,  1.0773241 ,  1.1107543 ,
        1.14418449,  1.17761468,  1.21104488,  1.24447507,  1.27790527,
        1.31133546,  1.34476565,  1.37819585,  1.41162604,  1.44505623,
        1.47848643,  1.51191662,  1.54534682,  1.57877701,  1.6122072 ,
        1.6456374 ,  1.67906759, -1.66395178, -1.63052158]}

    message = await request.json()

    hours =  message["hours"]
    location_name = message["location"]
    
    model = location[location_name]
    prediction = predict_next_hours(hours, model, form[location_name])

    #print("prediction",prediction)
    result = {"response":prediction.tolist()}
    return result #await request.json()
