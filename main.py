from typing import List
from fastapi import FastAPI, UploadFile
from keras.models import load_model
from pydantic import BaseModel
import numpy as np
import io
from PIL import Image

app = FastAPI()

# from model spec
model = load_model("mnist.h5")

# from output spec
class Output(BaseModel):
	prediction : List[float]

@app.post('/')
async def root(file : UploadFile): # from input spec		
	contents = await file.read()
	stream = io.BytesIO(contents)
	img = Image.open(stream)
	
	# pipelines
	# ---- BEGIN ----
	# resize image
	img = img.resize((32,32))
	# ---- END ----
	
	# model aggregation
	result = np.array(img).reshape((1,32,32,1))
	print(result.shape)

	# model prediction
	prediction = model.predict(x=result).tolist()[0]
	
	return Output(prediction=prediction)
