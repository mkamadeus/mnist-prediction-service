from typing import List
from fastapi import FastAPI, UploadFile
from keras.models import load_model
from pydantic import BaseModel
import numpy as np

app = FastAPI()

# from model spec
model = load_model("mnist.h5")

# from output spec
class Output(BaseModel):
	prediction : List[float]

@app.post('/')
async def root(file : UploadFile): # from input spec		
	# ---- BEGIN ----
	contents = await file.read()
	
	# resize
	import io
	from PIL import Image
	stream = io.BytesIO(contents)
	img = Image.open(stream)
	img = img.resize((32,32))

	result = np.array(img).reshape((1,32,32,1))
	print(result.shape)
	# ---- END ----
	
	prediction = model.predict(x=result)
	print(prediction)
	
	return Output(prediction=prediction.tolist()[0])
