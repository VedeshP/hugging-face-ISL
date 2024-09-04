# Available backend options are: "jax", "tensorflow", "torch".
import os
os.environ["KERAS_BACKEND"] = "tensorflow"
	
import keras

model = keras.saving.load_model("hf://cdsteameight/ISL-SignLanguageTranslation")

print(model)

# from transformers import AutoModel, AutoTokenizer

# model_name = "cdsteameight/ISL-SignLanguageTranslation"
# model = AutoModel.from_pretrained(model_name)
# tokenizer = AutoTokenizer.from_pretrained(model_name)

# # For info about model's layers and expected input format
# print(model)
# # For info about the tokenizer, such as the expected input format
# print(tokenizer)

# outputs = model(**inputs)