import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

# Load model and tokenizer from local path
model_path = "Helsinki-NLP/opus-mt-af-en"
model = AutoModelForSeq2SeqLM.from_pretrained(model_path)
tokenizer = AutoTokenizer.from_pretrained(model_path)
# Input text in Afrikaans
input_text = """Speaker 1: So Gerard, ek weet, jy is baie mal oor melkos. Eet jy dit nog gereeld? 

Speaker 2: Ja, so ek is baie mal daar oor, maar ek het nie eindelijk meer seiker, nie. So, nou moet ek ander plane maak. Nou, hierby my, ek bly in Pretoria, mos. Mos. 

Speaker 1: Mos nie, in Spanje, nie. Liefs. 

Speaker 2: Nou, by Hokkaai Slaghuis koop ek altyds ek een klein bakkies kouwe melkos, wat niks suiker in het nie, en die sakkies met die kaneelsuiker word so apart, en dan gooi ek die kaneelsuiker weg, en dan eet ek net so die skoen kouwe melkos, net soos dit is, baie lekker vir een nagerig. 

Speaker 1: Dit klink amper soos my maas' saaghoepoeding. 

Speaker 2: Ja, so, jy weet, as daar Sago in een melkkoos is, dan lig ek al 17 wenkbrauwe, want dit spel net 1 ding en dis lui. 

Speaker 1: O, nou ja, tuur die ene, denk, dit is die lui manse methode. So ons gaan nou uitvind, wat is die eigentlik traditionele manier van melkkoos maak, wat nie oor anfanglik Sago was nie? 

"""  # replace with your own text

# Tokenize the input text
inputs = tokenizer(input_text, return_tensors="pt")

# Generate translation
with torch.no_grad():
    outputs = model.generate(**inputs)

# Decode the generated translation
translation = tokenizer.decode(outputs[0], skip_special_tokens=True)
print("Translation:", translation)
