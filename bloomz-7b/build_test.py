import torch
import msgpack
from transformers import AutoModelForCausalLM, AutoTokenizer
checkpoint = "bigscience/bloomz-7b1-mt"

model = AutoModelForCausalLM.from_pretrained(checkpoint)
tokenizer = AutoTokenizer.from_pretrained(checkpoint)

text = [
'''
All three calls return the length of the message on successful completion.
If a message is too long to fit in the supplied buffer, excess bytes may be discarded depending  on  the  type  of socket the message is received from.
'''
,
'''
The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

'''
];

tks = tokenizer( text, truncation=True, padding="max_length", max_length=512, return_tensors="pt")


torch.save(tks, "xinput.pth");
vlist = tks["input_ids"].numpy().flatten().tolist()
d = msgpack.packb(vlist, use_bin_type=True);
with open( "xinput.ids.msg", "wb") as outfile:
    outfile.write(d)

vlist = tks["attention_mask"].numpy().flatten().tolist()
d = msgpack.packb(vlist, use_bin_type=True);
with open( "xinput.mask.msg", "wb") as outfile:
    outfile.write(d)

x = model(**tks, output_attentions = False, output_hidden_states = True );

