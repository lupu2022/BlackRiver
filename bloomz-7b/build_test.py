import torch
import msgpack
from transformers import AutoModelForCausalLM, AutoTokenizer
checkpoint = "bigscience/bloomz-7b1-mt"

model = AutoModelForCausalLM.from_pretrained(checkpoint)
tokenizer = AutoTokenizer.from_pretrained(checkpoint)

model.eval();

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

'''
torch.save(tks, "xinput.pth");
vlist = tks["input_ids"].numpy().flatten().tolist()
d = msgpack.packb(vlist, use_bin_type=True);
with open( "xinput.ids.msg", "wb") as outfile:
    outfile.write(d)

vlist = tks["attention_mask"].numpy().flatten().tolist()
d = msgpack.packb(vlist, use_bin_type=True);
with open( "xinput.mask.msg", "wb") as outfile:
    outfile.write(d)
'''

labels = tks["input_ids"];

mask = tks["attention_mask"];

labels = torch.masked_fill(labels, mask == 0, -100);
x = model(**tks, output_attentions = False, output_hidden_states = True, labels = labels );

x = x[3][30]
x = x.detach_();

fct = torch.nn.CrossEntropyLoss();
lm_head = model.lm_head;
lm_head.training = True;
x.requires_grad = True;
x1 = lm_head(x);
x1 = x1[..., :-1, :].contiguous();
x1 = x1.view(-1, 250880);
labels = labels[..., 1:].contiguous();
labels = labels.view(-1);

loss = fct(x1, labels);

loss.backward();

dx = x.grad;

x1 = x1.detach();
x1.requires_grad = True;

loss = fct(x1, labels);
loss.backward()

dx1 = x1.grad

