import torch
import msgpack
from transformers import AutoModelForCausalLM, AutoTokenizer
checkpoint = "bigscience/bloomz-7b1-mt"

#model = AutoModelForCausalLM.from_pretrained(checkpoint)
tokenizer = AutoTokenizer.from_pretrained(checkpoint)

text = [
'''
Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.
'''
,
'''
THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
'''
,
'''
select()  allows  a  program to monitor multiple file descriptors,
waiting until one or more of the file descriptors become "ready" for some class of I/O operation (e.g., input possible).
A file descriptor is considered ready if it is possible to perform a corresponding I/O operation (e.g., read(2),
or a sufficiently small write(2)) without blocking.
'''
,
'''
All three calls return the length of the message on successful completion.
If a message is too long to fit in the supplied buffer, excess bytes may be discarded depending  on  the  type  of socket the message is received from.
'''
];

tks = tokenizer( text, truncation=True, padding="max_length", max_length=512, return_tensors="pt")

torch.save(tks, "xinput.pth");

vlist = tks["input_ids"].int().numpy().flatten().tolist()
d = msgpack.packb(vlist, use_bin_type=True);
with open( "xinput.ids.msg", "wb") as outfile:
    outfile.write(d)

vlist = tks["attention_mask"].int().numpy().flatten().tolist()
d = msgpack.packb(vlist, use_bin_type=True);
with open( "xinput.mask.msg", "wb") as outfile:
    outfile.write(d)

##x = model(**ids, output_attentions=False);
