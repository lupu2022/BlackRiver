import math
import warnings
from typing import Optional, Tuple, Union
from dataclasses import dataclass
import msgpack

import torch
import torch.utils.checkpoint
from torch import nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, LayerNorm, MSELoss
from torch.nn import functional as F

def build_input_for_test():
    x = torch.rand(10, 2048, 4096);
    d = {"x": x};
    torch.save(d, "xinput.pth");

    vlist = x.float().numpy().flatten().tolist()
    d = msgpack.packb(vlist, use_bin_type=True);
    with open( "xinput.msg", "wb") as outfile:
        outfile.write(d)


def dump_qkv_file(sdict, k, target):
    print(target + " is writing...");

    k1 = k + ".weight"
    t = sdict[k1]
    t = t.view(32, 3, 128, 4096);
    t = t.transpose(1, 0)
    t = t.reshape(3*4096, 4096);
    vlist = t.float().numpy().flatten().tolist()
    d = msgpack.packb(vlist, use_bin_type=True);
    with open( target + ".weight.msg", "wb") as outfile:
        outfile.write(d)

    k2 = k + ".bias"
    t = sdict[k2]
    t = t.view(32, 3, 128);
    t = t.transpose(1, 0);
    t = t.reshape(3*4096);
    vlist = t.float().numpy().flatten().tolist()
    d = msgpack.packb(vlist, use_bin_type=True);
    with open( target + ".bias.msg", "wb") as outfile:
        outfile.write(d)

def dump_one_file(sdict, k, target):
    print(target + " is writing...");

    k1 = k + ".weight"
    t = sdict[k1]
    vlist = t.float().numpy().flatten().tolist()
    d = msgpack.packb(vlist, use_bin_type=True);
    with open( target + ".weight.msg", "wb") as outfile:
        outfile.write(d)

    k2 = k + ".bias"
    t = sdict[k2]
    vlist = t.float().numpy().flatten().tolist()
    d = msgpack.packb(vlist, use_bin_type=True);
    with open( target + ".bias.msg", "wb") as outfile:
        outfile.write(d)

def dump_attentions():
    path_src = "pth/"
    path_dst = "weights/"
    for i in range(30):
        hname = "h_" + str(i) + ".pth";
        print("load.. " + str(i)  + " attention");
        sdict = torch.load(path_src + hname );

        key = "input_layernorm"
        target = path_dst + "h" + str(i) + ".input_layernorm";
        dump_one_file(sdict, key, target);

        key = "self_attention.query_key_value"
        target = path_dst + "h" + str(i) + ".query_key_value";
        dump_qkv_file(sdict, key, target);

        key = "self_attention.dense"
        target = path_dst + "h" + str(i) + ".dense";
        dump_one_file(sdict, key, target);

        key = "post_attention_layernorm"
        target = path_dst + "h" + str(i) + ".post_attention_layernorm";
        dump_one_file(sdict, key, target);

        key = "mlp.dense_h_to_4h"
        target = path_dst + "h" + str(i) + ".dense_h_to_4h";
        dump_one_file(sdict, key, target);

        key = "mlp.dense_4h_to_h"
        target = path_dst + "h" + str(i) + ".dense_4h_to_h";
        dump_one_file(sdict, key, target);
