virtual ComputingReturn op_dump(tensor_t self) {
    br_assert(self.get() == this, "can't be here!");
    auto ret = impl()->op_dump(self);
    op_check(ret, "dump");
}
virtual ComputingReturn op_zero(tensor_t self) {
    br_assert(self.get() == this, "can't be here!");
    auto ret = impl()->op_zero(self);
    op_check(ret, "zero");
}
virtual ComputingReturn op_fill(tensor_t self, float value) {
    br_assert(self.get() == this, "can't be here!");
    auto ret = impl()->op_fill(self, value);
    op_check(ret, "fill");
}
virtual ComputingReturn op_copy(tensor_t self, tensor_t dst) {
    br_assert(self.get() == this, "can't be here!");
    br_assert(items() == dst->items(), "copy must has same size");
    auto ret = impl()->op_copy(self, dst);
    op_check(ret, "copy");
}

virtual std::variant<ComputingReturn, tensor_t> op_view(tensor_t self, size_t offset, const std::vector<size_t>& newShape) {
    br_assert(self.get() == this, "can't be here!");
    ShapeType s(newShape);
    br_assert(offset + s.numel() <= items() , "view out of shape!");
    auto result = impl()->op_view(self, offset, newShape);
    if ( result.index() == 0) {
        ComputingReturn ret = std::get<0>(result);
        op_check(ret, "view");
    }
    return result;
}
virtual ComputingReturn op_build_alibi(tensor_t self) {
    br_assert(self.get() == this, "can't be here!");
    auto ret = impl()->op_build_alibi(self);
    op_check(ret, "build_alibi");
}
virtual ComputingReturn op_add(tensor_t self, tensor_t b, tensor_t c) {
    br_assert(self.get() == this, "can't be here!");
    auto ret = impl()->op_add(self, b, c);
    op_check(ret, "add");
}
virtual ComputingReturn op_mul(tensor_t self, tensor_t b, tensor_t c) {
    br_assert(self.get() == this, "can't be here!");
    auto ret = impl()->op_mul(self, b, c);
    op_check(ret, "add");
}
virtual ComputingReturn op_linear(tensor_t self, tensor_t w, tensor_t b, tensor_t y) {
    br_assert(self.get() == this, "can't be here!");
    auto ret = impl()->op_linear(self, w, b, y);
    op_check(ret, "linear");
}
virtual ComputingReturn op_layernorm(tensor_t self, tensor_t mean, tensor_t var, tensor_t scale, tensor_t bias, tensor_t y) {
    br_assert(self.get() == this, "can't be here!");
    auto ret = impl()->op_layernorm(self, mean, var, scale, bias, y);
    op_check(ret, "layernorm");
}
virtual ComputingReturn op_transpos_0213(tensor_t self, tensor_t y) {
    br_assert(self.get() == this, "can't be here!");
    auto ret = impl()->op_transpos_0213(self, y);
    op_check(ret, "transpose_0213");
}
virtual ComputingReturn op_qk(tensor_t self, tensor_t k, tensor_t qk) {
    br_assert(self.get() == this, "can't be here!");
    auto ret = impl()->op_qk(self, k, qk);
    op_check(ret, "qk");
}
virtual ComputingReturn op_softmax(tensor_t self, tensor_t out) {
    br_assert(self.get() == this, "can't be here!");
    auto ret = impl()->op_softmax(self, out);
    op_check(ret, "softmax");
}
virtual ComputingReturn op_attn(tensor_t self, tensor_t v, tensor_t attn) {
    br_assert(self.get() == this, "can't be here!");
    auto ret = impl()->op_attn(self, v, attn);
    op_check(ret, "attn");
}
virtual ComputingReturn op_gelu(tensor_t self, tensor_t dst) {
    br_assert(self.get() == this, "can't be here!");
    auto ret = impl()->op_gelu(self, dst);
    op_check(ret, "attn");
}


