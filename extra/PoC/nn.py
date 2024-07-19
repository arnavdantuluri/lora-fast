from transformers import GPTNeoXForCausalLM, AutoTokenizer
from datetime import datetime
import torch
from hgemm import CublasLinear
hgemm_ = CublasLinear.apply

class linear_wrapper(torch.nn.Module):
    def __init__(self, layer):
        super().__init__()
        self.layer = layer
    
    def forward(self, x):
        out = hgemm_(x, self.layer.weight, self.layer.bias, self.layer.bias != None, "NONE")
        return out

torch.manual_seed(42)
model = GPTNeoXForCausalLM.from_pretrained(
  "EleutherAI/pythia-1b",
).cuda()

tokenizer = AutoTokenizer.from_pretrained(
  "EleutherAI/pythia-1b",
)

inputs = tokenizer("Hello, I am", return_tensors="pt", max_length=256).input_ids.to("cuda")
start = datetime.now()
tokens = model.generate(inputs, use_cache=False)
print(f"Time Elapsed: {datetime.now() - start}")
print(tokenizer.decode(tokens[0]))

for each in model.gpt_neox.layers:
    each.mlp.dense_h_to_4h = linear_wrapper(each.mlp.dense_h_to_4h)
    each.mlp.dense_4h_to_h = linear_wrapper(each.mlp.dense_4h_to_h)
    each.attention.query_key_value = linear_wrapper(each.attention.query_key_value)
    each.attention.dense = linear_wrapper(each.attention.dense)

model = model.half().cuda()
start = datetime.now()
tokens = model.generate(inputs)
print(f"Time Elapsed: {datetime.now() - start}")
print(tokenizer.decode(tokens[0]))