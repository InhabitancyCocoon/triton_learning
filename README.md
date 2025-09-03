## Triton self learning

### environment

triton 3.4.0  
torch 2.8.0+cu128  
cuda 12.8  
RTX5090  (capability 12.0)

### Note

- There are some accuracy problems with matmul and linear.
- RTX5090 doesn't support certain features like blockwise scaled matmul.
- RTX5090 will encounter some OOM error. Try to reduce the problem size.
- Please check the corresponding version tag of triton tutorial, don't use main branch.
- Some perfermance reports are weird.

```
import torch
torch.cuda.get_device_capability()
```

### link

[triton_tutorial](https://github1s.com/triton-lang/triton/blob/v3.4.0/python/tutorials/)