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
- Some perfermance reports are weird. Please check the correctness.
- The first param `torch.testing.assert_close()` is true value.
- Be careful of the dtype used for input, computation, accumulation and the output.
- Be careful of the alignment and edge case.
- Leave the ref link, it will help you in the future.



### set up

- linux bash shell settings

```
export PS1="\u@\h:\W> " (\W only display current directory)
export PS1="\u:\W> "

```

- query the device capability

```
import torch
torch.cuda.get_device_capability()
```

- check [autodl github](https://www.autodl.com/docs/network_turbo/) for github access `source /etc/network_turbo`


### link

[triton_tutorial](https://github1s.com/triton-lang/triton/blob/v3.4.0/python/tutorials/)