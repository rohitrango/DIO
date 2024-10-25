''' Run tests to check behavior of PyTorch '''
import torch

def gradient_within_nograd_check():
    ''' Given a, b, check if gradients w.r.t. a and b are correct 
    this is to emulate the DEQ solver
    '''
    a = torch.randn(300, 200).cuda()
    b = torch.randn_like(a)
    a.requires_grad_(True)
    b.requires_grad_(True)
    all_c = []
    with torch.no_grad():
        c1 = a + b
        all_c.append(c1)
        with torch.enable_grad():
            c2 = a + b**2 + c1
            all_c.append(c2)
            a_grad = torch.autograd.grad(c2.sum(), a)[0]
    print(all_c[0].grad_fn)
    print(all_c[1].grad_fn)
    print(a.grad, b.grad)
    (all_c[1].sum()).backward()
    print(a_grad - a.grad)
    print(b.grad)
    

if __name__ == '__main__':
    print("Running tests...")
    gradient_within_nograd_check()