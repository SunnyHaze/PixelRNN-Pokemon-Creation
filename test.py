import torch
a = torch.tensor(
    [[
        [1,1, 1, 1],
        [2, 2, 2, 2]
    ]], dtype=torch.float32
)
b = torch.tensor([0.5, 0, 0, 0, 0.5])

c = torch.zeros(1, 5)
d = b.unsqueeze(0)
print(a.shape)
print(torch.softmax(a, dim=2))
print(torch.multinomial(b, 1))
print(torch.concat([d, c], dim=0))