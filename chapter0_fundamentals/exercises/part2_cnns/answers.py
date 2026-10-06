# %%

print("hi")

# %%

class ReLU(nn.Module):
    def forward(self, x: Tensor) -> Tensor:
        self.x = nn.Param


tests.test_relu(ReLU)