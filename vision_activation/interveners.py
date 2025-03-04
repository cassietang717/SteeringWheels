import torch

def wrapper(intervener):
    def wrapped(*args, **kwargs):
        return intervener(*args, **kwargs)
    return wrapped

# tokens are processed sequentially, and the last token's hidden state (b[0, -1]) contains all previous context
# steering is applied at inference time, when tokens are produced one by one

class Collector():
    collect_state = True
    collect_action = False
    def __init__(self, head):
        self.head = head
        self.states = []
        self.actions = []
    def reset(self):
        self.states = []
        self.actions = []
    def __call__(self, base, source): 
        if self.head == -1:
            self.states.append(base[0, -1].detach().clone())  # original b is (batch_size, seq_len, #key_value_heads x D_head)
        else:
            self.states.append(base[0, -1].reshape(32, -1)[self.head].detach().clone())  # original b is (batch_size, seq_len, #key_value_heads x D_head)
        return base
    
    
class ITI_Intervener():
    collect_state = True
    collect_action = True
    attr_idx = -1
    def __init__(self, direction, multiplier):
        if not isinstance(direction, torch.Tensor):
            direction = torch.tensor(direction)
        self.direction = direction.cuda().half()
        self.multiplier = multiplier
        self.states = []
        self.actions = []
    def reset(self):
        self.states = []
        self.actions = []
    def __call__(self, base, source):
        self.states.append(base[0, -1].detach().clone())  # original b is (batch_size=1, seq_len, #head x D_head), now it's (#head x D_head)
        action = self.direction.to(base.device)
        self.actions.append(action.detach().clone())
        base[0, -1] = base[0, -1] + action * self.multiplier
        return base
