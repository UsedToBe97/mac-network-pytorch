import torch
from torch.autograd import Variable
from torch import nn
from torch.nn.init import kaiming_uniform_, xavier_uniform_, normal
import torch.nn.functional as F


def linear(in_dim, out_dim, bias=True):
    lin = nn.Linear(in_dim, out_dim, bias=bias)
    xavier_uniform_(lin.weight)
    if bias:
        lin.bias.data.zero_()
    return lin


class ControlUnit(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg

        self.position_aware = nn.ModuleList()
        for i in range(cfg.MAC.MAX_ITER):
            self.position_aware.append(linear(cfg.MAC.DIM * 2, cfg.MAC.DIM))

        self.control_question = linear(cfg.MAC.DIM * 2, cfg.MAC.DIM)
        self.attn = linear(cfg.MAC.DIM, 1)

        self.dim = cfg.MAC.DIM

    def forward(self, context, question, controls):
        cur_step = len(controls) - 1
        control = controls[-1]
        position_aware = self.position_aware[cur_step](question)

        control_question = torch.cat([control, position_aware], 1)
        control_question = self.control_question(control_question)
        control_question = control_question.unsqueeze(1)

        context_prod = control_question * context
        attn_weight = self.attn(context_prod)

        attn = F.softmax(attn_weight, 1)

        next_control = (attn * context).sum(1)

        return next_control


class ReadUnit(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg

        self.mem_proj = linear(cfg.MAC.DIM, cfg.MAC.DIM)
        self.kb_proj = linear(cfg.MAC.DIM, cfg.MAC.DIM)
        self.concat = linear(cfg.MAC.DIM * 2, cfg.MAC.DIM)
        self.attn = linear(cfg.MAC.DIM, 1)

    def forward(self, memory, know, control):
        proj_mem = self.mem_proj(memory[-1]).unsqueeze(2)
        proj_know = self.kb_proj(know.permute(0,2,1)).permute(0,2,1)
        concat = self.concat(torch.cat([proj_mem * proj_know, know], 1) \
                                .permute(0, 2, 1))
        attn = concat * control[-1].unsqueeze(1)
        attn = self.attn(attn).squeeze(2)
        attn = F.softmax(attn, 1).unsqueeze(1)

        read = (attn * know).sum(2)

        return read


class WriteUnit(nn.Module):
    def __init__(self, cfg):
        super().__init__()

        self.concat = linear(cfg.MAC.DIM * 2, cfg.MAC.DIM)

        if cfg.MAC.SELF_ATT:
            self.attn = linear(cfg.MAC.DIM, 1)
            self.W_s = linear(cfg.MAC.DIM, cfg.MAC.DIM)
            self.W_p = linear(cfg.MAC.DIM, cfg.MAC.DIM)

        self.self_attention = cfg.MAC.SELF_ATT

    def forward(self, memories, retrieved, controls):
        prev_mem = memories[-1]
        concat = self.concat(torch.cat([retrieved, prev_mem], 1))
        next_mem = concat

        if self.self_attention:
            controls_cat = torch.stack(controls[:-1], 2)
            attn = controls[-1].unsqueeze(2) * controls_cat
            attn = self.attn(attn.permute(0, 2, 1))
            attn = F.softmax(attn, 1).permute(0, 2, 1)

            memories_cat = torch.stack(memories, 2)
            attn_mem = (attn * memories_cat).sum(2)
            next_mem = self.W_s(attn_mem) + self.W_p(concat)

        return next_mem


class MACCell(nn.Module):
    def __init__(self, cfg):
        super().__init__()

        self.control = ControlUnit(cfg)
        self.read = ReadUnit(cfg)
        self.write = WriteUnit(cfg)

        self.mem_0 = nn.Parameter(torch.zeros(1, cfg.MAC.DIM))
        self.control_0 = nn.Parameter(torch.zeros(1, cfg.MAC.DIM))

        self.dim = cfg.MAC.DIM
        self.dropout = cfg.MAC.DROPOUT_PROB

    def get_mask(self, x, dropout):
        mask = torch.empty_like(x).bernoulli_(1 - dropout)
        mask = mask / (1 - dropout)

        return mask

    def init_hidden(self, b_size):
        control = self.control_0.expand(b_size, self.dim)
        memory = self.mem_0.expand(b_size, self.dim)

        if self.training:
            control_mask = self.get_mask(control, self.dropout)
            memory_mask = self.get_mask(memory, self.dropout)
            control = control * control_mask
            memory = memory * memory_mask
        else:
            control_mask = None
            memory_mask = None

        controls = [control]
        memories = [memory]

        return (controls, memories), (control_mask, memory_mask)

    def forward(self, inputs, state, masks):
        words, question, img = inputs
        controls, memories = state
        control_mask, memory_mask = masks
        
        control = self.control(words, question, controls)
        if self.training:
            control = control * control_mask
        controls.append(control)

        read = self.read(memories, img, controls)
        memory = self.write(memories, read, controls)
        if self.training:
            memory = memory * memory_mask
        memories.append(memory)

        return controls, memories


class RecurrentWrapper(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.controller = MACCell(cfg)
        self.classifier = nn.Sequential(linear(cfg.MAC.DIM * 3, cfg.MAC.DIM),
                                        nn.ELU(),
                                        linear(cfg.MAC.DIM, cfg.OUTPUT.DIM))
        kaiming_uniform_(self.classifier[0].weight)
        kaiming_uniform_(self.classifier[2].weight)

        self.cfg = cfg
        # gate with xavier uniform
        self.gate = linear(cfg.MAC.DIM, 1)
    
    def forward(self, *inputs):
        state, masks = self.controller.init_hidden(inputs[1].size(0))

        for _ in range(1, self.cfg.ACT.MAX_ITER + 1):
            state = self.controller(inputs, state, masks)

            # memory gate
            if self.cfg.MAC.MEMORY_GATE:
                controls, memories = state
                gate = torch.sigmoid(self.gate(controls[-1]) + self.cfg.MAC.MEMORY_GATE_BIAS)
                memories[-1] = gate * memories[-2] + (1 - gate) * memories[-1]
        
        _, memories = state

        out = torch.cat([memories[-1], inputs[1]], 1)
        out = self.classifier(out)

        return out


class MACNetwork(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.conv = nn.Sequential(nn.Conv2d(1024, cfg.MAC.DIM, 3, padding=1),
                                nn.ELU(),
                                nn.Conv2d(cfg.MAC.DIM, cfg.MAC.DIM, 3, padding=1),
                                nn.ELU())

        self.embed = nn.Embedding(cfg.INPUT.N_VOCAB, cfg.MAC.EMBD_DIM)
        self.lstm = nn.LSTM(cfg.MAC.EMBD_DIM, cfg.MAC.DIM,
                        batch_first=True, bidirectional=True)
        self.lstm_proj = nn.Linear(cfg.MAC.DIM * 2, cfg.MAC.DIM)

        # choose different wrappers for no-act/actSmooth/actBaseline
        self.actmac = RecurrentWrapper(cfg)
        self.dim = cfg.MAC.DIM

        self.reset()

    def reset(self):
        self.embed.weight.data.uniform_(0, 1)

        kaiming_uniform_(self.conv[0].weight)
        self.conv[0].bias.data.zero_()
        kaiming_uniform_(self.conv[2].weight)
        self.conv[2].bias.data.zero_()

    def forward(self, image, question, question_len, dropout=0.15):
        b_size = question.size(0)

        img = self.conv(image)
        img = img.view(b_size, self.dim, -1)

        embed = self.embed(question)
        embed = nn.utils.rnn.pack_padded_sequence(embed, question_len,
                                                batch_first=True)
        lstm_out, (h, _) = self.lstm(embed)
        lstm_out, _ = nn.utils.rnn.pad_packed_sequence(lstm_out,
                                                    batch_first=True)
        lstm_out = self.lstm_proj(lstm_out)
        h = h.permute(1, 0, 2).contiguous().view(b_size, -1)

        out = self.actmac(lstm_out, h, img)

        return out
