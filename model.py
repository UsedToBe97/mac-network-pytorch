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

        self.shared_control_proj = linear(cfg.MAC.DIM, cfg.MAC.DIM)
        self.position_aware = nn.ModuleList()
        for i in range(cfg.MAC.MAX_ITER):
            self.position_aware.append(linear(cfg.MAC.DIM, cfg.MAC.DIM))    # if controlInputUnshared

        self.control_question = linear(cfg.MAC.DIM * 2, cfg.MAC.DIM)
        self.attn = linear(cfg.MAC.DIM, 1)

        self.dim = cfg.MAC.DIM

    def forward(self, context, question, controls):
        cur_step = len(controls) - 1
        control = controls[-1]

        question = torch.tanh(self.shared_control_proj(question))       # TODO: avoid repeating call
        position_aware = self.position_aware[cur_step](question)

        control_question = torch.cat([control, position_aware], 1)
        control_question = self.control_question(control_question)
        control_question = control_question.unsqueeze(1)

        context_prod = control_question * context

        # ++ optionally concatenate words (= context)

        # optional projection (if config.controlProj) --> stacks another linear after activation

        attn_weight = self.attn(context_prod)

        attn = F.softmax(attn_weight, 1)

        # only valid if self.inwords == self.outwords
        next_control = (attn * context).sum(1)

        return next_control


class ReadUnit(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg

        self.read_dropout = nn.Dropout(cfg.MAC.READ_DROPOUT)
        self.mem_proj = linear(cfg.MAC.DIM, cfg.MAC.DIM)
        self.kb_proj = linear(cfg.MAC.DIM, cfg.MAC.DIM)
        self.concat = linear(cfg.MAC.DIM * 2, cfg.MAC.DIM)
        self.concat2 = linear(cfg.MAC.DIM, cfg.MAC.DIM)
        self.attn = linear(cfg.MAC.DIM, 1)

    def forward(self, memory, know, control):
        ## Step 1: knowledge base / memory interactions 
        last_mem = self.read_dropout(memory[-1])
        know = self.read_dropout(know.permute(0,2,1))
        proj_mem = self.mem_proj(last_mem).unsqueeze(1)
        proj_know = self.kb_proj(know)
        concat = torch.cat([
            proj_mem * proj_know, 
            proj_know,
            # proj_mem        # readMemConcatProj (this also makes the know above be the projection)
        ], 2)

        # project memory interactions back to hidden dimension
        concat = self.concat2(F.elu(self.concat(concat)))       # if readMemProj ++ second projection and nonlinearity if readMemAct 

        ## Step 2: compute interactions with control (if config.readCtrl)
        attn = concat * control[-1].unsqueeze(1)

        # if readCtrlConcatInter torch.cat([interactions, concat])

        # optionally concatenate knowledge base elements

        # optional nonlinearity 

        attn = self.read_dropout(attn)
        attn = self.attn(attn).squeeze(2)
        attn = F.softmax(attn, 1).unsqueeze(2)

        read = (attn * know).sum(1)

        return read


class WriteUnit(nn.Module):
    def __init__(self, cfg):
        super().__init__()

        if cfg.MAC.SELF_ATT:
            self.control = linear(cfg.MAC.DIM, cfg.MAC.DIM)
            self.attn = linear(cfg.MAC.DIM, 1)
            self.concat = linear(cfg.MAC.DIM * 3, cfg.MAC.DIM)
        else:
            self.concat = linear(cfg.MAC.DIM * 2, cfg.MAC.DIM)

        self.self_attention = cfg.MAC.SELF_ATT

    def forward(self, memories, retrieved, controls):
        # optionally project info if config.writeInfoProj:

        # optional info nonlinearity if writeInfoAct != 'NON'

        # compute self-attention vector based on previous controls and memories
        if self.self_attention:
            selfControl = controls[-1]
            selfControl = self.control(selfControl)
            controls_cat = torch.stack(controls[:-1], 2)
            attn = selfControl.unsqueeze(2) * controls_cat
            attn = self.attn(attn.permute(0, 2, 1))
            attn = F.softmax(attn, 1).permute(0, 2, 1)

            memories_cat = torch.stack(memories, 2)
            attn_mem = (attn * memories_cat).sum(2)
            # next_mem = self.W_s(attn_mem) + self.W_p(concat)


        prev_mem = memories[-1]
        # get write unit inputs: previous memory, the new info, optionally self-attention / control
        concat = torch.cat([retrieved, prev_mem], 1)

        if self.self_attention:
            concat = torch.cat([concat, attn_mem], 1)

        # project memory back to memory dimension if config.writeMemProj
        concat = self.concat(concat)

        # optional memory nonlinearity

        # write unit gate moved to RNNWrapper

        # optional batch normalization

        next_mem = concat

        return next_mem


class MACCell(nn.Module):
    def __init__(self, cfg):
        super().__init__()

        self.control = ControlUnit(cfg)
        self.read = ReadUnit(cfg)
        self.write = WriteUnit(cfg)

        self.mem_0 = nn.Parameter(torch.zeros(1, cfg.MAC.DIM))
        # control0 is most often question, other times (eg. args2.txt) its a learned parameter initialized as random normal
        if not cfg.MAC.INIT_CNTRL_AS_Q:
            self.control_0 = nn.Parameter(torch.zeros(1, cfg.MAC.DIM))

        self.cfg = cfg
        self.dim = cfg.MAC.DIM

    def init_hidden(self, b_size, question):
        if self.cfg.MAC.INIT_CNTRL_AS_Q:
            control = question
        else:
            control = self.control_0.expand(b_size, self.dim)
        memory = self.mem_0.expand(b_size, self.dim)

        controls = [control]
        memories = [memory]

        return (controls, memories)

    def forward(self, inputs, state):
        words, question, img = inputs
        controls, memories = state

        control = self.control(words, question, controls)
        controls.append(control)

        read = self.read(memories, img, controls)
        # if config.writeDropout < 1.0:     dropouts["write"]
        memory = self.write(memories, read, controls)
        memories.append(memory)

        return controls, memories


class OutputUnit(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.question_proj = nn.Linear(cfg.MAC.DIM, cfg.MAC.DIM)
        self.classifier_out = nn.Sequential(nn.Dropout(p=cfg.MAC.OUTPUT_DROPOUT),       # output dropout outputDropout=0.85
                                        nn.Linear(cfg.MAC.DIM * 2, cfg.MAC.DIM),
                                        nn.ELU(),
                                        nn.Dropout(p=cfg.MAC.OUTPUT_DROPOUT),       # output dropout outputDropout=0.85
                                        nn.Linear(cfg.MAC.DIM, cfg.OUTPUT.DIM))
        xavier_uniform_(self.classifier_out[1].weight)
        xavier_uniform_(self.classifier_out[4].weight)
    
    def forward(self, last_mem, question):
        question = self.question_proj(question)
        cat = torch.cat([last_mem, question], 1)
        out = self.classifier_out(cat)
        return out


class RecurrentWrapper(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.controller = MACCell(cfg)
        self.classifier_out = OutputUnit(cfg)

        self.cfg = cfg
        self.gate = linear(cfg.MAC.DIM, 1)
    
    def forward(self, *inputs):
        state = self.controller.init_hidden(inputs[1].size(0), inputs[1])

        for _ in range(1, self.cfg.MAC.MAX_ITER + 1):
            state = self.controller(inputs, state)

            # memory gate
            if self.cfg.MAC.MEMORY_GATE:
                controls, memories = state
                gate = torch.sigmoid(self.gate(controls[-1]) + self.cfg.MAC.MEMORY_GATE_BIAS)
                memories[-1] = gate * memories[-2] + (1 - gate) * memories[-1]
        
        _, memories = state

        out = self.classifier_out(memories[-1], inputs[1])

        return out


class MACNetwork(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Dropout(p=cfg.MAC.STEM_DROPOUT),             # stem dropout stemDropout=0.82
            nn.Conv2d(1024, cfg.MAC.DIM, 3, padding=1),
            nn.ELU(),
            nn.Dropout(p=cfg.MAC.STEM_DROPOUT),             # stem dropout stemDropout=0.82
            nn.Conv2d(cfg.MAC.DIM, cfg.MAC.DIM, 3, padding=1),
            nn.ELU())
        self.question_dropout = nn.Dropout(cfg.MAC.QUESTION_DROPOUT)
        self.embed = nn.Embedding(cfg.INPUT.N_VOCAB, cfg.MAC.EMBD_DIM)
        # if bi: (bidirectional)
        hDim = int(cfg.MAC.DIM / 2)
        self.lstm = nn.LSTM(cfg.MAC.EMBD_DIM, hDim,
                        # dropout=cfg.MAC.ENC_INPUT_DROPOUT,
                        batch_first=True, bidirectional=True)

        # choose different wrappers for no-act/actSmooth/actBaseline
        self.actmac = RecurrentWrapper(cfg)
        self.dim = cfg.MAC.DIM

        self.reset()

    def reset(self):
        # from original implementation
        xavier_uniform_(self.embed.weight)

        xavier_uniform_(self.conv[1].weight)
        self.conv[1].bias.data.zero_()
        xavier_uniform_(self.conv[4].weight)
        self.conv[4].bias.data.zero_()

    def forward(self, image, question, question_len, dropout=0.15):
        b_size = question.size(0)

        img = self.conv(image)
        img = img.view(b_size, self.dim, -1)

        embed = self.embed(question)
        embed = nn.utils.rnn.pack_padded_sequence(embed, question_len,
                                                batch_first=True)
        lstm_out, (h, _) = self.lstm(embed)
        question = torch.cat([h[0], h[1]], -1)
        question = self.question_dropout(question)
        lstm_out, _ = nn.utils.rnn.pad_packed_sequence(lstm_out,
                                                    batch_first=True)
        h = h.permute(1, 0, 2).contiguous().view(b_size, -1)

        out = self.actmac(lstm_out, question, img)

        return out
