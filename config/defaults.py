from yacs.config import CfgNode as CN

_C = CN()

# -----------------------------------------------------------------------------
# MAC
# -----------------------------------------------------------------------------
_C.MAC = CN()
_C.MAC.DIM = 512
_C.MAC.EMBD_DIM = 300
_C.MAC.ENC_INPUT_DROPOUT = 0.15
_C.MAC.STEM_DROPOUT = 0.18
_C.MAC.QUESTION_DROPOUT = 0.08
_C.MAC.MEM_DROPOUT = 0.15
_C.MAC.READ_DROPOUT = 0.15
_C.MAC.OUTPUT_DROPOUT = 0.15
_C.MAC.SELF_ATT = False
_C.MAC.MEMORY_GATE = True
_C.MAC.MEMORY_GATE_BIAS = 1.0
_C.MAC.INIT_CNTRL_AS_Q = True
_C.MAC.MAX_ITER = 12
_C.MAC.MEMORY_VAR_DROPOUT = True


# -----------------------------------------------------------------------------
# INPUT
# -----------------------------------------------------------------------------
_C.INPUT = CN()
_C.INPUT.N_VOCAB = 90


# -----------------------------------------------------------------------------
# OUTPUT
# -----------------------------------------------------------------------------
_C.OUTPUT = CN()
_C.OUTPUT.DIM = 28


# -----------------------------------------------------------------------------
# DataLoader
# -----------------------------------------------------------------------------
_C.DATALOADER = CN()
_C.DATALOADER.NUM_WORKERS = 0
_C.DATALOADER.BATCH_SIZE = 64
_C.DATALOADER.FEATURES_PATH = "/storage2/CLEVR_v1.0"


# ---------------------------------------------------------------------------- #
# Solver
# ---------------------------------------------------------------------------- #
_C.SOLVER = CN()
_C.SOLVER.EPOCHS = 20
_C.SOLVER.LR = 1e-4
_C.SOLVER.GRAD_CLIP = 8
_C.SOLVER.USE_SCHEDULER = False


# ---------------------------------------------------------------------------- #
# weight saving/loading options
# ---------------------------------------------------------------------------- #
_C.SAVE_PATH = 'mac'
_C.LOAD = False
_C.LOAD_PATH = ""
_C.DEVICE = "cuda"
