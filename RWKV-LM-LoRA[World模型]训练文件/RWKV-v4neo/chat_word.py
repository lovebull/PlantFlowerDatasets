########################################################################################################
# The RWKV Language Model - https://github.com/BlinkDL/RWKV-LM
########################################################################################################

print('Loading...')
from src.model_run import RWKV_RNN
import numpy as np
import os, copy, types, gc, sys
import torch

try:
    os.environ["CUDA_VISIBLE_DEVICES"] = sys.argv[1]
except:
    pass
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cuda.matmul.allow_tf32 = True
np.set_printoptions(precision=4, suppress=True, linewidth=200)

CHAT_LANG = 'Chinese'  # English Chinese

WORD_NAME = [
    "20B_tokenizer.json",
    "20B_tokenizer.json",
]  # [vocab, vocab] for Pile model
UNKNOWN_CHAR = None

args = types.SimpleNamespace()
args.RUN_DEVICE = "cuda"  # 'cpu' (already very fast) // 'cuda'
args.FLOAT_MODE = "bf16"  # fp32 (good for CPU) // fp16 (recommended for GPU) // bf16 (less accurate)
args.vocab_size = 65536
args.head_qk = 0
args.pre_ffn = 0
args.grad_cp = 0
args.my_pos_emb = 0

args.MODEL_NAME = 'models/RWKV-4-World-CHNtuned-1.5B-v1-20230620-ctx4096'
args.n_layer = 24
args.n_embd = 2048
args.ctx_len = 1024

# Modify this to use LoRA models; lora_r = 0 will not use LoRA weights.
args.MODEL_LORA = 'out/rwkv-3'
args.lora_r = 8
args.lora_alpha = 32

import os, copy, types, gc, sys

current_path = os.path.dirname(os.path.abspath(__file__))
print('current_path---',current_path)
sys.path.append(f'{current_path}/../rwkv_pip_package/src')

import numpy as np
from prompt_toolkit import prompt

try:
    os.environ["CUDA_VISIBLE_DEVICES"] = sys.argv[1]
except:
    pass
np.set_printoptions(precision=4, suppress=True, linewidth=200)

print('\n\nChatRWKV v2 https://github.com/BlinkDL/ChatRWKV')

import torch

torch.backends.cudnn.benchmark = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cuda.matmul.allow_tf32 = True



########################################################################################################

# args.strategy = 'cpu fp32'
# args.strategy = 'cuda fp16'
# args.strategy = 'cuda:0 fp16 -> cuda:1 fp16'
# args.strategy = 'cuda fp16i8 *10 -> cuda fp16'
# args.strategy = 'cuda fp16i8'
# args.strategy = 'cuda fp16i8 -> cpu fp32 *10'
# args.strategy = 'cuda fp16i8 *10+'

os.environ["RWKV_JIT_ON"] = '1'  # '1' or '0', please use torch 1.13+ and benchmark speed
os.environ["RWKV_CUDA_ON"] = '0'  # '1' to compile CUDA kernel (10x faster), requires c++ compiler & cuda libraries


CHAT_LEN_SHORT = 40
CHAT_LEN_LONG = 150
FREE_GEN_LEN = 256

# For better chat & QA quality: reduce temp, reduce top-p, increase repetition penalties
# Explanation: https://platform.openai.com/docs/api-reference/parameter-details
GEN_TEMP = 1.1  # It could be a good idea to increase temp when top_p is low
GEN_TOP_P = 0.7  # Reduce top_p (to 0.5, 0.2, 0.1 etc.) for better Q&A accuracy (and less diversity)
GEN_alpha_presence = 0.2  # Presence Penalty
GEN_alpha_frequency = 0.2  # Frequency Penalty
GEN_penalty_decay = 0.996
AVOID_REPEAT = '，：？！'

CHUNK_LEN = 256  # split input into chunks to save VRAM (shorter -> slower)


########################################################################################################

# print(f'\n{CHAT_LANG} - {args.strategy} - {PROMPT_FILE}')

from src.utils_word import PIPELINE


print(f'Loading model - {args.MODEL_NAME}')
model = RWKV_RNN(args)
pipeline = PIPELINE(model, f"rwkv_vocab_v20230424")
END_OF_TEXT = 0
END_OF_LINE = 11
# END_OF_TEXT = 0
# END_OF_LINE = 187
END_OF_LINE_DOUBLE = 535
# pipeline = PIPELINE(model, "cl100k_base")
# END_OF_TEXT = 100257
# END_OF_LINE = 198

model_tokens = []
model_state = None

AVOID_REPEAT_TOKENS = []
for i in AVOID_REPEAT:
    dd = pipeline.encode(i)
    assert len(dd) == 1
    AVOID_REPEAT_TOKENS += dd


########################################################################################################

def run_rnn(tokens, newline_adj=0):
    global model_tokens, model_state

    tokens = [int(x) for x in tokens]
    model_tokens += tokens
    # print(f'### model ###\n{tokens}\n[{pipeline.decode(model_tokens)}]')

    while len(tokens) > 0:
        out, model_state = model.forward(tokens[:CHUNK_LEN], model_state)
        tokens = tokens[CHUNK_LEN:]

    out[END_OF_LINE] += newline_adj  # adjust \n probability

    if model_tokens[-1] in AVOID_REPEAT_TOKENS:
        out[model_tokens[-1]] = -999999999
    return out


all_state = {}


def save_all_stat(srv, name, last_out):
    n = f'{name}_{srv}'
    all_state[n] = {}
    all_state[n]['out'] = last_out
    all_state[n]['rnn'] = copy.deepcopy(model_state)
    all_state[n]['token'] = copy.deepcopy(model_tokens)


def load_all_stat(srv, name):
    global model_tokens, model_state
    n = f'{name}_{srv}'
    model_state = copy.deepcopy(all_state[n]['rnn'])
    model_tokens = copy.deepcopy(all_state[n]['token'])
    return all_state[n]['out']


# Model only saw '\n\n' as [187, 187] before, but the tokenizer outputs [535] for it at the end
def fix_tokens(tokens):
    if 'world/' in args.MODEL_NAME or '-World-' in args.MODEL_NAME:
        return tokens
    if len(tokens) > 0 and tokens[-1] == END_OF_LINE_DOUBLE:
        tokens = tokens[:-1] + [END_OF_LINE, END_OF_LINE]
    return tokens


########################################################################################################

if CHAT_LANG == 'English':
    user = "Question"
    bot = "Answer"
    interface = ":"

    # The following is a verbose and detailed conversation between an AI assistant called {bot}, and a human user called {user}. {bot} is intelligent, knowledgeable, wise and polite.
    # The following is a conversation between a highly knowledgeable and intelligent AI called {bot}, and a human called {user}. In the following interactions, {user} and {bot} converse in natural language, and {bot} do its best to answer {user}'s questions. {bot} is respectful, polite and inclusive. {bot} knows a lot, and always tells the truth.

    init_prompt = f'''
The following is a verbose and detailed conversation between an AI assistant called {bot}, and a human user called {user}. {bot} is intelligent, knowledgeable, wise and polite.

{user}{interface} french revolution what year

{bot}{interface} The French Revolution started in 1789, and lasted 10 years until 1799.

{user}{interface} 3+5=?

{bot}{interface} The answer is 8.

{user}{interface} guess i marry who ?

{bot}{interface} Only if you tell me more about yourself - what are your interests?

{user}{interface} solve for a: 9-a=2

{bot}{interface} The answer is a = 7, because 9 - 7 = 2.

{user}{interface} wat is lhc

{bot}{interface} LHC is a high-energy particle collider, built by CERN, and completed in 2008. They used it to confirm the existence of the Higgs boson in 2012.

'''
    HELP_MSG = '''Commands:
say something --> chat with bot. use \\n for new line.
+alt --> alternate chat reply
+reset --> reset chat

+gen YOUR PROMPT --> free generation with any prompt. use \\n for new line.
+qa YOUR QUESTION --> free generation - ask any question (just ask the question). use \\n for new line.
+more --> continue last free generation (only for +gen / +qa)
+retry --> retry last free generation (only for +gen / +qa)

Now talk with the bot and enjoy. Remember to +reset periodically to clean up the bot's memory. Use RWKV-4 14B for best results.
This is not instruct-tuned for conversation yet, so don't expect good quality. Better use +gen for free generation.
'''
elif CHAT_LANG == 'Chinese':

    user = "Question"
    bot = "Answer"
    interface = ":"

    init_prompt = '''
Q: 企鹅会飞吗？

A: 企鹅是不会飞的。它们的翅膀主要用于游泳和平衡，而不是飞行。

Q: 西瓜是什么

A: 西瓜是一种常见的水果，是一种多年生蔓生藤本植物。西瓜的果实呈圆形或卵形，通常是绿色的，里面有红色或黄色的肉和很多的籽。西瓜味甜，多吃可以增加水分，是夏季非常受欢迎的水果之一。

'''
    HELP_MSG = '''指令:
直接输入内容 --> 和机器人聊天，用\\n代表换行
+alt --> 让机器人换个回答
+reset --> 重置对话

+gen 某某内容 --> 续写任何中英文内容，用\\n代表换行
+qa 某某问题 --> 问独立的问题（忽略上下文），用\\n代表换行
+more --> 继续 +gen / +qa 的回答
+retry --> 换个 +gen / +qa 的回答

现在可以输入内容和机器人聊天（注意它不怎么懂中文，它可能更懂英文）。请经常使用 +reset 重置机器人记忆。
'''

# Run inference
print(f'\nRun prompt...')

# user, bot, interface, init_prompt = load_prompt(PROMPT_FILE)
out = run_rnn(fix_tokens(pipeline.encode(init_prompt)))
save_all_stat('', 'chat_init', out)
gc.collect()
torch.cuda.empty_cache()

srv_list = ['dummy_server']
for s in srv_list:
    save_all_stat(s, 'chat', out)


def reply_msg(msg):
    print(f'{bot}{interface} {msg}\n')


def on_message(message):
    global model_tokens, model_state, user, bot, interface, init_prompt

    srv = 'dummy_server'

    msg = message.replace('\\n', '\n').strip()

    x_temp = GEN_TEMP
    x_top_p = GEN_TOP_P
    if ("-temp=" in msg):
        x_temp = float(msg.split("-temp=")[1].split(" ")[0])
        msg = msg.replace("-temp=" + f'{x_temp:g}', "")
        # print(f"temp: {x_temp}")
    if ("-top_p=" in msg):
        x_top_p = float(msg.split("-top_p=")[1].split(" ")[0])
        msg = msg.replace("-top_p=" + f'{x_top_p:g}', "")
        # print(f"top_p: {x_top_p}")
    if x_temp <= 0.2:
        x_temp = 0.2
    if x_temp >= 5:
        x_temp = 5
    if x_top_p <= 0:
        x_top_p = 0
    msg = msg.strip()

    if msg == '+reset':
        out = load_all_stat('', 'chat_init')
        save_all_stat(srv, 'chat', out)
        reply_msg("Chat reset.")
        return

    # use '+prompt {path}' to load a new prompt
    elif msg[:8].lower() == '+prompt ':
        print("Loading prompt...")
        try:
            # PROMPT_FILE = msg[8:].strip()
            # user, bot, interface, init_prompt = load_prompt(PROMPT_FILE)
            out = run_rnn(fix_tokens(pipeline.encode(init_prompt)))
            save_all_stat(srv, 'chat', out)
            print("Prompt set up.")
            gc.collect()
            torch.cuda.empty_cache()
        except:
            print("Path error.")

    elif msg[:5].lower() == '+gen ' or msg[:3].lower() == '+i ' or msg[:4].lower() == '+qa ' or msg[:4].lower() == '+qq ' or msg.lower() == '+++' or msg.lower() == '++':

        if msg[:5].lower() == '+gen ':
            new = '\n' + msg[5:].strip()
            # print(f'### prompt ###\n[{new}]')
            model_state = None
            model_tokens = []
            out = run_rnn(pipeline.encode(new))
            save_all_stat(srv, 'gen_0', out)

        elif msg[:3].lower() == '+i ':
            msg = msg[3:].strip().replace('\r\n', '\n').replace('\n\n', '\n')
            new = f'''
Below is an instruction that describes a task. Write a response that appropriately completes the request.

# Instruction:
{msg}

# Response:
'''
            # print(f'### prompt ###\n[{new}]')
            model_state = None
            model_tokens = []
            out = run_rnn(pipeline.encode(new))
            save_all_stat(srv, 'gen_0', out)

        elif msg[:4].lower() == '+qq ':
            new = '\nQ: ' + msg[4:].strip() + '\nA:'
            # print(f'### prompt ###\n[{new}]')
            model_state = None
            model_tokens = []
            out = run_rnn(pipeline.encode(new))
            save_all_stat(srv, 'gen_0', out)

        elif msg[:4].lower() == '+qa ':
            out = load_all_stat('', 'chat_init')

            real_msg = msg[4:].strip()
            new = f"{user}{interface} {real_msg}\n\n{bot}{interface}"
            # print(f'### qa ###\n[{new}]')

            out = run_rnn(pipeline.encode(new))
            save_all_stat(srv, 'gen_0', out)

        elif msg.lower() == '+++':
            try:
                out = load_all_stat(srv, 'gen_1')
                save_all_stat(srv, 'gen_0', out)
            except:
                return

        elif msg.lower() == '++':
            try:
                out = load_all_stat(srv, 'gen_0')
            except:
                return

        begin = len(model_tokens)
        out_last = begin
        occurrence = {}
        for i in range(FREE_GEN_LEN + 100):
            for n in occurrence:
                out[n] -= (GEN_alpha_presence + occurrence[n] * GEN_alpha_frequency)
            token = pipeline.sample_logits(
                out,
                temperature=x_temp,
                top_p=x_top_p,
            )
            if token == END_OF_TEXT:
                break
            if token not in occurrence:
                occurrence[token] = 1
            else:
                occurrence[token] += 1

            if msg[:4].lower() == '+qa ':  # or msg[:4].lower() == '+qq ':
                out = run_rnn([token], newline_adj=-2)
            else:
                out = run_rnn([token])

            xxx = pipeline.decode(model_tokens[out_last:])
            if '\ufffd' not in xxx:  # avoid utf-8 display issues
                print(xxx, end='', flush=True)
                out_last = begin + i + 1
                if i >= FREE_GEN_LEN:
                    break
        print('\n')
        # send_msg = pipeline.decode(model_tokens[begin:]).strip()
        # print(f'### send ###\n[{send_msg}]')
        # reply_msg(send_msg)
        save_all_stat(srv, 'gen_1', out)

    else:
        if msg.lower() == '+':
            try:
                out = load_all_stat(srv, 'chat_pre')
            except:
                return
        else:
            out = load_all_stat(srv, 'chat')
            msg = msg.strip().replace('\r\n', '\n').replace('\n\n', '\n')
            new = f"{user}{interface} {msg}\n\n{bot}{interface}"
            # print(f'### add ###\n[{new}]')
            out = run_rnn(pipeline.encode(new), newline_adj=-999999999)
            save_all_stat(srv, 'chat_pre', out)

        begin = len(model_tokens)
        out_last = begin
        print(f'{bot}{interface}', end='', flush=True)
        occurrence = {}
        for i in range(999):
            if i <= 0:
                newline_adj = -999999999
            elif i <= CHAT_LEN_SHORT:
                newline_adj = (i - CHAT_LEN_SHORT) / 10
            elif i <= CHAT_LEN_LONG:
                newline_adj = 0
            else:
                newline_adj = min(3, (i - CHAT_LEN_LONG) * 0.25)  # MUST END THE GENERATION

            for n in occurrence:
                out[n] -= (GEN_alpha_presence + occurrence[n] * GEN_alpha_frequency)
            token = pipeline.sample_logits(
                out,
                temperature=x_temp,
                top_p=x_top_p,
            )
            # if token == END_OF_TEXT:
            #     break
            if token not in occurrence:
                occurrence[token] = 1
            else:
                occurrence[token] += 1

            out = run_rnn([token], newline_adj=newline_adj)
            out[END_OF_TEXT] = -999999999  # disable <|endoftext|>

            xxx = pipeline.decode(model_tokens[out_last:])
            if '\ufffd' not in xxx:  # avoid utf-8 display issues
                print(xxx, end='', flush=True)
                out_last = begin + i + 1

            send_msg = pipeline.decode(model_tokens[begin:])
            if '\n\n' in send_msg:
                send_msg = send_msg.strip()
                break

            # send_msg = pipeline.decode(model_tokens[begin:]).strip()
            # if send_msg.endswith(f'{user}{interface}'): # warning: needs to fix state too !!!
            #     send_msg = send_msg[:-len(f'{user}{interface}')].strip()
            #     break
            # if send_msg.endswith(f'{bot}{interface}'):
            #     send_msg = send_msg[:-len(f'{bot}{interface}')].strip()
            #     break

        # print(f'{model_tokens}')
        # print(f'[{pipeline.decode(model_tokens)}]')

        # print(f'### send ###\n[{send_msg}]')
        # reply_msg(send_msg)
        save_all_stat(srv, 'chat', out)


########################################################################################################

if CHAT_LANG == 'English':
    HELP_MSG = '''Commands:
say something --> chat with bot. use \\n for new line.
+ --> alternate chat reply
+reset --> reset chat

+gen YOUR PROMPT --> free single-round generation with any prompt. use \\n for new line.
+i YOUR INSTRUCT --> free single-round generation with any instruct. use \\n for new line.
+++ --> continue last free generation (only for +gen / +i)
++ --> retry last free generation (only for +gen / +i)

Now talk with the bot and enjoy. Remember to +reset periodically to clean up the bot's memory. Use RWKV-4 14B (especially https://huggingface.co/BlinkDL/rwkv-4-raven) for best results.
'''
elif CHAT_LANG == 'Chinese':
    HELP_MSG = f'''指令:
直接输入内容 --> 和机器人聊天（建议问机器人问题），用\\n代表换行，必须用 Raven 模型
+ --> 让机器人换个回答
+reset --> 重置对话，请经常使用 +reset 重置机器人记忆

+i 某某指令 --> 问独立的问题（忽略聊天上下文），用\\n代表换行，必须用 Raven 模型
+gen 某某内容 --> 续写内容（忽略聊天上下文），用\\n代表换行，写小说用 testNovel 模型
+++ --> 继续 +gen / +i 的回答
++ --> 换个 +gen / +i 的回答

作者：彭博 请关注我的知乎: https://zhuanlan.zhihu.com/p/603840957
如果喜欢，请看我们的优质护眼灯: https://withablink.taobao.com

中文 Novel 模型，可以试这些续写例子（不适合 Raven 模型）：
+gen “区区
+gen 以下是不朽的科幻史诗长篇巨著，描写细腻，刻画了数百位个性鲜明的英雄和宏大的星际文明战争。\\n第一章
+gen 这是一个修真世界，详细世界设定如下：\\n1.
'''
elif CHAT_LANG == 'Japanese':
    HELP_MSG = f'''コマンド:
直接入力 --> ボットとチャットする．改行には\\nを使用してください．
+ --> ボットに前回のチャットの内容を変更させる．
+reset --> 対話のリセット．メモリをリセットするために，+resetを定期的に実行してください．

+i インストラクトの入力 --> チャットの文脈を無視して独立した質問を行う．改行には\\nを使用してください．
+gen プロンプトの生成 --> チャットの文脈を無視して入力したプロンプトに続く文章を出力する．改行には\\nを使用してください．
+++ --> +gen / +i の出力の回答を続ける．
++ --> +gen / +i の出力の再生成を行う.

ボットとの会話を楽しんでください。また、定期的に+resetして、ボットのメモリをリセットすることを忘れないようにしてください。
'''

# print(HELP_MSG)
# print(f'{CHAT_LANG} - {args.MODEL_NAME} - {args.strategy}')

# print(f'{pipeline.decode(model_tokens)}'.replace(f'\n\n{bot}',f'\n{bot}'), end='')

########################################################################################################

while True:
    msg = prompt(f'{user}{interface} ')
    if len(msg.strip()) > 0:
        on_message(msg)
    else:
        print('Error: please say something')