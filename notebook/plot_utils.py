import numpy as np
from fancy_einsum import einsum
import torch
import transformer_lens.utils as utils
import seaborn as sns
import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif']=['Times New Roman']

def draw_output_pattern_with_text(component, model, top_k=10):
    layers_unembedded = einsum(
            " ... d_model, d_model d_vocab -> ... d_vocab",
            component,
            model.W_U,
        )
    sorted_indices  = torch.argsort(layers_unembedded, dim=1, descending=True)
    temp_logits = layers_unembedded[0]
    tmp_sorted_indices = sorted_indices[0]
    top_logits = []
    top_tokens = []
    for i in range(top_k):
        top_logits.append(temp_logits[tmp_sorted_indices[i]].item())
        top_tokens.append(model.to_string(tmp_sorted_indices[i]))
    top_logits = np.expand_dims(np.array(top_logits),axis=-1)
    top_tokens = np.expand_dims(np.array(top_tokens),axis=-1)
    # 设置图形大小
    plt.figure(figsize=(1.5, 6),dpi=300)
    # 使用seaborn绘制热力图
    sns.heatmap(top_logits, annot=top_tokens, fmt='', cmap='Blues', cbar=True,xticklabels=False, yticklabels=False)
    # 添加标签和标题
    # plt.xlabel('Tokens')
    # plt.ylabel('Logits')
    # plt.title('Top Tokens Logits Heatmap')
    plt.show()
def draw_attention_pattern(Component,model,layer,head_index):
    fig = px.imshow(
        Component.cache["attn", layer][0, head_index][1:, 1:].cpu().numpy(),
        title=f"{layer}.{head_index} Attention",
        color_continuous_midpoint=0,
        color_continuous_scale="RdBu",
        labels={"y": "Queries", "x": "Keys"},
        height=500,
    )

    fig.update_layout(
        xaxis={
            "side": "top",
            "ticktext": Component.str_tokens[0][1:],
            "tickvals": list(range(len(Component.tokens[0])-1)),
            "tickfont": dict(size=15),
        },
        yaxis={
            "ticktext": Component.str_tokens[0][1:],
            "tickvals": list(range(len(Component.tokens[0])-1)),
            "tickfont": dict(size=15),
        },
    )
    # fig.write_image(f"{layer}.{head_index}_Attention.pdf")
    fig.show()
def draw_rank_logits(gpt2_medium, China):
    x=np.arange(gpt2_medium.cfg.n_layers+1)
    fig=plt.figure(figsize=(8,4),dpi=100)
    y1= China.get_token_rank(gpt2_medium.W_U,China.answer_token,pos=-1).cpu(),
    y1=utils.to_numpy(y1)[0]
    y2= China.get_token_probability(gpt2_medium,China.answer_token,pos=-1).squeeze(-1).cpu()
    y4= China.get_token_probability(gpt2_medium,China.subject_last_token,pos=-1).squeeze(-1).cpu()
    y2=utils.to_numpy(y2)
    # y3= China.get_token_rank(gpt2_medium.W_U,gpt2_medium.to_single_token(' Malaysian'),pos=-1).cpu()
    y3 = China.get_min_rank_at_subject(gpt2_medium.W_U,China.answer_token).cpu()
    y3=utils.to_numpy(y3)
    #ax1显示y1  ,ax2显示y2
    ax1=fig.subplots()
    ax1.plot(x, y1, 'g-', label='Target Entity at Last Position')
    ax1.plot(x, y3, 'r-', label='Target Entity at Subject Position')
    # ax1.grid(True)

    ax2 = ax1.twinx()
    ax2.set_ylim([0, 1])
    ax2.plot(x, y2, 'b--', label='Prob of Object Entity')
    ax2.plot(x, y4, 'g--', label='Prob of Subject Entity')

    ax1.set_xticks(np.arange(0, 24))
    ax1.set_xlabel('layer')
    ax1.set_yscale('log')
    ax1.set_ylabel('rank')
    ax2.set_ylabel('logits')

    # 添加图例
    ax1.legend(loc='upper left')
    ax2.legend(loc='upper right')
    plt.show()