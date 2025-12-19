# from .base_prompter import BasePrompter
# from ..models.wan_video_text_encoder import WanTextEncoder
# from transformers import AutoTokenizer
# import os, torch
# import ftfy
# import html
# import string
# import regex as re


# def basic_clean(text):
#     text = ftfy.fix_text(text)
#     text = html.unescape(html.unescape(text))
#     return text.strip()


# def whitespace_clean(text):
#     text = re.sub(r'\s+', ' ', text)
#     text = text.strip()
#     return text


# def canonicalize(text, keep_punctuation_exact_string=None):
#     text = text.replace('_', ' ')
#     if keep_punctuation_exact_string:
#         text = keep_punctuation_exact_string.join(
#             part.translate(str.maketrans('', '', string.punctuation))
#             for part in text.split(keep_punctuation_exact_string))
#     else:
#         text = text.translate(str.maketrans('', '', string.punctuation))
#     text = text.lower()
#     text = re.sub(r'\s+', ' ', text)
#     return text.strip()


# class HuggingfaceTokenizer:

#     def __init__(self, name, seq_len=None, clean=None, **kwargs):
#         assert clean in (None, 'whitespace', 'lower', 'canonicalize')
#         self.name = name
#         self.seq_len = seq_len
#         self.clean = clean

#         # init tokenizer
#         self.tokenizer = AutoTokenizer.from_pretrained(name, **kwargs)
#         self.vocab_size = self.tokenizer.vocab_size

#     def __call__(self, sequence, **kwargs):
#         return_mask = kwargs.pop('return_mask', False)

#         # arguments
#         _kwargs = {'return_tensors': 'pt'}
#         if self.seq_len is not None:
#             _kwargs.update({
#                 'padding': 'max_length',
#                 'truncation': True,
#                 'max_length': self.seq_len
#             })
#         _kwargs.update(**kwargs)

#         # tokenization
#         if isinstance(sequence, str):
#             sequence = [sequence]
#         if self.clean:
#             sequence = [self._clean(u) for u in sequence]
#         ids = self.tokenizer(sequence, **_kwargs)

#         # output
#         if return_mask:
#             return ids.input_ids, ids.attention_mask
#         else:
#             return ids.input_ids

#     def _clean(self, text):
#         if self.clean == 'whitespace':
#             text = whitespace_clean(basic_clean(text))
#         elif self.clean == 'lower':
#             text = whitespace_clean(basic_clean(text)).lower()
#         elif self.clean == 'canonicalize':
#             text = canonicalize(basic_clean(text))
#         return text


# class WanPrompter(BasePrompter):

#     def __init__(self, tokenizer_path=None, text_len=512):
#         super().__init__()
#         self.text_len = text_len
#         self.text_encoder = None
#         self.fetch_tokenizer(tokenizer_path)
        
#     def fetch_tokenizer(self, tokenizer_path=None):
#         if tokenizer_path is not None:
#             self.tokenizer = HuggingfaceTokenizer(name=tokenizer_path, seq_len=self.text_len, clean='whitespace')

#     def fetch_models(self, text_encoder: WanTextEncoder = None):
#         self.text_encoder = text_encoder

#     def encode_prompt(self, prompt, positive=True, device="cuda"):
#         prompt = self.process_prompt(prompt, positive=positive)
        
#         ids, mask = self.tokenizer(prompt, return_mask=True, add_special_tokens=True)
#         ids = ids.to(device)
#         mask = mask.to(device)
#         seq_lens = mask.gt(0).sum(dim=1).long()
#         prompt_emb = self.text_encoder(ids, mask)
#         for i, v in enumerate(seq_lens):
#             prompt_emb[:, v:] = 0
#         return prompt_emb

from .base_prompter import BasePrompter
from ..models.wan_video_text_encoder import WanTextEncoder
from transformers import AutoTokenizer
import os, torch
import ftfy
import html
import string
# 你的代码使用了 regex，这是一个很好的选择，比标准 re 库功能更全
import regex as re
import torch.nn.functional as F

def basic_clean(text):
    text = ftfy.fix_text(text)
    text = html.unescape(html.unescape(text))
    return text.strip()


def whitespace_clean(text):
    text = re.sub(r'\s+', ' ', text)
    text = text.strip()
    return text


def canonicalize(text, keep_punctuation_exact_string=None):
    text = text.replace('_', ' ')
    if keep_punctuation_exact_string:
        text = keep_punctuation_exact_string.join(
            part.translate(str.maketrans('', '', string.punctuation))
            for part in text.split(keep_punctuation_exact_string))
    else:
        text = text.translate(str.maketrans('', '', string.punctuation))
    text = text.lower()
    text = re.sub(r'\s+', ' ', text)
    return text.strip()


class HuggingfaceTokenizer:

    def __init__(self, name, seq_len=None, clean=None, **kwargs):
        assert clean in (None, 'whitespace', 'lower', 'canonicalize')
        self.name = name
        self.seq_len = seq_len
        self.clean = clean

        # init tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(name, **kwargs, use_fast=True)
        self.vocab_size = self.tokenizer.vocab_size

    def __call__(self, sequence, **kwargs):
        # --- MODIFICATION START ---
        # 原来的 if/elif 结构无法同时返回 mask 和 offset_mapping。
        # 我们将其修改为返回一个字典，这更加灵活和明确。
        return_mask = kwargs.pop('return_mask', False)
        return_offsets_mapping = kwargs.pop('return_offsets_mapping', False)

        # arguments
        _kwargs = {'return_tensors': 'pt'}
        if self.seq_len is not None:
            _kwargs.update({
                'padding': 'max_length',
                'truncation': True,
                'max_length': self.seq_len
            })
        # 将 return_offsets_mapping 直接传递给底层的 tokenizer
        _kwargs['return_offsets_mapping'] = return_offsets_mapping
        _kwargs.update(**kwargs)

        # tokenization
        if isinstance(sequence, str):
            sequence = [sequence]
        if self.clean:
            sequence = [self._clean(u) for u in sequence]
        
        # 调用 Hugging Face tokenizer
        encoding = self.tokenizer(sequence, **_kwargs)

        # 构建一个包含所有请求信息的输出字典
        output = {'input_ids': encoding.input_ids}
        if return_mask:
            output['attention_mask'] = encoding.attention_mask
        if return_offsets_mapping:
            # 返回的 offset_mapping 带有批次维度，我们为单条文本处理移除它
            output['offset_mapping'] = encoding.offset_mapping[0] 
        
        return output
        # --- MODIFICATION END ---

    def _clean(self, text):
        if self.clean == 'whitespace':
            text = whitespace_clean(basic_clean(text))
        elif self.clean == 'lower':
            text = whitespace_clean(basic_clean(text)).lower()
        elif self.clean == 'canonicalize':
            text = canonicalize(basic_clean(text))
        return text


class WanPrompter(BasePrompter):

    def __init__(self, tokenizer_path=None, text_len=512):
        super().__init__()
        self.text_len = text_len
        self.text_encoder = None
        self.fetch_tokenizer(tokenizer_path)
        
    def fetch_tokenizer(self, tokenizer_path=None):
        if tokenizer_path is not None:
            self.tokenizer = HuggingfaceTokenizer(name=tokenizer_path, seq_len=self.text_len, clean='whitespace')

    def fetch_models(self, text_encoder: WanTextEncoder = None):
        self.text_encoder = text_encoder


    # def encode_prompt(self, prompt, positive=True, device="cuda"):
#     prompt = self.process_prompt(prompt, positive=positive)
    
#     ids, mask = self.tokenizer(prompt, return_mask=True, add_special_tokens=True)
#     ids = ids.to(device)
#     mask = mask.to(device)
#     seq_lens = mask.gt(0).sum(dim=1).long()
#     prompt_emb = self.text_encoder(ids, mask)
#     for i, v in enumerate(seq_lens):
#         prompt_emb[:, v:] = 0
#     return prompt_emb

    def encode_prompt(self, prompt, positive=True, device="cuda"):
        """
        编码提示，并根据特殊标记返回嵌入向量和 token 位置。
        此实现遵循新的解析规则：[per shot caption] 仅出现一次，[shot cut] 是唯一的分隔符。
        """
        # 1. 文本清理 (与分词器内部逻辑保持一致)
        
        cleaned_prompt = prompt = self.process_prompt(prompt, positive=positive)
        # cleaned_prompt = prompt = 'blue trajectory: The man remains stationary throughout the video.\n'

        # 2. 解析字符级位置 (Character-level Spans)
        # --- MODIFICATION START: 全新的、更精确的解析逻辑 ---
        char_spans = []
        shot_cut_matches = list(re.finditer(os.linesep, cleaned_prompt))

        # Part B: 提取所有 per-shot captions 的范围
            # 第一个 shot 的文本从 [per shot caption] 标记之后开始
        current_start_pos = 0
        shot_id = 0

        # 遍历所有的 [shot cut] 来切分出 shot 1, 2, 3...
        for shot_cut_match in shot_cut_matches:
            # 当前 shot 的结束位置是 [shot cut] 标记的开始
            end_char = shot_cut_match.end()
            
            # 添加当前 shot 的 span
            char_spans.append({'id': shot_id, 'start': current_start_pos, 'end': end_char})
            
            # 为下一个 shot 更新起始位置和 ID
            current_start_pos = shot_cut_match.end()
            shot_id += 1

        # 处理最后一个 shot (在最后一个 [shot cut] 之后，或如果没有 [shot cut] 的情况)
        # 这个逻辑也处理了你提到的 "tricky point"
        # 如果在最后一个 [shot cut] 后还有文本，它就构成了最后一个 shot
        if current_start_pos < len(cleaned_prompt):
            char_spans.append({
                'id': shot_id, 
                'start': current_start_pos, 
                'end': len(cleaned_prompt)
            })
        # --- MODIFICATION END ---

        # 3. 分词并获取偏移映射 (此部分及后续逻辑无需更改)
        enc_output = self.tokenizer(
            prompt,
            return_mask=True,
            return_offsets_mapping=True,
            add_special_tokens=True
        )
        ids = enc_output['input_ids']
        mask = enc_output['attention_mask']
        offsets = enc_output['offset_mapping']

        # 4. 对齐与映射: 将每个 token 映射到一个 shot ID
        token_shot_ids = torch.full((ids.shape[1],), fill_value=-2, dtype=torch.long)
        for i, (token_start, token_end) in enumerate(offsets):
            if token_start == token_end:
                continue
            for span in char_spans:
                if not (token_end <= span['start'] or token_start >= span['end']):
                    token_shot_ids[i] = span['id']
                    break
        
        # 5. 生成最终的 token 序列位置
        positions = []
        
        # 动态确定 shot 的数量
        max_shot_id = token_shot_ids.max().item()
        for i in range(max_shot_id + 1):
            shot_indices = torch.where(token_shot_ids == i)[0]
            if len(shot_indices) > 0:
                positions.append([shot_indices.min().item(), shot_indices.max().item() + 1])

        # 6. 编码与后处理
        ids = ids.to(device)
        mask = mask.to(device)
        
        if self.text_encoder is None:
            raise ValueError("Text encoder has not been fetched. Call fetch_models() first.")

        prompt_emb = self.text_encoder(ids, mask)
        
        seq_lens = mask.gt(0).sum(dim=1).long()
        for i, v in enumerate(seq_lens):
            prompt_emb[i, v:] = 0
        return prompt_emb, positions
    


# 假设 self.tokenizer, self.text_encoder, self.process_prompt 已经存在
# 并且self.text_len = 512 (或其他目标长度)


# 假设 self.tokenizer, self.text_encoder, self.process_prompt 已经存在
# 并且 self.text_len = 512 (或其他目标长度)

    def encode_prompt_separately(self, prompt, positive=True, device="cuda"):
        """
        编码提示，采用“分段编码后拼接”的策略。

        此函数首先解析出 global 和 per-shot 的文本片段，然后对每个片段
        独立进行编码，最后将得到的嵌入向量拼接在一起，并填充到指定的最大长度。
        """
        # --- 1. 解析Prompt，提取出独立的文本片段 ---
        # 这部分逻辑可以复用，但我们的目标不再是字符位置，而是文本本身。
        
        cleaned_prompt = self.process_prompt(prompt, positive=positive)
        
        prompt_parts = []
        
        # 使用正则表达式查找所有标记
        global_match = re.search(r'\[global caption\]', cleaned_prompt)
        per_shot_match = re.search(r'\[per shot caption\]', cleaned_prompt)
        shot_cut_matches = list(re.finditer(r'\[shot cut\]', cleaned_prompt))

        if global_match is None:
            output = self.tokenizer(cleaned_prompt, return_mask=True, add_special_tokens=True)
            
            ids = output['input_ids'].to(device)
            mask = output['attention_mask'].to(device)
            seq_lens = mask.gt(0).sum(dim=1).long()
            prompt_emb = self.text_encoder(ids, mask)
            for i, v in enumerate(seq_lens): 
                prompt_emb[:, v:] = 0
            return prompt_emb, {"global": None, "shots": []}

        # Part A: 提取 global caption 的文本
        if global_match:
            start_pos = global_match.start()
            end_pos = per_shot_match.start() if per_shot_match else len(cleaned_prompt)
            global_text = cleaned_prompt[start_pos:end_pos].strip()
            if global_text:
                prompt_parts.append({'id': -1, 'text': global_text})

        # Part B: 提取 per-shot captions 的文本
        if per_shot_match:
            current_start_pos = per_shot_match.start()
            shot_id = 0

            # 遍历所有 [shot cut] 来切分
            for shot_cut_match in shot_cut_matches:
                end_pos = shot_cut_match.start()
                shot_text = cleaned_prompt[current_start_pos:end_pos].strip()
                if shot_text:
                    prompt_parts.append({'id': shot_id, 'text': shot_text})
                
                current_start_pos = shot_cut_match.start()
                shot_id += 1

            # 处理最后一个 shot
            last_shot_text = cleaned_prompt[current_start_pos:].strip()
            if last_shot_text:
                prompt_parts.append({'id': shot_id, 'text': last_shot_text})

        # --- 2. 对每个文本片段进行独立编码 ---
        
        if self.text_encoder is None:
            raise ValueError("Text encoder has not been fetched. Call fetch_models() first.")

        embeddings_list = []
        positions = {"global": None, "shots": {}} # 使用字典以防 shot ID 不连续
        current_token_idx = 0

        

        for part in prompt_parts:
            text = part['text']
            shot_id = part['id']

            # 对单个片段进行分词和编码
            enc_output = self.tokenizer(
                text,
                return_mask=True,
                add_special_tokens=True, # 每个片段都是一个独立的序列
                return_tensors="pt"
            )
            ids = enc_output['input_ids'].to(device)
            mask = enc_output['attention_mask'].to(device)
            
            # 获取该片段的嵌入向量
            part_emb = self.text_encoder(ids, mask) # shape: (1, seq_len, hidden_dim)

            # 计算实际的 token 长度 (排除填充)
            seq_len = mask.sum().item()
            
            # 记录该片段在最终拼接向量中的位置
            start_idx = current_token_idx
            end_idx = current_token_idx + seq_len
            
            if shot_id == -1: # Global prompt
                positions["global"] = [start_idx, end_idx]
            else: # Per-shot prompt
                positions["shots"][shot_id] = [start_idx, end_idx]

            # 将有效的嵌入向量部分 (去除padding) 添加到列表中
            embeddings_list.append(part_emb[0, :seq_len, :])
            
            # 更新下一个片段的起始位置
            current_token_idx += seq_len

        # --- 3. 拼接所有嵌入向量并进行填充 ---

        if not embeddings_list:
            # 如果没有解析出任何文本，返回一个零向量
            return torch.zeros(1, self.text_len, self.text_encoder.config.hidden_size, device=device), {"global": None, "shots": []}

        # 拼接所有片段的嵌入向量
        concatenated_emb = torch.cat(embeddings_list, dim=0) # shape: (total_seq_len, hidden_dim)
        
        # 检查是否超出最大长度
        total_len = concatenated_emb.shape[0]
        if total_len > self.text_len:
            # 这里可以根据策略选择截断或抛出异常
            # 截断是一种常见的处理方式
            print(f"Warning: Concatenated prompt length ({total_len}) exceeds max length ({self.text_len}). Truncating.")
            concatenated_emb = concatenated_emb[:self.text_len, :]
            total_len = self.text_len
            # 注意：截断后，positions字典中的某些位置可能就不准确了，需要更复杂的逻辑来修正。
            # 为简单起见，这里仅作截断。

        # 计算需要填充的长度
        pad_len = self.text_len - total_len
        
        # 在序列长度维度上进行右填充 (padding on the right)
        # F.pad 的参数格式是 (左边填充数, 右边填充数) 从最后一个维度开始
        # 我们要填充的是倒数第二个维度 (sequence length)
        prompt_emb = F.pad(concatenated_emb, (0, 0, 0, pad_len), 'constant', 0)
        
        # 添加 batch 维度，使其 shape 变为 (1, max_length, hidden_dim)
        prompt_emb = prompt_emb.unsqueeze(0)

        # 将 positions 中的 shots 字典转换为列表，以匹配原始输出格式
        final_positions = {"global": positions["global"], "shots": []}
        if positions["shots"]:
            # 按 shot_id 排序，确保列表顺序正确
            sorted_shots = sorted(positions["shots"].items())
            # 检查 shot_id 是否连续，并填充空缺
            max_shot_id = sorted_shots[-1][0]
            shot_map = dict(sorted_shots)
            for i in range(max_shot_id + 1):
                final_positions["shots"].append(shot_map.get(i, None)) # 如果某个shot缺失，则为None

        return prompt_emb, final_positions