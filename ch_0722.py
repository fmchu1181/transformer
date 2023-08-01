from transformers import BertTokenizer

tokenizer = BertTokenizer.from_pretrained(pretrained_model_name_or_path='bert-base-chinese')


input_text = "這是一個示範句子，用於測試 BERT 的 tokenizer。"
input_text1 = "这是一个示范句子，用于测试 BERT 的 tokenizer。"

tokens = tokenizer.tokenize(input_text1)
print(tokens)

input_ids = tokenizer.convert_tokens_to_ids(tokens)
print(input_ids)