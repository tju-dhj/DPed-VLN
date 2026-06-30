import sys
print('Python version:', sys.version)

try:
    from transformers import AutoModelForCausalLM, AutoTokenizer
    print('Transformers imported successfully')
    
    model_path = '/share/home/u19666033/dhj/models/Qwen3.6-27B'
    print(f'Loading tokenizer from {model_path}...')
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    print('Tokenizer loaded successfully')
    
    print('Loading model...')
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        device_map='auto',
        trust_remote_code=True,
        torch_dtype='auto'
    )
    print('Model loaded successfully!')
    
    # 测试生成
    messages = [{'role': 'user', 'content': 'Say hello in one sentence.'}]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer([text], return_tensors='pt').to(model.device)
    
    outputs = model.generate(**inputs, max_new_tokens=50)
    response = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
    print(f'Test response: {response}')
    
except Exception as e:
    print(f'Error: {e}')
    import traceback
    traceback.print_exc()
