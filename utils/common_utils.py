import copy


def check_system_prompt_support(tokenizer):
    chat = [{"role": "system", "content": 'Test'},]
    try:
        tokenizer.apply_chat_template(chat, tokenize=False)
        return True
    except:
        return False

def fix_tokenizer_chat(tokenizer, chat):
    #tokenizer = copy.deepcopy(tokenizer)
    chat = copy.deepcopy(chat)
    if tokenizer.chat_template == None:
        tokenizer.chat_template ='''{%- for message in messages %}
    {%- if message['role'] == 'user' %}
        {{ message['content'].strip() + '\n' }}
    {%- elif message['role'] == 'system' %}
        {{ message['content'].strip() + '\n' }}
    {%- elif message['role'] == 'assistant' %}
        {{ message['content'].strip() + '\n' }}
    {%- endif %}
{%- endfor %}'''.strip()
    else:
        if check_system_prompt_support(tokenizer) == False:
            #replace system prompt with the next user prompt
            for i in range(len(chat)):
                if chat[i]['role'] == 'system':
                    try:
                        if chat[i+1]['role'] == 'user':
                            chat[i]['role'] = 'user'
                            chat[i]['content'] = chat[i]['content'] + ' ' + chat[i+1]['content']
                            chat[i+1]['role'] = 'popped'
                        else:
                            chat[i]['role'] = 'user'
                        
                    except:
                        chat[i]['role'] = 'user'
            #remove popped elements
            chat = [chat[i] for i in range(len(chat)) if chat[i]['role'] != 'popped']
                      
    return tokenizer, chat