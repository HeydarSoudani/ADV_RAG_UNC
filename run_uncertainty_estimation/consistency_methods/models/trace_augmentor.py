import re
import torch
from utils.general_utils import passages2string, get_think, get_query

def get_paraphrased_query(text):
    pattern = re.compile(r"<paraphrased_query>(.*?)</paraphrased_query>", re.DOTALL)
    matches = pattern.findall(text)
    if matches:
        return matches
    else:
        return None

def get_paraphrased_think(text):
    pattern = re.compile(r"<paraphrased_think>(.*?)</paraphrased_think>", re.DOTALL)
    matches = pattern.findall(text)
    if matches:
        return matches
    else:
        return None

class SearchQueryParaphraser:
    def __init__(self, args, generator, tokenizer) -> None:
        self.args = args
        self.generator = generator
        self.tokenizer = tokenizer
        
        self.examples = [
            {
                "original": "popular industry in the neighborhood of Willow Vale, New South Wales",
                "paraphrased": [
                    "What industries are most common in Willow Vale, NSW?",
                    "Main economic activities in the Willow Vale area of New South Wales",
                    "Leading local businesses or sectors in Willow Vale, NSW",
                    "Top industries driving the economy in Willow Vale, New South Wales",
                    "Which industry is most prominent around Willow Vale, NSW?",
                    "What people mostly work in at Willow Vale New South Wales",
                    "Primary occupation in the Willow Vale NSW locality",
                    "Major employment sector in Willow Vale, New South Wales",
                    "Key local industry in Willow Vale, New South Wales",
                    "Dominant industry around Willow Vale, New South Wales"
                ] 
            },
            {
                "original": "when was the Minnesota Vikings founded",
                "paraphrased": [
                    "what year were the Minnesota Vikings established",
                    "founding date of the Minnesota Vikings",
                    "Minnesota Vikings team origin year",
                    "when did the Minnesota Vikings start",
                    "year the Minnesota Vikings football team was created",
                    "Minnesota Vikings team inception date",
                    "origin year of the Minnesota Vikings",
                    "when was the Minnesota Vikings football team formed",
                    "date the Minnesota Vikings were founded",
                    "Minnesota Vikings creation date"
                ] 
            }
        ]

    def get_instruction(self, original_search_query, n=5):
        input_text = ''
        input_text += "You are an expert in information retrieval."
        input_text += f"Given an original search query, generate {n} semantically diverse and effective paraphrased search queries that capture the same intent but use different wording or structure."
        input_text += "These paraphrased queries should be suitable for improving search engine results by covering various phrasings a user might employ."
        input_text += "Do not add extra information in the new queries.\n\n"
        
        input_text += "Here are some examples:"
        for example in self.examples:
            input_text += f"<original_query> {example['original']} </original_query>\n"
            for pr_query in example['paraphrased'][:n]:
                input_text += f"<paraphrased_query> {pr_query} </paraphrased_query>\n"
            input_text += '\n'
            
        input_text += f"<original_query> {original_search_query.strip()} </original_query>\n"
        
        return input_text
  
    def generate(self,
        input_text,
        max_new_tokens = 1024,
        num_return:int = 1,
        temperature:float = 1.0,
        do_sample:bool = True
    ):
        if self.tokenizer.chat_template:
            input_prompt = self.tokenizer.apply_chat_template(
                [{"role": "user", "content": input_text}],
                add_generation_prompt=True,
                tokenize=False
            )
        
        input_ids = self.tokenizer.encode(input_prompt, return_tensors='pt').to(self.generator.device)
        attention_mask = torch.ones_like(input_ids)
        
        generated_texts = []
        for i in range(num_return):
            with torch.no_grad():
                outputs = self.generator.generate(
                    input_ids,
                    attention_mask=attention_mask,
                    max_new_tokens=max_new_tokens,
                    pad_token_id=self.tokenizer.eos_token_id,
                    temperature=temperature,
                    do_sample=do_sample,
                )
                generated_tokens = outputs[0][input_ids.shape[1]:]
                output_text = self.tokenizer.decode(generated_tokens, skip_special_tokens=True)
                generated_texts.append(output_text)
        
        return generated_texts

    def inference(self, qid, original_sq, repeat=5):
        sq_prompt = self.get_instruction(original_sq, n=repeat)
        sq_output = self.generate(sq_prompt, temperature=0.7)[0]
        paraphrased_queries = get_paraphrased_query(sq_output)

        # check if paraphrased_queries are None
        if paraphrased_queries == None:
            print(f"Paraphrased queries are not provided for query {qid} ...")
            for i in range(self.args.retry):
                print(f"Paraphrased queries, try {i+1} ...")
                sq_output = self.generate(sq_prompt, temperature=1.0)[0]
                paraphrased_queries = get_paraphrased_query(sq_output)
                if paraphrased_queries != None:
                    break
            else:
                print(f"Failed to generate 'paraphrased queries' after all retries for query {qid}!!!")
                paraphrased_queries = []
        
        # Check if the number of paraphrased_queries is equal to "repeat"
        if paraphrased_queries is None:
            paraphrased_queries = []

        max_iterations = 10
        iteration = 0
        while len(paraphrased_queries) != repeat and iteration < max_iterations:
            remaining = repeat - len(paraphrased_queries)        
            extra_prompt = self.get_instruction(original_sq, n=remaining)
            extra_output = self.generate(extra_prompt, temperature=1.0)[0]
            extra_queries = get_paraphrased_query(extra_output)

            if extra_queries:
                paraphrased_queries.extend(extra_queries)
                paraphrased_queries = paraphrased_queries[:repeat]  # trim if over
            else:
                print(f"Failed to generate extra queries on iteration {iteration + 1}")
            iteration += 1
        if len(paraphrased_queries) != repeat:
            print(f"Warning: Only generated {len(paraphrased_queries)} queries out of {repeat} after {iteration} iterations.")
        
        return paraphrased_queries

class ThinkParaphraser:
    def __init__(self, args, generator, tokenizer) -> None:
        self.args = args
        self.generator = generator
        self.tokenizer = tokenizer
        
        self.examples = [
            {
                "original": "Beady Eye was formed in 2009 by former members of the band Oasis. Now, let's find the information about the Minutemen.",
                "paraphrased": [
                    "Beady Eye came together in 2009, created by ex-members of Oasis. With that covered, let's now look into the Minutemen.",
                    "Since Beady Eye was established in 2009 by former Oasis members, we can now move on to exploring the Minutemen.",
                    "Formed in 2009 by Oasis alumni, Beady Eye is now accounted for—let’s shift focus to the Minutemen.",
                    "Having noted that Beady Eye originated in 2009 through ex-Oasis members, the next step is to find details about the Minutemen.",
                    "We’ve confirmed that Beady Eye, formed in 2009 by past Oasis members, is documented—time to turn our attention to the Minutemen.",
                ] 
            },
            {
                "original": "Based on the information provided, the book 'Albion's Seed' by David Hackett Fischer details the four British folkways in America. This matches the query about a story related to Four British Folkways in America.",
                "paraphrased": [
                    "The book 'Albion's Seed' by David Hackett Fischer discusses the four British folkways in America, which aligns with the query asking about a story involving them."
                    "Given the information, 'Albion's Seed' by David Hackett Fischer appears to cover the topic of the Four British Folkways in America, directly relating to the query.",
                    "'Albion's Seed' addresses the theme of the four British folkways in America, making it relevant to the story mentioned in the query.",
                    "The provided content indicates that 'Albion's Seed' is about the four British folkways in America, which corresponds to the subject of the query.",
                    "From the details given, it's clear that Fischer's 'Albion's Seed' covers the Four British Folkways in America, making it a suitable match for the query's topic.",
                ] 
            }
        ]
    
    def get_instruction(self, original_think, n=5):
        input_text = ''
        input_text += "You are an expert in reasoning tasks.\n"
        input_text += f"Given an original reasoning thought, generate {n} semantically diverse and effective paraphrases that preserve the original intent but vary in wording and structure.\n"
        input_text += "These paraphrased thoughts should help improve LLM performance by representing the range of expressions an LLM might use.\n"
        input_text += "Do not introduce any new information in the paraphrased thoughts.\n\n"
        
        input_text += "Here are some examples:\n"
        for example in self.examples:
            input_text += f"<original_think> {example['original']} </original_think>\n"
            for pr_think in example['paraphrased'][:n]:
                input_text += f"<paraphrased_think> {pr_think} </paraphrased_think>\n"
            input_text += '\n'
            
        input_text += f"<original_think> {original_think.strip()} </original_think>\n"
        
        return input_text

    def generate(self,
        input_text,
        max_new_tokens=1024,
        num_return:int = 1,
        temperature:float = 1.0,
        do_sample:bool = True
    ):
        if self.tokenizer.chat_template:
            input_prompt = self.tokenizer.apply_chat_template(
                [{"role": "user", "content": input_text}],
                add_generation_prompt=True,
                tokenize=False
            )
        
        input_ids = self.tokenizer.encode(input_prompt, return_tensors='pt').to(self.generator.device)
        attention_mask = torch.ones_like(input_ids)
        
        generated_texts = []
        for i in range(num_return):
            with torch.no_grad():
                outputs = self.generator.generate(
                    input_ids,
                    attention_mask=attention_mask,
                    max_new_tokens=max_new_tokens,
                    pad_token_id=self.tokenizer.eos_token_id,
                    temperature=temperature,
                    do_sample=do_sample,
                )
                generated_tokens = outputs[0][input_ids.shape[1]:]
                output_text = self.tokenizer.decode(generated_tokens, skip_special_tokens=True)
                generated_texts.append(output_text)
        
        return generated_texts

class RetrievalPerturber:
    def __init__(self, args, retriever) -> None:
        self.args = args
        self.retriever = retriever
    
    def inference(self, query):
        self.retriever.set_topk(5)
        initial_retrieved_docs = self.retriever.search(query)
        perturbed_docs = [initial_retrieved_docs[i] for i in [1, 2, 4]]
        self.retriever.set_topk(self.args.retrieval_topk)
        
        return perturbed_docs

class CriticalThinkGenerator:
    def __init__(self, args, generator, tokenizer) -> None:
        self.args = args
        self.generator = generator
        self.tokenizer = tokenizer
    
        self.examples = [
            {
                "original": "popular industry in the neighborhood of Willow Vale, New South Wales",
                "critical_rethinking": "The query “popular industry in the neighborhood of Willow Vale, New South Wales” is too vague and assumes that a singular dominant industry defines the area, which may not be the case. It also treats “popularity” as a measurable economic indicator without context—ignoring that rural localities like Willow Vale may lack concentrated industry altogether, with economic activity instead dispersed across agriculture, tourism, or small-scale services. Furthermore, neighborhood-level data may not exist for such a small area, rendering most retrieved information speculative or irrelevant regional summaries.",
                "query": "economic activities and land use patterns in Southern Highlands region NSW including Willow Vale"
            },
            {
                "original": "when was the Minnesota Vikings founded",
                "critical_rethinking": "The query 'when was the Minnesota Vikings founded' yields a single factual date, which is trivially available and lacks depth or contextual insight. It also limits exploration into the broader historical, social, or organizational factors that led to the team's creation, making the retrieved information shallow and of limited intellectual value.",
                "query": "historical circumstances behind the founding of the Minnesota Vikings NFL team"
            }
        ]
    
    def extract_tag_content(self, text, tag):
        pattern = re.compile(fr"<{tag}>(.*?)</{tag}>", re.DOTALL)
        matches = pattern.findall(text)
        return matches[0] if matches else None

    def get_instruction(self, original_search_query, n=5):
        input_text = ''
        input_text += "You are a highly capable critical rethinker agent.\n"
        input_text += "Given an original search query, you are tasked to critically assess the search query, and then generate a new and creative search query to support your critical thought.\n"
        input_text += "you are also tasked to return one reasoning thought for the new search query, explaining why the new query works better.\n"
        input_text += "The search query should be precise and focused.\n"
        input_text += "Your output must include:\n"
        input_text += "- One complete reasoning step that strongly rejects the entire retrieved information as unhelpful, irrelevant, or misleading, wrapped in a single pair of <critical_rethinking> and </critical_rethinking> tags.\n"
        input_text += "- One creative and fundamentally new search query, wrapped in <search> and </search> tags.\n\n"
        input_text += "Only use the following format, in this exact order:\n"
        input_text += "<critical_rethinking> one complete reasoning step that strongly rejects the entire retrieved information as unhelpful, irrelevant, or misleading </critical_rethinking>\n"
        input_text += "<search> a creative, focused, and fundamentally new search query </search>\n"
        
        input_text += "Here are some examples:\n"
        for example in self.examples:
            input_text += f"<original_query> {example['original']} </original_query>\n"
            input_text += f"<critical_rethinking> {example['critical_rethinking']} </critical_rethinking>\n"
            input_text += f"<search> {example['query']} </search>\n\n"
        
        input_text += f"<original_query> {original_search_query.strip()} </original_query>\n"
        
        return input_text
    
    def generate_batch(self,
        input_text,
        num_return:int = 5,
        max_new_tokens = 1024,
        temperature:float = 1.0,
        do_sample:bool = True
    ):    
        if self.tokenizer.chat_template:
            input_prompt = self.tokenizer.apply_chat_template(
                [{"role": "user", "content": input_text}],
                add_generation_prompt=True,
                tokenize=False
            )
        
        inputs = self.tokenizer(input_prompt, return_tensors='pt').to(self.generator.device)
        input_ids = inputs["input_ids"]
        batch_size = input_ids.shape[0]
        
        with torch.no_grad():
            model_output = self.generator.generate(
                **inputs,
                return_dict_in_generate=True,
                output_logits=True,
                max_new_tokens=max_new_tokens,
                num_return_sequences=num_return,
                pad_token_id=self.tokenizer.eos_token_id,
                temperature=temperature,
                do_sample=do_sample,
            )
            all_generated = model_output.sequences[:, input_ids.shape[1]:]  # strip prompt

            # Group into batches
            grouped = all_generated.view(batch_size, num_return, -1)
            generated_texts = [
                self.tokenizer.batch_decode(group, skip_special_tokens=True)
                for group in grouped
            ][0]
        
        return generated_texts
    
    def generate_sequential(self,
        input_text,
        max_new_tokens=1024,
        num_return:int = 1,
        temperature:float = 0.7,
        do_sample:bool = True,
        stopping_criteria = None
    ):
        if self.tokenizer.chat_template:
            input_prompt = self.tokenizer.apply_chat_template(
                [{"role": "user", "content": input_text}],
                add_generation_prompt=True,
                tokenize=False
            )
        
        input_ids = self.tokenizer.encode(input_prompt, return_tensors='pt').to(self.generator.device)
        attention_mask = torch.ones_like(input_ids)
        
        generated_texts = []
        for i in range(num_return):
            with torch.no_grad():
                outputs = self.generator.generate(
                    input_ids,
                    attention_mask=attention_mask,
                    max_new_tokens=max_new_tokens,
                    stopping_criteria=stopping_criteria,
                    pad_token_id=self.tokenizer.eos_token_id,
                    temperature=temperature,
                    do_sample=do_sample,
                )
                generated_tokens = outputs[0][input_ids.shape[1]:]
                output_text = self.tokenizer.decode(generated_tokens, skip_special_tokens=True)
                generated_texts.append(output_text)
        
        return generated_texts
    
    def inference(self, qid, original_sq, repeat=5):
        input_prompt = self.get_instruction(original_sq, n=repeat)
        output_texts = self.generate_batch(input_prompt, num_return=repeat, temperature=1.0)
        thinks, queries = [], []
        for output_text in output_texts:
            critical_thinking = self.extract_tag_content(output_text, "critical_rethinking")
            critical_query = self.extract_tag_content(output_text, "search")
            if critical_thinking and critical_query:
                thinks.append(critical_thinking)
                queries.append(critical_query)
        
        # assert len(thinks) == repeat and len(queries) == repeat, f"Expected {repeat} items, got {len(thinks)} thinks and {len(queries)} queries"
        if len(thinks) < repeat or len(queries) < repeat:
            print(f"[Warning] Expected {repeat} items, got {len(thinks)} thinks and {len(queries)} queries. Padding with empty strings.")
            missing = repeat - len(thinks)
            thinks += [""] * missing
            queries += [""] * missing
        
        
        return thinks, queries

class AnswerValidator:
    def __init__(self, args, generator, tokenizer) -> None:
        self.args = args
        self.generator = generator
        self.tokenizer = tokenizer
        self.examples = []
    
    def get_unique_docs(self, docs_lst:list):
        return list({doc['id']: doc for doc in docs_lst}.values()) 
    
    def get_information_summary(self, text):
        pattern = re.compile(r"<summary>(.*?)</summary>", re.DOTALL)
        matches = pattern.findall(text)
        if matches:
            return matches[0]
        else:
            return ''
    
    def get_think(self, text):
        pattern = re.compile(r"<think>(.*?)</think>", re.DOTALL)
        matches = pattern.findall(text)
        if matches:
            return matches[-1]
        else:
            return ''

    def get_query(self, text):
        pattern = re.compile(r"<search>(.*?)</search>", re.DOTALL)
        matches = pattern.findall(text)
        if matches:
            return matches[-1]
        else:
            return ''
    
    def generate_batch(self,
        input_text,
        num_return:int = 5,
        max_new_tokens = 1024,
        temperature:float = 1.0,
        do_sample:bool = True
    ):    
        if self.tokenizer.chat_template:
            input_prompt = self.tokenizer.apply_chat_template(
                [{"role": "user", "content": input_text}],
                add_generation_prompt=True,
                tokenize=False
            )
        
        inputs = self.tokenizer(input_prompt, return_tensors='pt').to(self.generator.device)
        input_ids = inputs["input_ids"]
        batch_size = input_ids.shape[0]
        
        with torch.no_grad():
            model_output = self.generator.generate(
                **inputs,
                return_dict_in_generate=True,
                output_logits=True,
                max_new_tokens=max_new_tokens,
                num_return_sequences=num_return,
                pad_token_id=self.tokenizer.eos_token_id,
                temperature=temperature,
                do_sample=do_sample,
            )
            all_generated = model_output.sequences[:, input_ids.shape[1]:]  # strip prompt

            # Group into batches
            grouped = all_generated.view(batch_size, num_return, -1)
            generated_texts = [
                self.tokenizer.batch_decode(group, skip_special_tokens=True)
                for group in grouped
            ][0]
        
        return generated_texts
    
    def generate_sequential(self,
        input_text,
        max_new_tokens=1024,
        num_return:int = 1,
        temperature:float = 0.7,
        do_sample:bool = True,
        stopping_criteria = None
    ):
        if self.tokenizer.chat_template:
            input_prompt = self.tokenizer.apply_chat_template(
                [{"role": "user", "content": input_text}],
                add_generation_prompt=True,
                tokenize=False
            )
        
        input_ids = self.tokenizer.encode(input_prompt, return_tensors='pt').to(self.generator.device)
        attention_mask = torch.ones_like(input_ids)
        
        generated_texts = []
        for i in range(num_return):
            with torch.no_grad():
                outputs = self.generator.generate(
                    input_ids,
                    attention_mask=attention_mask,
                    max_new_tokens=max_new_tokens,
                    stopping_criteria=stopping_criteria,
                    pad_token_id=self.tokenizer.eos_token_id,
                    temperature=temperature,
                    do_sample=do_sample,
                )
                generated_tokens = outputs[0][input_ids.shape[1]:]
                output_text = self.tokenizer.decode(generated_tokens, skip_special_tokens=True)
                generated_texts.append(output_text)
        
        return generated_texts
    
    def get_summarization_instruction(self, question, docs_text):
        input_text = ''
        input_text += "You are a highly capable summarization agent.\n"
        input_text += "Your task is to generate a detailed summary based on the provided information and the user's query.\n"
        input_text += "Your output must include:\n"
        input_text += "- One complete and detailed summary, wrapped in a single pair of <summary> and </summary> tags.\n"
        input_text += "Your output must follow this exact format and order:\n"
        input_text += "<summary> one complete summary of the information considering the user query </summary>\n\n"
        
        input_text +=f"<information> {docs_text} </information>\n"
        input_text += f"<user_query> {question.strip()} </user_query>\n"
        
        return input_text

    def get_validation_instruction(self, question, prediction, info_summary):
        input_text = ''
        input_text += "You are a highly capable response validation agent.\n"
        input_text += "Given a user query and a summary of the retrieved documents used during the reasoning process, your task is to verify whether the generated response satisfies two evaluation criteria.\n"
        input_text += "The two criteria are: 1) Is the response grounded in the provided information? 2) Does the response correctly and fully answer the user query?\n"
        input_text += "You must provide a single, coherent reasoning step that examines both criteria and suggests how the response could be improved.\n"
        input_text += "After your reasoning, you must return a precise search query that can help retrieve better information to improve the answer.\n"
        input_text += "The document summary will be enclosed in <information> and </information> tags. This content is read-only: NEVER generate, modify, or repeat the <information> tags.\n"
        input_text += "The predicted answer will be enclosed in <prediction> and </prediction> tags. This content is also read-only: NEVER generate, modify, or repeat the <prediction> tags.\n"
        input_text += "The search query must be focused, informative, and aimed at enhancing the predicted answer.\n"

        input_text += "Your output must include:\n"
        input_text += "- One complete reasoning step that: (a) references the predicted answer, (b) evaluates it using the two criteria, and (c) proposes specific improvements. Wrap this reasoning inside a single pair of <think> and </think> tags.\n"
        input_text += "- One search query that would help improve the answer. Wrap it inside <search> and </search> tags.\n\n"

        input_text += "Only use the following format, in this exact order:\n"
        input_text += "<think> one complete reasoning step that assesses the answer </think>\n"
        input_text += "<search> a creative and focused search query </search>\n"

        input_text += f"Question: {question}\n"
        input_text += f"<information> {info_summary} </information>\n"
        input_text += f"<prediction> {prediction} </prediction>\n"
        
        return input_text

    def summarization_inference(self, qid, question, trace, repeat=5):
        summarizations = []
        all_docs = [doc for step in trace[:-1] if 'docs' in step for doc in step['docs']]
        unq_docs = self.get_unique_docs(all_docs)
        docs_text = passages2string(unq_docs)
        
        input_prompt = self.get_summarization_instruction(question, docs_text)
        output_texts = self.generate_batch(input_prompt, num_return=repeat, temperature=self.args.consistency_temperature)
        summarizations = [self.get_information_summary(output_text) for output_text in output_texts]
        assert len(summarizations) == repeat, f"For query {qid}, expected {repeat} items, got {len(summarizations)} summarizations!!!"
    
        return summarizations
        
    def validation_inference(self, qid, question, prediction, summarization_list, repeat=5):
        thinks, queries = [], []
        for info_summary in summarization_list:
            input_prompt = self.get_validation_instruction(question, prediction, info_summary)
            output_text = self.generate_sequential(input_prompt, num_return=1, temperature=self.args.consistency_temperature)[0]
            thinks.append(self.get_think(output_text))
            queries.append(self.get_query(output_text))
        
        assert len(thinks) == repeat and len(queries) == repeat, f"Expected {repeat} items, got {len(thinks)} thinks and {len(queries)} queries"
        return thinks, queries







# class AnalysisThinkGenerator:
#     def __init__(self, args, generator, tokenizer) -> None:
#         self.args = args
#         self.generator = generator
#         self.tokenizer = tokenizer
    
#         self.examples = [
#             {
#                 "original": "popular industry in the neighborhood of Willow Vale, New South Wales",
#                 "think": "",
#                 "query": ""
#             },
#             {
#                 "original": "when was the Minnesota Vikings founded",
#                 "think": "",
#                 "query": ""
#             }
#         ]
        
    
#     def get_instruction(self, original_search_query, n=5):
#         input_text = ''
#         input_text += "You are a highly capable query paraphraser agent.\n"
#         input_text += "Given an original search query, generate one semantically diverse and effective paraphrased search query that capture the same intent but use different wording or structure.\n"
#         input_text += "The paraphrased query should be suitable for improving search engine results by covering various phrasings a user might employ.\n"
#         input_text += "You are also tasked with providing one reasoning step explaining why the new query is effective.\n"
#         input_text += "The search query must be precise and focused.\n"
#         input_text += "Do not add extra information that is not present in the original query.\n\n"
        
#         input_text += "Your output must follow this exact format and order:\n"
#         input_text += "<think> One complete reasoning step explaining why the new query works better </think>\n"
#         input_text += "<search> One creative, focused, and new search query </search>\n\n"
        
#         input_text += "Here are some examples:\n"
#         for example in self.examples:
#             input_text += f"<original_query> {example['original']} </original_query>\n"
#             input_text += f"<think> {example['think']} </think>\n"
#             input_text += f"<search> {example['search_query']} </search>\n\n"
        
#         input_text += f"<original_query> {original_search_query.strip()} </original_query>\n"
        
#         return input_text
