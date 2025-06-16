
PTRUE_SYSTEM_PROMPT = 'You are a helpful, respectful and honest question-answer evaluator. You will be given a question, some brainstormed ideas and a generated answer. Evaluate the generate answer as true or false considering the question and brainstormed ideas. Output "The generated answer is true" or "The generated answer is false".'
PTRUE_USER_PROMPT = "Question: {question}\nHere are some ideas that were brainstormed:\n{ideas}\nGenerated answer:{generated_text}"
PTRUE_USER_PROMPT_WITH_CONTEXT = "Context: {context}\nQuestion: {question}\nHere are some ideas that were brainstormed:\n{ideas}\nGenerated answer:{generated_text}"
PTRUE_MODEL_OUTPUT = "The generated answer is true"