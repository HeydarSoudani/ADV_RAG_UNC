import math
from typing import Dict, Tuple
from enum import Enum, unique
from colorama import Fore, Style
from run_searchr1.inference import _passages2string

def verbose_print(s: str, verbose: bool):
    if verbose:
        print(s)

@unique
class Node_Type(Enum):
    USER_QUESTION = "USER_QUESTION"
    REPHRASED_QUERY = "REPHRASED_QUERY"
    DIRECT_ANSWER = "DIRECT_ANSWER"
    RAG_ANSWER = "RAG_ANSWER"
    SUBQUESTIONS = "SUBQUESTIONS"
    SUBQ_DIRECT_ANSWER = "SUBQ_DIRECT_ANSWER"
    SUBQ_RAG_ANSWER = "SUBQ_RAG_ANSWER"
    
    THINK_SERACH = "THINK_SERACH"
    THINK_ANSWER = "THINK_ANSWER"
    CRITIQUE_SEARCH = "CRITIQUE_SEARCH"
    CRITIQUE_ANSWER = "CRITIQUE_ANSWER"

def split_user_question(user_question: str):
    user_question = user_question.strip().rstrip(".")
    last_period_id = user_question.rfind(".")
    assert last_period_id < len(user_question) - 1
    user_question_context = user_question[: last_period_id + 1].strip()
    user_question_problem = user_question[last_period_id + 1 :].strip()
    return user_question_context, user_question_problem

def reach_terminal_subquestion(subquestion: str, user_question: str):
    assert subquestion is not None
    if "Now we can answer" in subquestion:
        #! remember that: when the original question is answerable, please start the subquestion with "Now we can answer the question: "
        return True
    user_question_2nd_part = split_user_question(user_question)[1]
    if user_question_2nd_part.lower() in subquestion.lower():
        return True
    return False
    
def find_valid_solution_nodes(root_node):
    valid_solution_nodes = []

    def recursion(node):
        if node.is_valid_solution_node():
            valid_solution_nodes.append(node)
            return

        if not node.children:  #! no children
            return

        for child in node.children:
            recursion(child)

    recursion(root_node)

    return valid_solution_nodes

def stochastic_find_best_solution(
    root_node,
    # evaluator,
    enable_potential_score,
):
    # todo: what strategy do we use to select best node?
    """The function finds the best solution from the solution nodes in the MCTS tree.
    Return: top answer, top solution, confidence of the top answer, the corresponding node of the answer, all solution nodes
    """
    solution_nodes = find_valid_solution_nodes(root_node)

    if len(solution_nodes) == 0:
        return None, None

    def extract_solution_from_node(node):
        if node.node_type is Node_Type.DIRECT_ANSWER:
            return node.direct_answer
        else:
            return None

    solutions = [extract_solution_from_node(node) for node in solution_nodes]
    return solution_nodes, solutions

def print_tree_from_root(mcts_searcher, rollout_id, root_node, chosen_node=None, file=None):
    color_print = False if file else True

    def my_print(text):
        if file:
            file.write(text + "\n")
        else:
            print(text)

    def print_tree(parent_node, node, file, rollout_id):
        to_print = ""

        num_indent = 4
        dash = "-" * num_indent * node.depth
        space = " " * num_indent * node.depth

        attributes = f"Q: {round(mcts_searcher.Q[node], 2)}" + "; " + f"N: {mcts_searcher.N[node]}" + "; "
        attributes += f"V: {round(node.node_reward, 2)}" if node.node_reward is not None else "V: None"

        uct_value = "UCT: " + str(
            round(mcts_searcher._compute_uct(parent_node=parent_node, node=node, rollout_id=rollout_id), 2)
        )
        attributes += "; " + uct_value

        solution_marker = "(T) " if node.is_valid_solution_node() else ""

        node_info = "[" + solution_marker + node.__str__() + ": " + attributes + "]"
        if chosen_node and node == chosen_node:
            node_info = "[" + node_info + "]"
        node_info += " "

        if color_print and node.is_valid_solution_node():
            node_details = Fore.RED + Style.BRIGHT + node_info + Fore.RESET + Style.RESET_ALL
        else:
            node_details = node_info

        if node.node_type is Node_Type.USER_QUESTION:
            gt = ", ".join(node.gt_answer)
            node_details += f"User: {node.user_question} | Ground truth: {gt} | Candidates: {node.answer_candidates} | Path: {node.gt_reasoning_steps}" +  "\n" + space + " " * len(node_info) 
        elif node.node_type is Node_Type.DIRECT_ANSWER:
            node_details += f"Ans: {node.direct_answer.replace("\n", " ")}"
        elif node.node_type is Node_Type.RAG_ANSWER:
            node_details += f"Ans: {node.rag_answer.replace("\n", " ")}" + f" Doc: ..." + "  " + "\n" + space + " " * len(node_info) # {node.retrieved_document}
        elif node.node_type is Node_Type.SUBQUESTIONS:
            node_details += f"Sub-Qs: {node.subquestions}" + "\n" + space + " " * len(node_info)
        elif node.node_type is Node_Type.SUBQ_DIRECT_ANSWER:
            node_details += f"Query: {node.subquestion}" + "  Ans: " + f"{node.subanswer.replace("\n", " ")}" + f"  Pointer: {node.subquestion_pointer}" + "\n" + space + " " * len(node_info)
        elif node.node_type is Node_Type.SUBQ_RAG_ANSWER:
            node_details += f"Query: {node.subquestion}" + "  Ans: " + f"{node.subanswer.replace("\n", " ")}" + f"  Pointer: {node.subquestion_pointer}" + f"  Docs: ..." +  "\n" + space + " " * len(node_info) # {node.subq_retrieved_documents}
        elif node.node_type is Node_Type.REPHRASED_QUERY:
            node_details += f"Rep-Query: {node.rephrased_query}" + "\n" + space + " " * len(node_info)
    
        elif node.node_type is Node_Type.THINK_SERACH:
            node_details += f"Search: {node.search_query} | Think: {node.think.replace("\n", " ")}" +  "\n" + space + " " * len(node_info)
        elif node.node_type is Node_Type.THINK_ANSWER:
            node_details += f"Answer: {node.answer} | Think: {node.think.replace("\n", " ")}" + "\n" + space + " " * len(node_info)
        elif node.node_type is Node_Type.CRITIQUE_SEARCH:
            node_details += f"Search: {node.search_query} | Critique: {node.critique.replace("\n", " ")}" +  "\n" + space + " " * len(node_info)
        elif node.node_type is Node_Type.CRITIQUE_ANSWER:
            node_details += f"Answer: {node.answer} | Critique: {node.critique.replace("\n", " ")}" + "\n" + space + " " * len(node_info)

    
        to_print += dash + node_details
        my_print(to_print)

        for child in node.children:
            print_tree(node, child, file, rollout_id)
        if node.depth == 0:
            my_print("\n" + "=" * 50 + "\n")

    print_tree(parent_node=None, node=root_node, file=file, rollout_id=rollout_id)

def concat_solution_trace(solution_trace: Dict[int, Dict[str, str]]):
    """Note that the solution trace might be subqs-subas and also one-step thought steps."""
    solution_trace_str = ""
    final_step_str = ""
    end_node_type = None
    reward_value = 0.0

    for item_idx in solution_trace:
        solution_item = solution_trace[item_idx]
        keys = list(solution_item.keys())
        
        # else:  # answer step
        node_type = keys[0]
        # non-answer step
        if node_type == 'user_question':
            solution_trace_str += f'Given the question: \n{solution_item[node_type]}\n'
        elif node_type == 'rephrased_query':
            solution_trace_str += f'We rephrase the question, which can also be expressed as: \n{solution_item[node_type]}\n'
        elif node_type == 'subquestions':
            solution_trace_str += f'We then decompose the question into several sub-questions, namely: \n{", ".join(solution_item[node_type])}\n'
        elif node_type in ['subq_direct_answer', 'subq_rag_answer']:
            solution_trace_str += f'We then generate answers for the sub-questions, namely: \n{solution_item[node_type]['subquestion']} {solution_item[node_type]['subanswer']}\n'
        
        # answer step
        elif node_type == 'rag_answer':
            solution_trace_str += f'For this type of question, we retrieve a series of relevant documents, referred to as:\n'
            solution_trace_str += '<document>\n'
            for i, doc in enumerate(solution_item[node_type]['documents']):
                solution_trace_str += f'[{i+1}] {doc}\n'
            solution_trace_str += '</document>\n'
            solution_trace_str += f'\nSummarizing the information above, now we extract the answer, the answer is: \n<Answer>\n{solution_item[node_type]['text']}\n</Answer>'    
            final_step_str = solution_item[node_type]['text']
            end_node_type = Node_Type.RAG_ANSWER
            reward_value = solution_item[node_type]['value']
        
        elif node_type == 'direct_answer':
            solution_trace_str += f'\nSummarizing the information above, now we extract the answer, the answer is: \n<Answer>\n{solution_item[node_type]['text']}\n</Answer>'    
            final_step_str = solution_item[node_type]['text']
            end_node_type = Node_Type.DIRECT_ANSWER
            reward_value = solution_item[node_type]['value']

    return solution_trace_str.strip(), final_step_str.strip(), end_node_type, min(0, reward_value) + 1

def concat_solution_trace_v2(solution_trace: Dict[int, Dict[str, str]]):
    """Note that the solution trace might be subqs-subas and also one-step thought steps."""
    solution_trace_str = ""
    final_step_str = ""
    end_node_type = None
    reward_value = 0.0

    for item_idx in solution_trace:
        solution_item = solution_trace[item_idx]
        keys = list(solution_item.keys())
        node_type = keys[0]
        
        if node_type == 'user_question':
            solution_trace_str += f'Question: \n{solution_item[node_type]}\n'
        
        elif node_type == 'think_search':
            solution_trace_str += f'<think> {solution_item[node_type]['think']} </think>\n'
            solution_trace_str += f'<search> {solution_item[node_type]['search_query']} </search>\n'
            solution_trace_str += f'<information> {_passages2string(solution_item[node_type]['retrieved_documents'])}<\information>\n'
        
        elif node_type == 'think_answer':
            solution_trace_str += f'<think> {solution_item[node_type]['think']} </think>\n'
            solution_trace_str += f'<answer> {solution_item[node_type]['answer']} </answer>\n'
            
            final_step_str = solution_item[node_type]['answer']
            end_node_type = Node_Type.THINK_ANSWER
            reward_value = solution_item[node_type]['value']
            

    return solution_trace_str.strip(), final_step_str.strip(), end_node_type, min(0, reward_value) + 1
    
def mask_solution_trace(
    solution_trace_str: str, num_return: int, left_boundary: float, right_boundary: float
) -> list[str]:
    # opasdjifpoaisdfjpoasidfjapsodifj, num_return: 4, left: 0.2, right: 0.8
    # return: opasd, opasdjifp, opasdjifpoaisdfj, opasdjifpoaisdfjpoasidfjaps
    if num_return == 1:
        interval = 0
    else:
        assert num_return > 1
        assert right_boundary >= left_boundary, f"right_boundary: {right_boundary} < left_boundary: {left_boundary}"
        interval = (right_boundary - left_boundary) / (num_return - 1)

    words_in_solution_trace = solution_trace_str.split(" ")
    ost_len = len(words_in_solution_trace)
    # Mask the solution trace string from least to most
    masked_solution_traces = []
    for i in range(num_return):
        prefix_part_ratio = left_boundary + i * interval
        prefix_part_num_words = math.ceil(ost_len * prefix_part_ratio)
        prefix_part_str = " ".join(words_in_solution_trace[:prefix_part_num_words])
        masked_solution_traces.append(prefix_part_str)

    return masked_solution_traces


def rag_mask_solution_trace(
    solution_trace_str: str, num_return: int, left_boundary: float, right_boundary: float
) -> list[str]:
    # only mask the reasoning steps behind the retrieved documents
    if num_return == 1:
        interval = 0
    else:
        assert num_return > 1
        assert right_boundary >= left_boundary, f"right_boundary: {right_boundary} < left_boundary: {left_boundary}"
        interval = (right_boundary - left_boundary) / (num_return - 1)

    words_in_solution_trace = solution_trace_str.split(" ")
    
    last_position = next((idx for idx in range(len(words_in_solution_trace) - 1, -1, -1) if "</information>" in words_in_solution_trace[idx]), -1)
    mask_len = len(words_in_solution_trace[last_position+1:])
    
    # Mask the solution trace string from least to most
    masked_solution_traces = []
    for i in range(num_return):
        prefix_part_ratio = left_boundary + i * interval
        prefix_part_num_words = last_position + math.ceil(mask_len * prefix_part_ratio) + 1
        prefix_part_str = " ".join(words_in_solution_trace[:prefix_part_num_words])
        masked_solution_traces.append(prefix_part_str)

    return masked_solution_traces


