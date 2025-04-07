from enum import Enum, unique
from colorama import Fore, Style

def verbose_print(s: str, verbose: bool):
    if verbose:
        print(s)

@unique
class Node_Type(Enum):
    USER_QUESTION = "USER_QUESTION"
    DIRECT_ANSWER = "DIRECT_ANSWER"
    RAG_ANSWER = "RAG_ANSWER"
    SUBQUESTIONS = "SUBQUESTIONS"
    SUBQ_DIRECT_ANSWER = "SUBQ_DIRECT_ANSWER"
    SUBQ_RAG_ANSWER = "SUBQ_RAG_ANSWER"

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
    evaluator,
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
        attributes += f"V: {round(node.node_value, 2)}" if node.node_value is not None else "V: None"

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
            node_details += f"User: {node.user_question}" + "  " + f"Ground truth: {gt}" +  "\n" + space + " " * len(node_info) 
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
    
    
        to_print += dash + node_details
        my_print(to_print)

        for child in node.children:
            print_tree(node, child, file, rollout_id)
        if node.depth == 0:
            my_print("\n" + "=" * 50 + "\n")

    print_tree(parent_node=None, node=root_node, file=file, rollout_id=rollout_id)

