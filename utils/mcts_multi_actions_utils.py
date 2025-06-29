from enum import Enum, unique
from colorama import Fore, Style

def verbose_print(s: str, verbose: bool):
    if verbose:
        print(s)

@unique
class Node_Type(Enum):
    USER_QUERY = "USER_QUERY"
    RETRIEVE_DOCUMENTS = "RETRIEVE_DOCUMENTS"
    DOCUMENTS_ANALYSIS = "DOCUMENTS_ANALYSIS"
    DOCUMENTS_RETHINKING = "DOCUMENTS_RETHINKING"
    ANSWER_GENERATION = "ANSWER_GENERATION"
    ANSWER_VALIDATION = "ANSWER_VALIDATION"
    FINISH = "FINISH"

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
        if node.node_type is Node_Type.FINISH:
            return node.is_finished
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

        if node.node_type is Node_Type.USER_QUERY:
            node_details += f"User: {node.user_query} | Ground truth: {", ".join(node.gt_answers)} | Path: {node.gt_reasoning_steps}" +  "\n" + space + " " * len(node_info) 
        elif node.node_type is Node_Type.RETRIEVE_DOCUMENTS:
            node_details += f"Search: {node.search_query} | Think: {node.think.replace("\n", " ")}" +  "\n" + space + " " * len(node_info)
        elif node.node_type is Node_Type.DOCUMENTS_ANALYSIS:
            node_details += f"Search: {node.search_query} | Analysis: {node.documents_analysis.replace("\n", " ")}" +  "\n" + space + " " * len(node_info)
        elif node.node_type is Node_Type.DOCUMENTS_RETHINKING:
            node_details += f"Search: {node.search_query} | Rethinking: {node.critical_rethinking.replace("\n", " ")}" +  "\n" + space + " " * len(node_info)
        elif node.node_type is Node_Type.ANSWER_GENERATION:
            node_details += f"Answer: {node.answer} | Think: {node.think.replace("\n", " ")}" +  "\n" + space + " " * len(node_info)
        elif node.node_type is Node_Type.ANSWER_VALIDATION:
            node_details += f"Validation: {node.answer_validation.replace("\n", " ")}" +  "\n" + space + " " * len(node_info)
        elif node.node_type is Node_Type.FINISH:
            node_details += f"Finish: {node.is_finished}" +  "\n" + space + " " * len(node_info)

        to_print += dash + node_details
        my_print(to_print)

        for child in node.children:
            print_tree(node, child, file, rollout_id)
        if node.depth == 0:
            my_print("\n" + "=" * 50 + "\n")

    print_tree(parent_node=None, node=root_node, file=file, rollout_id=rollout_id)
