from tree_of_thoughts.openaiModels import OpenAILanguageModel, OptimizedOpenAILanguageModel
from tree_of_thoughts.treeofthoughts import TreeofThoughts, OptimizedTreeofThoughts

use_v2 = True
api_key=""
api_base= "" # leave it blank if you simply use default openai api url
api_model= "gpt-3.5-turbo"


model = OptimizedOpenAILanguageModel(api_key=api_key, api_base=api_base, api_model=api_model)
#choose search algorithm('BFS' or 'DFS')
search_algorithm = "BFS"



# value or vote
evaluation_strategy = "value"

if not use_v2:
    #create an instance of the tree of thoughts class v1
    tree_of_thoughts = TreeofThoughts(model, search_algorithm)
else:
    #or v2 -> dynamic beam width -< adjust the beam width [b] dynamically based on the search depth quality of the generated thoughts
    tree_of_thoughts= OptimizedTreeofThoughts(model, search_algorithm)

input_problem = "use 4 numbers and basic arithmetic operations (+-*/) to obtain 24"
k = 5
T = 3
b = 5
vth = 0.5
pruning_threshold = 0.5
timeout = 10
confidence = 0.8 #cmodel is confident on performance
max_iterations = 40 #tree branh nodes 
convergence_threshold = 0.01
convergence_count = 5

#call the solve emthod with the input problem and other params
solution = tree_of_thoughts.solve(input_problem, k, T, b, vth, timeout, pruning_threshold, confidence, max_iterations, convergence_threshold, convergence_count)
    
                  
    
#use the solution in your production environment
print(f"solution: {solution}")


# Save the tree and metrics to a JSON file
file_name = "logs/tree_of_thoughts_output.json"
tree_of_thoughts.save_tree_to_json(file_name)

    # k = 1#number of thoughts to input
    # T = 1 # maximum depth of the search tree
    # b = 1 # branching factor -< number of child nodes for each branch
    # vth = 0.9 # pruning state -> any evaluated thought below this is eliminated
    # timeout = 10 #10 seconds timeout before stop
    # confidence = 0.8 #cmodel is confident on performance
    # max_iterations = 40 #tree branch nodes 
    # convergence_threshold = 0.01 #determining when the search process has converged
    # convergence_count = 5 # number of searchers to be considered converged
    # # read documentation for more
