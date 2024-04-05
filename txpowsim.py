import simpy
import networkx as nx
import matplotlib.pyplot as plt
import random
import time
import copy
import networkx as nx
import matplotlib.pyplot as plt
import sys
import json
import threading  # Import the threading library


sys.set_int_max_str_digits(0)  # Set to no limit (use with caution)

class Transaction:
    def __init__(self, originator, timestamp, tx_hash, tx_pow, target_difficulty=None, sender=None):
        self.originator = originator
        self.timestamp = timestamp
        self.tx_hash = tx_hash
        self.tx_pow = tx_pow  # Newly added field
        self.hops = 0  # Initialize hops to 0
        self.hop_limit = 0  # Initialize the hop limit
        self.target_difficulty = target_difficulty  # New attribute
        self.sender = sender


    def __str__(self): 
        return f"Transaction(originator={self.originator}, timestamp={self.timestamp}, txhash={self.tx_hash}, tx_pow={self.tx_pow})"

def create_blocks(G):  # Take the network graph as input
    while True:
        selected_node = random.choice(list(G.nodes()))  # Random node selection

        # Transaction Selection (with dynamic limit)
        transaction_limit = 100
        if len(selected_node['seen_transactions']) > 200:
            transaction_limit = 150
        oldest_transactions = sorted(selected_node['seen_transactions'], key=lambda tx: tx['timestamp'])[:transaction_limit]

        # Block Formation
        block_id = generate_block_id() 
        timestamp =  time.time()
        block = {
            'block_id': block_id,
            'timestamp': timestamp,
            'creating_node': selected_node,  # Store the node ID directly
            'transactions': oldest_transactions
        }

        # File Storage
        with open('blockchain.txt', 'a') as f: 
            f.write(json.dumps(block) + '\n')

        # Transaction Clearing
        for tx in oldest_transactions:
            for node_id in G.nodes():  # Iterate over node IDs
                node = G.nodes[node_id]  # Get the node object
                if tx in node['seen_transactions']:
                    node['seen_transactions'].remove(tx)

        time.sleep(120) 

block_creation_thread = threading.Thread(target=create_blocks, args=(G,))
block_creation_thread.start()

def generate_mesh_network(num_nodes, min_peers=12, max_peers=24, user_type_ratio=0.1):
    """
    Generates a mesh network with the specified parameters.

    Args:
        num_nodes: The number of nodes in the network.
        min_peers: The minimum number of peers for each node.
        max_peers: The maximum number of peers for each node.

    Returns:
        A NetworkX graph object representing the mesh network.
    """
    G = nx.Graph()

    # Add nodes to the graph
    for i in range(num_nodes):
        G.add_node(i)

    # Initialize nodes with a starting PoW difficulty

    for node in G.nodes():
        G.nodes[node]['transaction_timestamps'] = {} 
        G.nodes[node]['repackage_probabilities'] = {}  # Initialize with probability
        G.nodes[node]['txPoW_mindiff'] = 5 
        G.nodes[node]['seen_transactions'] = set()  
        G.nodes[node]['neighbors'] = {} 

        if random.random() < user_type_ratio:
            G.nodes[node]['user_type'] = 'spammer'
            # Initialize a list to store dynamically chosen spam targets
            G.nodes[node]['spam_targets'] = []
        else:
            G.nodes[node]['user_type'] = 'normal'

    # Create edges while ensuring the peer count constraints

    for node in G.nodes():
        num_peers_to_add = random.randint(min_peers, max_peers)
        potential_peers = [n for n in G.nodes() if n != node and (n not in G.neighbors(node))]

        for _ in range(num_peers_to_add):
            if potential_peers:
                peer = random.choice(potential_peers)
                G.add_edge(node, peer)
                potential_peers.remove(peer)

                # Update neighbor dictionaries for both nodes
                G.nodes[node]['repackage_probabilities'].setdefault(peer, {'probability': 1.0}) # Initialize with probability
                G.nodes[node]['neighbors'][peer] = {'txPoW_mindiff': G.nodes[peer]['txPoW_mindiff']} 

                G.nodes[peer]['repackage_probabilities'].setdefault(node, {'probability': 1.0}) # Initialize with probability
                G.nodes[peer]['neighbors'][node] = {'txPoW_mindiff': G.nodes[node]['txPoW_mindiff']} 

            else:
                break
    
    # Assign spam targets to spammer nodes (after network creation)
    for node in G.nodes():
        if G.nodes[node]['user_type'] == 'spammer':
            # Select a random subset of size 'spam_target_subset_size' from all nodes
            spam_target_subset_size = 12
            G.nodes[node]['spam_targets'] = random.sample(list(G.nodes()), spam_target_subset_size)

    return G

generation_probability = 0.9  # Example probability (adjust as needed)
def generate_transaction(env, node, network):
    transaction = None  # Initialize before the conditionals
    global num_transactions_generated  # Declare the global variable
    min_stem_hops = 5  # Minimum number of hops in stem phase
    max_stem_hops = 9  # Maximum number of hops in stem phase
    random_hops = random.randint(min_stem_hops, max_stem_hops) 
 
    if random.random() < generation_probability:  # Overall probability
        user_type = network.nodes[node]['user_type']
        if user_type == 'normal':
            # Normal user behavior (you can adjust these parameters)
            yield env.timeout(random.expovariate(1.0))  # Longer interval between transactions
            # Select any random neighbor for transaction creation
            neighbor = random.choice(list(network.neighbors(node)))
            neighbor_mindiff = network.nodes[neighbor]['txPoW_mindiff']  # Retrieve neighbor's difficulty
            if neighbor_mindiff > MAX_POW_DIFF:
                print("DIFF TOO HIGH DURING CREATION")
        # Handle exceeding maximum initial difficulty (e.g., transaction failure)
                return None  # Or raise an exception, depending on your logic

            transaction_hash = hash(str(node) + str(env.now))  # Generate hash based on node and time
            short_hash = str(transaction_hash)[:10] # Take the first 10 characters of the hash
            tx = Transaction(node, env.now, short_hash, network.nodes[node]['txPoW_mindiff'], target_difficulty=neighbor_mindiff)
            tx.hop_limit = random_hops  # Add the hop limit to the Transaction object
        else:  # Spammer behavior
            # Spammer behavior (adjust these parameters)
            yield env.timeout(random.expovariate(0.5))  # More frequent transactions
            # Select a random neighbor from the pre-defined list of spam_targets
            neighbor = random.choice(network.nodes[node]['spam_targets'])
            very_low_difficulty=5
            transaction_hash = hash(str(node) + str(env.now))  # Generate hash based on node and time
            short_hash = str(transaction_hash)[:10] # Take the first 10 characters of the hash
            tx = Transaction(node, env.now, short_hash, network.nodes[node]['txPoW_mindiff'], target_difficulty=very_low_difficulty)
            tx.hop_limit = random_hops  # Add the hop limit to the Transaction object


        # You'll need to implement logic to actually create and send the transaction here
        # (considering PoW checks in the future)
        print(f"Node {node} trying to generate a transaction {tx} (user type: {user_type})")
        num_transactions_generated += 1  # Increment the counter after a transaction is generated
        env.process(process_transaction(env, node, network, copy.deepcopy(tx)))


def process_transaction(env, node, network, transaction):
    delay = random.uniform(0.005, 0.02)  # Simulate processing and transmission time
    yield env.timeout(delay) 
    print(f"Node {node} received transaction {transaction.tx_hash} from originator {transaction.originator}")

    if transaction.tx_hash in network.nodes[node]['seen_transactions']:
        print(f"######################### Node {node} has already received transaction {transaction.tx_hash} from originator {transaction.originator}, sent by sender {transaction.sender}")
        return  # If transaction already seen, stop processing

    #sender = node  # Identify the neighbor
    transaction.sender = node  # Update the sender before forwarding
    sender = transaction.sender #instead of modding the other calls

    print("Before check:", sender)  # Add this line

    network.nodes[node]['repackage_probabilities'].setdefault(sender, {})['last_transaction_time'] = env.now
    print("After check:", sender)  # Add this line

    # Timestamp Management
    current_time = env.now
    network.nodes[node]['transaction_timestamps'].setdefault(sender, []).append(current_time)  

    # Clean up old timestamps
    time_window = 1.0  # Your desired time window in seconds
    cutoff_time = current_time - time_window
    network.nodes[node]['transaction_timestamps'][sender] = [
        timestamp for timestamp in network.nodes[node]['transaction_timestamps'][sender] 
        if timestamp >= cutoff_time
    ]

    # Difficulty Adjustment  
    transaction_rate = calculate_transaction_rate(network.nodes[node]['transaction_timestamps'][sender]) 
    if transaction_rate > 2:  # Threshold of 2 transactions per second 
        new_min_diff = calculate_new_min_diff(network.nodes[node]['txPoW_mindiff'])  
        network.nodes[node]['txPoW_mindiff'] = new_min_diff

    # Dandelion Propagation Logic 
    if transaction.tx_hash not in network.nodes[node]['seen_transactions']:
        network.nodes[node]['seen_transactions'].add(transaction.tx_hash)

        if is_in_stem_phase(transaction, node, network): 
            transaction.hops += 1 
            print(f"Node {node} forwarding {transaction.tx_hash} in stem phase (hop={transaction.hops})")

            neighbor_id = random.choice(list(network.neighbors(node))) 
            if need_to_repackage(transaction, network.nodes[node], neighbor_id):
                # ... (Repackaging logic: probability check, repackaging, probability update) ...

                # Retrieve probability
                repackage_probability = network.nodes[node]['repackage_probabilities'][sender]['probability']  
                # Randomness check 
                if random.random() < repackage_probability: 
                    target_difficulty = network.nodes[neighbor_id]['txPoW_mindiff'] # Get the neighbor's min_diff
                    repackage_transaction(transaction, target_difficulty)  
                    print(f"Node {node} repackaged and forwarding {transaction.tx_hash} in stem phase (hop={transaction.hops})")

                # Update probability and timestamp (regardless of repackaging)

                last_transaction_time = network.nodes[node]['repackage_probabilities'].get(sender, {}).get('last_transaction_time', 0.0)
                repackage_probability = update_repackage_probability(last_transaction_time)

                network.nodes[node]['repackage_probabilities'][sender]['probability'] = repackage_probability # Update the probability
                network.nodes[node]['repackage_probabilities'].setdefault(sender, {})['last_transaction_time'] = env.now # Update timestamp

            # Update timestamp


            # The following is for if the transacti
            network.nodes[node]['repackage_probabilities'].setdefault(neighbor_id, {})['last_transaction_time'] = env.now

            env.process(process_transaction(env, neighbor_id, network, copy.deepcopy(transaction))) 

        else:  # Fluff Phase
            #print(f"Node {node} forwarding {transaction.tx_hash} in fluff phase")
            for neighbor_id in network.neighbors(node):  
                if need_to_repackage(transaction, network.nodes[node], neighbor_id):
                    # ... (Repackaging logic: probability check, repackaging, probability update) ...

                    # Retrieve probability
                    repackage_probability = network.nodes[node]['repackage_probabilities'][sender]['probability']  
                    # Randomness check 
                    print (f"repackage_probability at line 201, in fluff: {repackage_probability}")
 
                    if random.random() < repackage_probability: 
                        target_difficulty = network.nodes[neighbor_id]['txPoW_mindiff'] # Get the neighbor's min_diff
                        repackage_transaction(transaction, target_difficulty)  
                        print(f"Node {node} repackaged and forwarding {transaction.tx_hash} in fluff phase")

                    # Update probability and timestamp (regardless of repackaging)
                    #network.nodes[node]['repackage_probabilities'].setdefault(neighbor, {})['last_transaction_time'] = env.now
                    #repackage_probability = update_repackage_probability(
                    #network.nodes[node]['repackage_probabilities'].get(sender, {}).get('last_transaction_time', 0.0))


                    last_transaction_time = network.nodes[node]['repackage_probabilities'].get(sender, {}).get('last_transaction_time', 0.0)
                    repackage_probability = update_repackage_probability(last_transaction_time)

                    network.nodes[node]['repackage_probabilities'][sender]['probability'] = repackage_probability # Update the probability
                    network.nodes[node]['repackage_probabilities'].setdefault(sender, {})['last_transaction_time'] = env.now # Update timestamp


                # Update timestamp for the current neighbor
                network.nodes[node]['repackage_probabilities'].setdefault(neighbor_id, {})['last_transaction_time'] = env.now
                #print(f"Node {node} forwarding {transaction.tx_hash} in fluff phase")

                env.process(process_transaction(env, neighbor_id, network, copy.deepcopy(transaction)))  







def is_in_stem_phase(transaction, node, network):
    return transaction.hops < transaction.hop_limit  # Use the stored hop limit


def calculate_transaction_rate(timestamps):
    """Calculates the transaction rate in transactions per second based on provided timestamps.

    Args:
        timestamps: A list of timestamps representing when transactions were received.

    Returns:
        float: The transaction rate in transactions per second.
    """

    if len(timestamps) < 2:  # Need at least two timestamps for a meaningful rate
        return 0.0

    time_window = 1.0  # Consider transactions within the last 1 second
    cutoff_time = env.now - time_window

    recent_transactions = sum(timestamp >= cutoff_time for timestamp in timestamps)
    transaction_rate = recent_transactions / time_window 

    return transaction_rate 

def calculate_new_min_diff(current_min_diff):
  """Calculates a new minimum difficulty based on the current value.

  Args:
      current_min_diff: The current minimum difficulty requirement.

  Returns:
      float: The new minimum difficulty requirement.
  """

  # Increase the difficulty by squaring the current value
  new_min_diff = current_min_diff * current_min_diff

  return new_min_diff


def need_to_repackage(transaction, node_data, neighbor_id):
    """Determines whether a transaction needs to be repackaged.

    Args:
        transaction: The transaction object.
        node_data: The data associated with the processing node (from G.nodes[node]).
        neighbor_id: The ID of the neighbor the transaction needs to be relayed to.

    Returns:
        bool: True if repackaging is needed, False otherwise.
    """

    neighbor_min_diff = node_data['neighbors'][neighbor_id]['txPoW_mindiff']  

    if neighbor_min_diff > MAX_POW_DIFF:
        print ("DIFF TOO HIGH IN REPACKAGE")
        return False  # Rejection if mindiff is too high

    return transaction.tx_pow < neighbor_min_diff 


def repackage_transaction(transaction, target_difficulty):
    """Assigns a new simulated PoW value to the transaction.

    Args:
        transaction: The transaction object to be repackaged.
        target_difficulty: The target difficulty to meet.
    """

    transaction.tx_pow = target_difficulty  # Directly assign the new PoW value


def update_repackage_probability(last_transaction_time=None):
  """
  Updates the repackaging probability based on the time since the last transaction,
  incorporating decay over time.

  Args:
      last_transaction_time: Timestamp of the last transaction received from this neighbor.

  Returns:
      float: The updated repackaging probability (between 0.0 and 1.0).
  """

  time_now = env.now
  time_since_transaction = time_now - last_transaction_time if last_transaction_time is not None else None

  if time_since_transaction is not None and time_since_transaction > 4 * 60:  # 4 minutes in seconds
      # Reset probability (increase)
      return 1.0  # Maximum repackaging probability
  else:
      # Decrease probability over time (decay)
      current_probability = 0.0  # Start with low probability 
      return max(current_probability - 0.1, 0.0)  # Decay, minimum of 0.0



def visualize_network(G):
    """Visualizes the given NetworkX graph."""
    pos = nx.spring_layout(G)  # Use a layout algorithm for positioning nodes
    nx.draw(G, pos, with_labels=True, node_size=500)
    plt.show()



#################################################################


# Create a SimPy environment
env = simpy.Environment()

# Example: Generate a network of 100 nodes
network = generate_mesh_network(2000)

MAX_POW_DIFF = 100  # You can adjust this value as needed

# Visualize the network (optional)
#visualize_network(network)

# Example: Have node 0 generate transactions every so often
#env.process(generate_transaction(env, 0, network))

# Example: Have multiple nodes generate transactions

num_transactions_generated = 0  # Add this line before your simulation loop. NOT A SETTABLE VARIABLE
for node in range(1000):  # Adjust the number of generating nodes
    env.process(generate_transaction(env, node, network))

# Run the simulation for some time
env.run(until=100)



# Analysis
analysis_data = []
total_unique_transactions = set()

for node in network.nodes():
    num_unique_seen = len(network.nodes[node]['seen_transactions'])
    analysis_data.append({'node': node, 'unique_transactions_seen': num_unique_seen})
    total_unique_transactions.update(network.nodes[node]['seen_transactions'])  # Update global set


# Analysis
analysis_data = []
total_unique_transactions = set()

for node in network.nodes():
    num_unique_seen = len(network.nodes[node]['seen_transactions'])
    analysis_data.append({'node': node, 'unique_transactions_seen': num_unique_seen})
    total_unique_transactions.update(network.nodes[node]['seen_transactions'])  # Update global set

# Calculate any overall metrics
total_transactions_created = len(total_unique_transactions) 
dropped_percentage = (1 - (total_transactions_created / num_transactions_generated)) * 100  # Corrected calculation


# Print a table 
print("Transaction Propagation Analysis:")
print("-----------------------------")
print(f"Total unique transactions generated: {total_transactions_created}")
print(f"Network-wide dropping percentage (approx): {dropped_percentage:.1f}%")
print("Per-node statistics:")
print("Node ID | Unique Transactions Seen")
#for data in analysis_data:
#    print(f"{data['node']} | {data['unique_transactions_seen']}")


# Visualization
#pos = nx.spring_layout(network)  # Use a layout algorithm for positioning nodes
#nx.draw(network, pos, with_labels=True, node_size=500)
#plt.show()
