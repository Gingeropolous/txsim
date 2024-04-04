import simpy
import networkx as nx
import matplotlib.pyplot as plt
import random
import time
import copy
import networkx as nx
import matplotlib.pyplot as plt



class Transaction:
    def __init__(self, originator, timestamp, tx_hash, tx_pow):
        self.originator = originator
        self.timestamp = timestamp
        self.tx_hash = tx_hash
        self.tx_pow = tx_pow  # Newly added field
        self.hops = 0  # Initialize hops to 0
        self.hop_limit = 0  # Initialize the hop limit

    def __str__(self): 
        return f"Transaction(originator={self.originator}, timestamp={self.timestamp}, txhash={self.tx_hash}, tx_pow={self.tx_pow})"



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
        G.nodes[node]['txPoW_mindiff'] = 5  
        G.nodes[node]['seen_transactions'] = set()  
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
            transaction_hash = hash(str(node) + str(env.now))  # Generate hash based on node and time
            tx = Transaction(node, env.now, transaction_hash, network.nodes[node]['txPoW_mindiff'])  # Valid PoW
            tx.hop_limit = random_hops  # Add the hop limit to the Transaction object
        else:  # Spammer behavior
            # Spammer behavior (adjust these parameters)
            yield env.timeout(random.expovariate(0.5))  # More frequent transactions
            # Select a random neighbor from the pre-defined list of spam_targets
            neighbor = random.choice(network.nodes[node]['spam_targets'])
            very_low_difficulty=4
            transaction_hash = hash(str(node) + str(env.now))  # Generate hash based on node and time
            tx = Transaction(node, env.now, transaction_hash, very_low_difficulty)  # Invalid PoW (low difficulty)
            tx.hop_limit = random_hops  # Add the hop limit to the Transaction object

        # You'll need to implement logic to actually create and send the transaction here
        # (considering PoW checks in the future)
        print(f"Node {node} trying to generate a transaction {tx} (user type: {user_type})")
        num_transactions_generated += 1  # Increment the counter after a transaction is generated
  # ... (Transaction creation with stem phase flag) ...


    # Propagation Logic - Basic
        #for neighbor in network.neighbors(node):  # Broadcast to all neighbors
         #   #print(f"Node {node} sending transaction to neighbor {neighbor}")
          #  env.process(process_transaction(env, neighbor, network, copy.deepcopy(tx)))

    # Dandelion Stem Phase: Send to a single random neighbor
        neighbor = random.choice(list(network.neighbors(node)))
        env.process(process_transaction(env, neighbor, network, copy.deepcopy(tx)))


def process_transaction(env, node, network, transaction):
    delay = random.uniform(0.005, 0.02)  # Simulate processing and transmission time
    yield env.timeout(delay) 
    #print(f"Node {node} received transaction from originator {transaction.originator}")

    # Track the transaction 
#    if transaction.tx_hash not in network.nodes[node]['seen_transactions']:
#        network.nodes[node]['seen_transactions'].add(transaction.tx_hash)
        #print(f"Node {node} received NEW transaction from originator {transaction.originator}")

    # Dandelion Propagation Logic 
    if transaction.tx_hash not in network.nodes[node]['seen_transactions']:
         network.nodes[node]['seen_transactions'].add(transaction.tx_hash)
        #print(f"Node {node} received NEW transaction from originator {transaction.originator}")

         if is_in_stem_phase(transaction, node, network):  # Implement stem phase logic
             transaction.hops += 1  # Increment the hop count
             print(f"Node {node} forwarding {transaction.tx_hash} in stem phase (hop={transaction.hops})")
             neighbor = random.choice(list(network.neighbors(node)))
             env.process(process_transaction(env, neighbor, network, copy.deepcopy(transaction)))
         else:  # Fluff Phase
             #print(f"Node {node} forwarding {transaction.tx_hash} in fluff phase")
             for neighbor in network.neighbors(node):
                 env.process(process_transaction(env, neighbor, network, copy.deepcopy(transaction)))


        # Relay the transaction to neighbors (Corrected)
#        for neighbor in network.neighbors(node):
#            env.process(process_transaction(env, neighbor, network, copy.deepcopy(transaction)))  # Pass the 'transaction' object 
#    else:
        #print(f"Node {node} received DUPLICATE transaction from originator {transaction.originator}")


#def is_in_stem_phase(transaction, node, network):
#    min_stem_hops = 4  # Minimum number of hops in stem phase
#    max_stem_hops = 9  # Maximum number of hops in stem phase
#    random_hops = random.randint(min_stem_hops, max_stem_hops) 
#    return transaction.hops < random_hops

def is_in_stem_phase(transaction, node, network):
    return transaction.hops < transaction.hop_limit  # Use the stored hop limit


def visualize_network(G):
    """Visualizes the given NetworkX graph."""
    pos = nx.spring_layout(G)  # Use a layout algorithm for positioning nodes
    nx.draw(G, pos, with_labels=True, node_size=500)
    plt.show()



#################################################################


# Create a SimPy environment
env = simpy.Environment()

# ... (Network generation code here) ...

# Example: Generate a network of 100 nodes
network = generate_mesh_network(1000)

# Visualize the network (optional)
#visualize_network(network)

# Example: Have node 0 generate transactions every so often
env.process(generate_transaction(env, 0, network))

# Example: Have multiple nodes generate transactions

num_transactions_generated = 0  # Add this line before your simulation loop
for node in range(400):  # Adjust the number of generating nodes
    env.process(generate_transaction(env, node, network))

# Run the simulation for some time
env.run(until=100000000000000)



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
