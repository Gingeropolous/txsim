# txsim

The ultimate goal is to build a simulation script so that I can test whether incorporating a proof of work for transaction relay is capable of these things:

1. Preventing spam attacks. Currently in the network, a malicious user can send 10s of thousands of transactions in very short time interval. Packing a transaction with a PoW ensures that a spammer will be limited due to computation resources. This will mitigate spam attacks.  
2. Allowing users to still transact during spam attacks. A user without a lot of processing power must still be able to send a transaction during a spam attack. The ability of a node to re-package a transaction with a new PoW (in order to assist a transaction through the network) may allow average users to send transactions during a spam attack. This is what we are trying to figure out. 
